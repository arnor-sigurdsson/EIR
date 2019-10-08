from typing import List, Dict

import aislib.pytorch as torch_utils
import torch
from aislib.misc_utils import get_logger
from torch import nn

from . import embeddings
from .model_utils import find_no_resblocks_needed

logger = get_logger(__name__)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1
        )

        self.key_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1
        )

        self.value_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y=None):
        batch_size, channels, width, height = x.size()

        proj_query = self.query_conv(x)
        proj_query = proj_query.view(batch_size, -1, width * height).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key = proj_key.view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        attention = attention.permute(0, 2, 1)

        proj_value = self.value_conv(x)
        proj_value = proj_value.view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention)
        out = out.view(batch_size, channels, width, height)

        out = self.gamma * out + x
        return out


class AbstractBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rb_do: float,
        conv_1_kernel_w: int = 12,
        conv_1_padding: int = 4,
        down_stride_w: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1_kernel_w = conv_1_kernel_w
        self.conv_1_padding = conv_1_padding
        self.down_stride_w = down_stride_w

        self.conv_1_kernel_h = 4 if isinstance(self, FirstBlock) else 1
        self.down_stride_h = self.conv_1_kernel_h

        self.rb_do = nn.Dropout2d(rb_do)
        self.act = nn.LeakyReLU()

        self.bn_1 = nn.BatchNorm2d(in_channels, out_channels)
        self.conv_1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(self.conv_1_kernel_h, conv_1_kernel_w),
            stride=(self.down_stride_h, down_stride_w),
            padding=(0, conv_1_padding),
            bias=False,
        )

        conv_2_kernel_w = (
            conv_1_kernel_w - 1 if conv_1_kernel_w % 2 == 0 else conv_1_kernel_w
        )
        conv_2_padding = conv_2_kernel_w // 2

        self.bn_2 = nn.BatchNorm2d(out_channels, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, conv_2_kernel_w),
            stride=(1, 1),
            padding=(0, conv_2_padding),
            bias=False,
        )

        self.downsample_identity = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(self.conv_1_kernel_h, conv_1_kernel_w),
                stride=(self.down_stride_h, down_stride_w),
                padding=(0, conv_1_padding),
                bias=False,
            )
        )

    def forward(self, x):
        raise NotImplementedError


class FirstBlock(AbstractBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "act")
        delattr(self, "downsample_identity")
        delattr(self, "bn_1")
        delattr(self, "conv_2")
        delattr(self, "bn_2")

    def forward(self, x):

        out = self.conv_1(x)
        out = self.rb_do(out)

        return out


class Block(AbstractBlock):
    def __init__(self, full_preact=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.full_preact = full_preact

    def forward(self, x):
        out = self.bn_1(x)
        out = self.act(out)

        if self.full_preact:
            identity = self.downsample_identity(out)
        else:
            identity = self.downsample_identity(x)

        out = self.conv_1(out)

        out = self.bn_2(out)
        out = self.act(out)

        out = self.rb_do(out)
        out = self.conv_2(out)

        out = out + identity

        return out


def set_up_conv_params(current_width: int, kernel_size: int, stride: int):
    if current_width % 2 != 0:
        kernel_size -= 1

    padding = torch_utils.calc_conv_padding_needed(current_width, kernel_size, stride)

    return kernel_size, padding


def make_conv_layers(
    residual_blocks: List[int],
    kernel_base_width: int,
    channel_exp_base: int,
    input_width: int,
    rb_do: float,
) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param kernel_base_width: Width of the kernels applied during convolutions.
    :param channel_exp_base: Number of channels in first layer (2 in the power of).
    :param input_width: Used to calculate convolutional parameters.
    :param rb_do: Percentage of dropout to pass to blocks.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    down_stride_w = 4
    first_kernel, first_pad = set_up_conv_params(
        input_width, kernel_base_width, down_stride_w
    )
    base_layers = [
        FirstBlock(
            in_channels=1,
            out_channels=2 ** channel_exp_base,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=down_stride_w,
            rb_do=rb_do,
        )
    ]

    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_conv = nn.Sequential(*base_layers)
            cur_width = torch_utils.calc_size_after_conv_sequence(input_width, cur_conv)

            cur_kern, cur_padd = set_up_conv_params(
                cur_width, kernel_base_width, down_stride_w
            )

            cur_in_channels = base_layers[-1].out_channels
            cur_out_channels = 2 ** (channel_exp_base + layer_arch_idx)
            cur_layer = Block(
                in_channels=cur_in_channels,
                out_channels=cur_out_channels,
                conv_1_kernel_w=cur_kern,
                conv_1_padding=cur_padd,
                down_stride_w=down_stride_w,
                full_preact=True if len(base_layers) == 1 else False,
                rb_do=rb_do,
            )

            base_layers.append(cur_layer)

    # attention_channels = base_layers[-2].out_channels
    # base_layers.insert(-1, SelfAttention(attention_channels))
    return base_layers


def calc_extra_embed_dim(embeddings_dict):
    base = 0
    if not embeddings_dict:
        return base

    for embed_col in embeddings_dict:
        cur_embedding = embeddings_dict[embed_col]["embedding_module"]
        base += cur_embedding.embedding_dim

    return base


class Model(nn.Module):
    def __init__(self, run_config, num_classes, embeddings_dict=None):
        super().__init__()

        self.run_config = run_config
        self.num_classes = num_classes
        self.embeddings_dict = embeddings_dict

        emb_total_dim = 0
        if embeddings_dict:
            emb_total_dim = embeddings.attach_embeddings(self, embeddings_dict)

        self.conv = nn.Sequential(
            *make_conv_layers(
                self.resblocks,
                run_config.kernel_width,
                run_config.channel_exp_base,
                run_config.target_width,
                run_config.rb_do,
            )
        )

        self.data_size_after_conv = torch_utils.calc_size_after_conv_sequence(
            run_config.target_width, self.conv
        )

        self.no_out_channels = self.conv[-1].out_channels

        fc_1_in_features = self.data_size_after_conv * self.no_out_channels
        fc_base = 128

        self.fc_1 = nn.Sequential(
            nn.BatchNorm1d(fc_1_in_features),
            nn.LeakyReLU(),
            nn.Linear(fc_1_in_features, fc_base, bias=False),
        )

        if emb_total_dim:
            self.fc_e = nn.Linear(emb_total_dim, emb_total_dim, bias=False)
            fc_base += emb_total_dim

        self.fc_2 = nn.Sequential(
            nn.BatchNorm1d(fc_base),
            nn.LeakyReLU(),
            nn.Linear(fc_base, fc_base, bias=False),
        )

        self.fc_3 = nn.Sequential(
            nn.BatchNorm1d(fc_base),
            nn.LeakyReLU(),
            nn.Dropout(run_config.fc_do),
            nn.Linear(fc_base, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, extra_labels: List[Dict[str, str]] = None):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)

        out = self.fc_1(out)

        if extra_labels:
            all_embeddings = []
            for col_key in self.embeddings_dict:
                cur_embedding = embeddings.lookup_embeddings(
                    self,
                    self.embeddings_dict,
                    col_key,
                    extra_labels,
                    self.run_config.device,
                )
                all_embeddings.append(cur_embedding)

            all_embeddings = torch.cat(all_embeddings, dim=1)
            out_e = self.fc_e(all_embeddings)
            out = torch.cat((out_e, out), dim=1)

        out = self.fc_2(out)
        out = self.fc_3(out)
        return out

    @property
    def resblocks(self):
        if not self.run_config.resblocks:
            residual_blocks = find_no_resblocks_needed(self.run_config.target_width, 4)
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.run_config.resblocks
