from argparse import Namespace
from typing import List, Tuple

import aislib.pytorch as torch_utils
import torch
from aislib.misc_utils import get_logger
from torch import nn

from . import embeddings
from .embeddings import al_emb_lookup_dict
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
        self.act_1 = nn.ReLU()

        self.bn_1 = nn.BatchNorm2d(in_channels)
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

        self.act_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm2d(out_channels)
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

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class FirstBlock(AbstractBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "act_1")
        delattr(self, "act_2")
        delattr(self, "downsample_identity")
        delattr(self, "bn_1")
        delattr(self, "conv_2")
        delattr(self, "bn_2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv_1(x)
        out = self.rb_do(out)

        return out


class Block(AbstractBlock):
    def __init__(self, full_preact: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.full_preact = full_preact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn_1(x)
        out = self.act_1(out)

        if self.full_preact:
            identity = self.downsample_identity(out)
        else:
            identity = self.downsample_identity(x)

        out = self.conv_1(out)

        out = self.bn_2(out)
        out = self.act_2(out)

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
    residual_blocks: List[int], run_config: Namespace
) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param run_config: Experiment hyperparameters / configuration needed for the
    convolution setup.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    rc = run_config

    down_stride_w = rc.down_stride

    first_conv_kernel = rc.kernel_width * rc.first_kernel_expansion
    first_conv_stride = down_stride_w * rc.first_stride_expansion
    first_kernel, first_pad = set_up_conv_params(
        rc.target_width, first_conv_kernel, first_conv_stride
    )

    base_layers = [
        FirstBlock(
            in_channels=1,
            out_channels=2 ** rc.channel_exp_base,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=down_stride_w,
            rb_do=rc.rb_do,
        )
    ]

    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_conv = nn.Sequential(*base_layers)
            cur_width = torch_utils.calc_size_after_conv_sequence(
                rc.target_width, cur_conv
            )

            cur_kern, cur_padd = set_up_conv_params(
                cur_width, rc.kernel_width, down_stride_w
            )

            cur_in_channels = base_layers[-1].out_channels
            cur_out_channels = 2 ** (rc.channel_exp_base + layer_arch_idx)
            cur_layer = Block(
                in_channels=cur_in_channels,
                out_channels=cur_out_channels,
                conv_1_kernel_w=cur_kern,
                conv_1_padding=cur_padd,
                down_stride_w=down_stride_w,
                full_preact=True if len(base_layers) == 1 else False,
                rb_do=rc.rb_do,
            )

            base_layers.append(cur_layer)

    # attention_channels = base_layers[-2].out_channels
    # base_layers.insert(-1, SelfAttention(attention_channels))
    return base_layers


class Model(nn.Module):
    def __init__(
        self,
        run_config: Namespace,
        num_classes: int,
        embeddings_dict: al_emb_lookup_dict = None,
        extra_continuous_inputs: Tuple[str, ...] = None,
    ):
        super().__init__()

        self.run_config = run_config
        self.num_classes = num_classes
        self.embeddings_dict = embeddings_dict
        self.extra_continuous_inputs = extra_continuous_inputs

        emb_total_dim = con_total_dim = 0
        if embeddings_dict:
            emb_total_dim = embeddings.attach_embeddings(self, embeddings_dict)
        if extra_continuous_inputs:
            con_total_dim = len(self.extra_continuous_inputs)

        self.conv = nn.Sequential(*make_conv_layers(self.resblocks, run_config))

        self.data_size_after_conv = torch_utils.calc_size_after_conv_sequence(
            run_config.target_width, self.conv
        )

        self.no_out_channels = self.conv[-1].out_channels

        fc_1_in_features = self.data_size_after_conv * self.no_out_channels
        fc_base = run_config.fc_dim

        self.fc_1 = nn.Sequential(
            nn.BatchNorm1d(fc_1_in_features),
            nn.ReLU(),
            nn.Linear(fc_1_in_features, fc_base, bias=False),
        )

        if emb_total_dim or con_total_dim:
            extra_dim = emb_total_dim + con_total_dim
            self.fc_extra = nn.Linear(extra_dim, extra_dim, bias=False)
            fc_base += extra_dim

        self.fc_2 = nn.Sequential(
            nn.BatchNorm1d(fc_base), nn.ReLU(), nn.Linear(fc_base, fc_base, bias=False)
        )

        self.fc_3 = nn.Sequential(
            nn.BatchNorm1d(fc_base),
            nn.ReLU(),
            nn.Dropout(run_config.fc_do),
            nn.Linear(fc_base, self.num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, extra_inputs: torch.Tensor = None):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)

        out = self.fc_1(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = self.fc_2(out)
        out = self.fc_3(out)

        return out

    @property
    def resblocks(self):
        if not self.run_config.resblocks:
            residual_blocks = find_no_resblocks_needed(
                self.run_config.target_width,
                self.run_config.down_stride,
                self.run_config.first_stride_expansion,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.run_config.resblocks
