from argparse import Namespace
from typing import List, Union, Tuple

from aislib import pytorch_utils
import torch
from aislib.misc_utils import get_logger
from torch import nn

from . import embeddings
from .embeddings import al_emb_lookup_dict
from .model_utils import find_no_resblocks_needed

logger = get_logger(__name__)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.conv_theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
        )
        self.conv_phi = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 8,
            kernel_size=1,
            bias=False,
        )
        self.conv_g = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            bias=False,
        )
        self.conv_o = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AvgPool2d((1, 4), stride=(1, 4), padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1), True)

    def forward(self, x):
        _, ch, h, w = x.size()

        # Theta path
        theta = self.conv_theta(x)
        theta = theta.view(-1, ch // 8, h * w)

        # Phi path
        phi = self.conv_phi(x)
        phi = self.pool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)

        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)

        # g path
        g = self.conv_g(x)
        g = self.pool(g)
        g = g.view(-1, ch // 2, h * w // 4)

        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv_o(attn_g)

        # Out
        out = x + self.gamma * attn_g
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
    if current_width % 2 != 0 or stride == 1:
        kernel_size -= 1

    padding = pytorch_utils.calc_conv_padding_needed(
        current_width, kernel_size, stride, 1
    )

    return kernel_size, padding


def get_block(
    conv_blocks: List[nn.Module],
    layer_arch_idx: int,
    down_stride: int,
    cl_args: Namespace,
) -> Tuple[Block, int]:
    ca = cl_args

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width = pytorch_utils.calc_size_after_conv_sequence(ca.target_width, cur_conv)

    cur_kern, cur_padd = set_up_conv_params(cur_width, ca.kernel_width, down_stride)

    cur_in_channels = conv_blocks[-1].out_channels
    cur_out_channels = 2 ** (ca.channel_exp_base + layer_arch_idx)
    cur_layer = Block(
        in_channels=cur_in_channels,
        out_channels=cur_out_channels,
        conv_1_kernel_w=cur_kern,
        conv_1_padding=cur_padd,
        down_stride_w=down_stride,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=ca.rb_do,
    )

    return cur_layer, cur_width


def make_conv_layers(residual_blocks: List[int], cl_args: Namespace) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param cl_args: Experiment hyperparameters / configuration needed for the
    convolution setup.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    ca = cl_args

    down_stride_w = ca.down_stride

    first_conv_kernel = ca.kernel_width * ca.first_kernel_expansion
    first_conv_stride = down_stride_w * ca.first_stride_expansion
    first_kernel, first_pad = set_up_conv_params(
        ca.target_width, first_conv_kernel, first_conv_stride
    )

    conv_blocks = [
        FirstBlock(
            in_channels=1,
            out_channels=2 ** ca.channel_exp_base,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=first_conv_stride,
            rb_do=ca.rb_do,
        )
    ]

    sa_added = False
    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_layer, cur_width = get_block(
                conv_blocks, layer_arch_idx, down_stride_w, ca
            )

            if cl_args.sa and cur_width < 1024 and not sa_added:
                attention_channels = conv_blocks[-1].out_channels
                conv_blocks.append(SelfAttention(attention_channels))
                sa_added = True

            conv_blocks.append(cur_layer)

    return conv_blocks


class Model(nn.Module):
    def __init__(
        self,
        cl_args: Namespace,
        num_classes: int,
        embeddings_dict: Union[al_emb_lookup_dict, None] = None,
        extra_continuous_inputs_columns: Union[List[str], None] = None,
    ):
        super().__init__()

        self.cl_args = cl_args
        self.num_classes = num_classes
        self.embeddings_dict = embeddings_dict
        self.extra_continuous_inputs_columns = extra_continuous_inputs_columns

        emb_total_dim = con_total_dim = 0
        if embeddings_dict:
            emb_total_dim = embeddings.attach_embeddings(self, embeddings_dict)
        if extra_continuous_inputs_columns:
            con_total_dim = len(self.extra_continuous_inputs_columns)

        self.conv = nn.Sequential(*make_conv_layers(self.resblocks, cl_args))

        self.data_size_after_conv = pytorch_utils.calc_size_after_conv_sequence(
            cl_args.target_width, self.conv
        )

        self.no_out_channels = self.conv[-1].out_channels

        fc_1_in_features = self.data_size_after_conv * self.no_out_channels
        fc_base = cl_args.fc_dim

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
            nn.Dropout(cl_args.fc_do),
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
        if not self.cl_args.resblocks:
            residual_blocks = find_no_resblocks_needed(
                self.cl_args.target_width,
                self.cl_args.down_stride,
                self.cl_args.first_stride_expansion,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.cl_args.resblocks
