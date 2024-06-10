from typing import Tuple

import torch
from aislib.misc_utils import get_logger
from torch import nn
from torchvision.ops import StochasticDepth

from eir.models.layers.norm_layers import GRN

logger = get_logger(name=__name__)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_down = nn.Conv2d(
            in_channels=channels,
            out_channels=reduced_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.act_1 = nn.GELU()

        self.conv_up = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)

        out = self.conv_down(out)
        out = self.act_1(out)

        out = self.conv_up(out)
        out = self.sigmoid(out)

        return out


class ConvAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        num_heads: int = 4,
        dropout_p: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.in_height = height
        self.in_width = width

        self.embedding_dim = height * width
        self.num_heads = _adjust_num_heads(
            num_heads=num_heads, embedding_dim=self.embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embedding_dim * 4,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=dropout_p,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.grn = GRN(in_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(-1, self.in_channels, self.in_height * self.in_width)
        out = self.encoder(out)
        out = out.view(-1, self.in_channels, self.in_height, self.in_width)
        out = self.grn(out)

        return x + out


def _adjust_num_heads(num_heads: int, embedding_dim: int) -> int:
    while embedding_dim % num_heads != -0:
        num_heads -= 1

    logger.debug(
        f"Adjusted base number of heads to {num_heads} "
        f"according to embedding dim {embedding_dim}."
    )
    return num_heads


class CNNResidualBlockBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rb_do: float,
        dilation_w: int,
        dilation_h: int,
        conv_1_kernel_h: int = 1,
        conv_1_kernel_w: int = 12,
        conv_1_padding_w: int = 4,
        conv_1_padding_h: int = 4,
        down_stride_w: int = 4,
        down_stride_h: int = 1,
        stochastic_depth_p: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1_kernel_h = conv_1_kernel_h
        self.conv_1_kernel_w = conv_1_kernel_w

        self.dilation_w = dilation_w
        self.dilation_h = dilation_h

        self.conv_1_padding_w = conv_1_padding_w
        self.conv_1_padding_h = conv_1_padding_h

        self.down_stride_w = down_stride_w
        self.down_stride_h = down_stride_h

        self.stochastic_depth_p = stochastic_depth_p

        self.rb_do = nn.Dropout2d(rb_do)

        self.conv_ds = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=in_channels,
        )

        self.act_1 = nn.GELU()

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.conv_1_kernel_h, self.conv_1_kernel_w),
            stride=(self.down_stride_h, self.down_stride_w),
            padding=(self.conv_1_padding_h, self.conv_1_padding_w),
            bias=True,
        )

        conv_2_kernel_h, conv_2_padding_h = _compute_conv_2_parameters(
            conv_1_kernel_size=conv_1_kernel_h, dilation=dilation_h
        )

        conv_2_kernel_w, conv_2_padding_w = _compute_conv_2_parameters(
            conv_1_kernel_size=conv_1_kernel_w, dilation=dilation_w
        )

        self.grn = GRN(in_channels=out_channels)

        self.conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(conv_2_kernel_h, conv_2_kernel_w),
            stride=(1, 1),
            padding=(conv_2_padding_h, conv_2_padding_w),
            dilation=(dilation_h, dilation_w),
            bias=True,
        )

        self.downsample_identity = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(self.conv_1_kernel_h, self.conv_1_kernel_w),
                stride=(self.down_stride_h, self.down_stride_w),
                padding=(self.conv_1_padding_h, self.conv_1_padding_w),
                bias=True,
            )
        )

        self.stochastic_depth = StochasticDepth(p=self.stochastic_depth_p, mode="batch")

        self.se_block = SEBlock(channels=out_channels, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


def _compute_conv_2_parameters(
    conv_1_kernel_size: int,
    dilation: int,
) -> Tuple[int, int]:
    conv_2_kernel = (
        conv_1_kernel_size - 1 if conv_1_kernel_size % 2 == 0 else conv_1_kernel_size
    )

    conv_2_padding = conv_2_kernel // 2
    conv_2_padding = conv_2_padding * dilation

    return conv_2_kernel, conv_2_padding


class FirstCNNBlock(CNNResidualBlockBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "conv_ds")
        delattr(self, "norm_1")
        delattr(self, "downsample_identity")
        delattr(self, "act_1")
        delattr(self, "grn")
        delattr(self, "rb_do")
        delattr(self, "conv_2")
        delattr(self, "se_block")
        delattr(self, "stochastic_depth")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)

        return out


class CNNResidualBlock(CNNResidualBlockBase):
    def __init__(self, full_preact: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.full_preact = full_preact

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv_ds(x)

        out = self.norm_1(out)

        if self.full_preact:
            identity = self.downsample_identity(out)
        else:
            identity = self.downsample_identity(x)

        out = self.conv_1(out)

        out = self.act_1(out)
        out = self.grn(out)
        out = self.rb_do(out)
        out = self.conv_2(out)

        channel_recalibrations = self.se_block(out)
        out = out * channel_recalibrations

        out = self.stochastic_depth(out)

        out = out + identity

        return out


class DownSamplingResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.stride_h, self.stride_w = _compute_params_for_down_sampling(
            cur_height=in_height,
            cur_width=in_width,
        )

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act_1 = nn.GELU()

        self.conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=(self.stride_h, self.stride_w),
            padding=1,
            bias=True,
        )

        self.identity = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=(self.stride_h, self.stride_w),
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)

        out = self.norm_1(x)
        out = self.act_1(out)
        out = self.conv_1(out)

        out = out + identity

        return out


def _compute_params_for_down_sampling(
    cur_height: int, cur_width: int
) -> tuple[int, int]:

    if cur_height == 1:
        stride_h = 1
    else:
        stride_h = 2

    if cur_width == 1:
        stride_w = 1
    else:
        stride_w = 2

    return stride_h, stride_w


class UpSamplingResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        upsample_height: bool = True,
        upsample_width: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width
        self.upsample_height = upsample_height
        self.upsample_width = upsample_width

        self.stride_h, self.stride_w = _compute_params_for_up_sampling(
            upsample_height=upsample_height,
            upsample_width=upsample_width,
        )

        kernel_h = 4 if upsample_height else 3
        kernel_w = 4 if upsample_width else 3
        kernel_size = (kernel_h, kernel_w)

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act_1 = nn.GELU()

        self.conv_1 = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=(self.stride_h, self.stride_w),
            padding=1,
            bias=True,
        )

        self.identity = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=(self.stride_h, self.stride_w),
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)

        out = self.norm_1(x)
        out = self.act_1(out)
        out = self.conv_1(out)

        out = out + identity

        return out


def _compute_params_for_up_sampling(
    upsample_height: bool,
    upsample_width: bool,
):
    stride_h = 2 if upsample_height else 1
    stride_w = 2 if upsample_width else 1
    return stride_h, stride_w
