import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth

from eir.models.layers.attention_layers import LinearAttention
from eir.models.layers.norm_layers import GRN
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_down = nn.Conv2d(
            in_channels=channels,
            out_channels=reduced_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.act_1 = nn.GELU()

        self.conv_up = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)

        out = self.conv_down(out)
        out = self.act_1(out)

        out = self.conv_up(out)
        out = self.sigmoid(out)

        return out


def get_eca_kernel_size(
    channels: int,
    gamma: int = 2,
    b: int = 1,
) -> int:
    t = int(abs(math.log2(channels) / gamma + b / gamma))
    return t if t % 2 else t + 1


class ECABlock(nn.Module):
    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        b: int = 1,
    ):
        super().__init__()

        kernel_size = get_eca_kernel_size(
            channels=channels,
            gamma=gamma,
            b=b,
        )
        padding = (kernel_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # [B, C, 1, 1]

        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)

        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)

        return x * y


class ConvAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        num_heads: int = 4,
        dropout_p: float = 0.1,
        attention_mode: Literal["spatial", "channel"] = "spatial",
        attention_type: Literal["full", "linear"] = "full",
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.in_height = height
        self.in_width = width
        self.attention_mode = attention_mode
        self.attention_type = attention_type

        self.embedding_dim = channels if attention_mode == "spatial" else height * width
        self.num_heads = adjust_num_heads(
            num_heads=num_heads,
            embedding_dim=self.embedding_dim,
        )

        self.attention: nn.MultiheadAttention | LinearAttention
        if attention_type == "full":
            self.norm = nn.RMSNorm(normalized_shape=self.embedding_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=dropout_p,
            )
        elif attention_type == "linear":
            self.norm = nn.RMSNorm(normalized_shape=[channels, height, width])
            self.attention = LinearAttention(
                embed_dim=self.embedding_dim,
                heads=self.num_heads,
                dim_head=self.embedding_dim // self.num_heads,
            )

        self.grn = GRN(in_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        if self.attention_type == "full":
            if self.attention_mode == "spatial":
                out = rearrange(x, "b c h w -> b (h w) c")
            elif self.attention_mode == "channel":
                out = rearrange(x, "b c h w -> b (h w) c").permute(0, 2, 1)
            else:
                raise ValueError()

            out = self.norm(out)
            attn_output, _ = self.attention(out, out, out)
            out = out + attn_output

            if self.attention_mode == "spatial":
                out = rearrange(out, "b (h w) c -> b c h w", h=height, w=width)
            elif self.attention_mode == "channel":
                out = rearrange(out, "b c (h w) -> b c h w", h=height, w=width)
            else:
                raise ValueError()

        elif self.attention_type == "linear":
            assert self.attention_mode == "spatial"
            out = self.norm(x)
            attn_output = self.attention(x)
            out = out + attn_output

        else:
            raise ValueError()

        out = self.grn(out)

        return x + out


def adjust_num_heads(num_heads: int, embedding_dim: int) -> int:
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
        conv_downsample_identity: bool = True,
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

        self.conv_downsample_identity = conv_downsample_identity

        self.rb_do = nn.Dropout2d(rb_do)

        self.conv_ds = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
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
            dilation=(self.dilation_h, self.dilation_w),
            bias=False,
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
            bias=False,
        )

        self.downsample_identity: nn.Module = nn.Identity()
        if self.conv_downsample_identity:
            self.downsample_identity = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(self.conv_1_kernel_h, self.conv_1_kernel_w),
                stride=(self.down_stride_h, self.down_stride_w),
                padding=(self.conv_1_padding_h, self.conv_1_padding_w),
                dilation=(self.dilation_h, self.dilation_w),
                bias=True,
            )

        self.stochastic_depth = StochasticDepth(p=self.stochastic_depth_p, mode="batch")

        self.eca_block = ECABlock(channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


def _compute_conv_2_parameters(
    conv_1_kernel_size: int,
    dilation: int,
) -> tuple[int, int]:
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
        delattr(self, "eca_block")
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

        channel_recalibrations = self.eca_block(out)
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
        """
        Note: Slightly different approach here compared to the upsampling below,
        here we do the downsampling in one-go with a strided convolution.
        """
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
            bias=False,
        )

        self.identity = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=(self.stride_h, self.stride_w),
            padding=1,
            bias=False,
        )

        self.grn = GRN(in_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)

        out = self.norm_1(x)
        out = self.act_1(out)
        out = self.conv_1(out)
        out = self.grn(out)

        return out + identity


def _compute_params_for_down_sampling(
    cur_height: int, cur_width: int
) -> tuple[int, int]:
    stride_h = 1 if cur_height == 1 else 2

    stride_w = 1 if cur_width == 1 else 2

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
        """
        Note: Always applying a Conv to the upsampled identity seems to help
        stabilize training.
        """

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

        self.upsample = nn.Upsample(
            scale_factor=(self.stride_h, self.stride_w),
            mode="bilinear",
            align_corners=None,
        )

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.act_1 = nn.GELU()

        self.conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.identity = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.grn = GRN(in_channels=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(x)
        identity = self.identity(upsampled)

        out = upsampled
        out = self.norm_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)
        out = self.grn(out)

        return out + identity


def _compute_params_for_up_sampling(
    upsample_height: bool,
    upsample_width: bool,
):
    stride_h = 2 if upsample_height else 1
    stride_w = 2 if upsample_width else 1
    return stride_h, stride_w
