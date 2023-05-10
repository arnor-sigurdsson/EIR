import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn
from torch.nn import Parameter
from torchvision.ops.stochastic_depth import StochasticDepth

logger = get_logger(__name__)


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
        self.act_1 = Swish()

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
            encoder_layer=encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(-1, self.in_channels, self.in_height * self.in_width)
        out = self.encoder(out)
        out = out.view(-1, self.in_channels, self.in_height, self.in_width)
        return out


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
        self.act_1 = Swish()

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

        self.conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(conv_2_kernel_h, conv_2_kernel_w),
            stride=(1, 1),
            padding=(conv_2_padding_h * dilation_h, conv_2_padding_w * dilation_w),
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

        delattr(self, "norm_1")
        delattr(self, "downsample_identity")
        delattr(self, "act_1")
        delattr(self, "rb_do")
        delattr(self, "conv_2")
        delattr(self, "se_block")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)

        return out


class CNNResidualBlock(CNNResidualBlockBase):
    def __init__(self, full_preact: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.full_preact = full_preact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        if self.full_preact:
            identity = self.downsample_identity(out)
        else:
            identity = self.downsample_identity(x)

        out = self.conv_1(out)

        out = self.act_1(out)
        out = self.rb_do(out)
        out = self.conv_2(out)

        channel_recalibrations = self.se_block(out)
        out = out * channel_recalibrations

        out = self.stochastic_depth(out)

        out = out + identity

        return out


class SplitLinear(nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_feature_sets: int,
        num_chunks: int = 10,
        split_size: int = None,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_feature_sets = out_feature_sets
        self.num_chunks = num_chunks
        self.split_size = split_size

        if split_size:
            self.num_chunks = int(math.ceil(in_features / split_size))
            logger.debug(
                "%s: Setting num chunks to %d as split size of %d was passed in.",
                self.__class__,
                self.num_chunks,
                self.split_size,
            )
        else:
            self.split_size = int(math.ceil(self.in_features / self.num_chunks))
            logger.debug(
                "%s :Setting split size to %d as number of chunks of %d was passed in.",
                self.__class__,
                self.split_size,
                self.num_chunks,
            )

        self.out_features = self.out_feature_sets * self.num_chunks
        self.padding = _find_split_padding_needed(
            input_size=self.in_features,
            split_size=self.split_size,
            num_chunks=self.num_chunks,
        )

        self.weight = Parameter(
            torch.Tensor(self.out_feature_sets, self.num_chunks, self.split_size),
            requires_grad=True,
        )

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        NOTE: This default init actually works quite well, as compared to initializing
        for each chunk (meaning higher weights at init). In that case, the model takes
        longer to get to a good performance as it spends a while driving the weights
        down.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self):
        return (
            "in_features={}, num_chunks={}, split_size={}, "
            "out_feature_sets={}, out_features={}, bias={}".format(
                self.in_features,
                self.num_chunks,
                self.split_size,
                self.out_feature_sets,
                self.out_features,
                self.bias is not None,
            )
        )

    def forward(self, input: torch.Tensor):
        input_padded = F.pad(input=input, pad=[0, self.padding, 0, 0])

        input_reshaped = input_padded.reshape(
            input.shape[0], 1, self.num_chunks, self.split_size
        )

        out = calc_split_input(input=input_reshaped, weight=self.weight, bias=self.bias)
        return out


def _find_split_padding_needed(input_size: int, split_size: int, num_chunks: int):
    return num_chunks * split_size - input_size


def calc_split_input(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    n: num samples
    c: num chunks (height)
    s: split size (width)
    o: output sets
    """

    summed = torch.einsum("nhw, ohw -> noh", input.squeeze(1), weight)
    flattened = summed.flatten(start_dim=1)

    final = flattened
    if bias is not None:
        final = flattened + bias

    return final


@dataclass
class ResidualMLPConfig:
    """
    :param layers:
        Number of residual MLP layers to use in for each output predictor after fusing.

    :param fc_task_dim:
        Number of hidden nodes in each MLP residual block.

    :param rb_do:
        Dropout in each MLP residual block.

    :param fc_do:
        Dropout before final layer.

    :param stochastic_depth_p:
        Probability of dropping input.

    """

    layers: List[int] = field(default_factory=lambda: [2])

    fc_task_dim: int = 256

    rb_do: float = 0.10
    fc_do: float = 0.10

    stochastic_depth_p: float = 0.10


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.0,
        full_preactivation: bool = False,
        zero_init_last_bn: bool = False,
        stochastic_depth_p: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.zero_init_last_bn = zero_init_last_bn
        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.LayerNorm(normalized_shape=in_features)

        self.fc_1 = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )

        self.act_1 = Swish()
        self.do = nn.Dropout(p=dropout_p)
        self.fc_2 = nn.Linear(
            in_features=out_features, out_features=out_features, bias=True
        )

        if in_features == out_features:
            self.downsample_identity = lambda x: x
        else:
            self.downsample_identity = nn.Linear(
                in_features=in_features, out_features=out_features, bias=True
            )

        self.stochastic_depth = StochasticDepth(p=self.stochastic_depth_p, mode="batch")

        if self.zero_init_last_bn:
            nn.init.zeros_(self.norm_2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.act_1(out)
        out = self.do(out)
        out = self.fc_2(out)

        out = self.stochastic_depth(out)

        return out + identity


class SplitMLPResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_feature_sets: int,
        split_size: int,
        dropout_p: float = 0.0,
        stochastic_depth_p: float = 0.0,
        full_preactivation: bool = False,
        zero_init_last_bn: bool = False,
        reduce_both: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.split_size = split_size
        self.out_feature_sets = out_feature_sets

        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.zero_init_last_bn = zero_init_last_bn
        self.reduce_both = reduce_both
        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.LayerNorm(normalized_shape=in_features)
        self.fc_1 = SplitLinear(
            in_features=self.in_features,
            out_feature_sets=self.out_feature_sets,
            bias=True,
            split_size=self.split_size,
        )

        self.act_1 = Swish()
        self.do = nn.Dropout(p=dropout_p)

        fc_2_kwargs = _get_split_fc_2_kwargs(
            in_features=self.fc_1.out_features,
            out_feature_sets=self.out_feature_sets,
            bias=True,
            split_size=self.split_size,
            reduce_both=self.reduce_both,
        )
        self.fc_2 = SplitLinear(**fc_2_kwargs)

        if in_features == out_feature_sets:
            self.downsample_identity = lambda x: x
        else:
            self.downsample_identity = SplitLinear(
                in_features=self.in_features,
                out_feature_sets=1,
                bias=True,
                num_chunks=self.fc_2.out_features,
            )

        self.stochastic_depth = StochasticDepth(p=stochastic_depth_p, mode="batch")

        self.out_features = self.fc_2.out_features

        if self.zero_init_last_bn:
            nn.init.zeros_(self.norm_2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.act_1(out)
        out = self.do(out)
        out = self.fc_2(out)

        out = self.stochastic_depth(out)

        return out + identity


def _get_split_fc_2_kwargs(
    in_features: int,
    out_feature_sets: int,
    bias: bool,
    reduce_both: bool,
    split_size: int,
):
    common_kwargs = {
        "in_features": in_features,
        "out_feature_sets": out_feature_sets,
        "bias": bias,
    }

    if reduce_both:
        common_kwargs["split_size"] = split_size
    else:
        num_chunks = _calculate_num_chunks_for_equal_split_out_features(
            in_features=in_features, out_feature_sets=out_feature_sets
        )
        common_kwargs["num_chunks"] = num_chunks

    return common_kwargs


def _calculate_num_chunks_for_equal_split_out_features(
    in_features: int,
    out_feature_sets: int,
) -> int:
    """
    Ensure total out features are equal to in features.
    """
    return in_features // out_feature_sets
