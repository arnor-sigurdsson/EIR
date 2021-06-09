import math

import torch
import torch.nn.functional as F
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn
from torch.nn import Parameter

logger = get_logger(__name__)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = max(self.in_channels // 8, 1)

        self.conv_theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.reduction,
            kernel_size=1,
            bias=False,
        )
        self.conv_phi = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.reduction,
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
        theta = theta.view(-1, self.reduction, h * w)

        # Phi path
        phi = self.conv_phi(x)
        phi = self.pool(phi)
        phi = phi.view(-1, self.reduction, h * w // 4)

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
            bias=False,
        )
        self.act_1 = Swish()

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


class CNNResidualBlockBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rb_do: float,
        dilation: int,
        conv_1_kernel_h: int = 1,
        conv_1_kernel_w: int = 12,
        conv_1_padding: int = 4,
        down_stride_w: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_1_kernel_h = conv_1_kernel_h
        self.conv_1_kernel_w = conv_1_kernel_w
        self.conv_1_padding = conv_1_padding
        self.down_stride_w = down_stride_w

        self.down_stride_h = self.conv_1_kernel_h

        self.rb_do = nn.Dropout2d(rb_do)
        self.act_1 = Swish()

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

        self.act_2 = Swish()
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, conv_2_kernel_w),
            stride=(1, 1),
            padding=(0, conv_2_padding * dilation),
            dilation=(1, dilation),
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

        self.se_block = SEBlock(channels=out_channels, reduction=16)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class FirstCNNBlock(CNNResidualBlockBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "bn_1")
        delattr(self, "act_1")
        delattr(self, "downsample_identity")
        delattr(self, "bn_2")
        delattr(self, "act_2")
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

        channel_recalibrations = self.se_block(out)
        out = out * channel_recalibrations

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
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

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


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.0,
        full_preactivation: bool = False,
        zero_init_last_bn: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.zero_init_last_bn = zero_init_last_bn

        self.bn_1 = nn.BatchNorm1d(num_features=in_features)
        self.act_1 = Swish()
        self.fc_1 = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

        self.bn_2 = nn.BatchNorm1d(num_features=out_features)
        self.act_2 = Swish()
        self.do = nn.Dropout(p=dropout_p)
        self.fc_2 = nn.Linear(
            in_features=out_features, out_features=out_features, bias=False
        )

        if in_features == out_features:
            self.downsample_identity = lambda x: x
        else:
            self.downsample_identity = nn.Linear(
                in_features=in_features, out_features=out_features, bias=False
            )

        if self.zero_init_last_bn:
            nn.init.zeros_(self.bn_2.weight)

    def forward(self, x):
        out = self.bn_1(x)
        out = self.act_1(out)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.bn_2(out)
        out = self.act_2(out)
        out = self.do(out)
        out = self.fc_2(out)

        return out + identity


class SplitMLPResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_feature_sets: int,
        split_size: int,
        dropout_p: float = 0.0,
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

        self.bn_1 = nn.BatchNorm1d(num_features=in_features)
        self.act_1 = Swish()
        self.fc_1 = SplitLinear(
            in_features=self.in_features,
            out_feature_sets=self.out_feature_sets,
            bias=False,
            split_size=self.split_size,
        )

        self.bn_2 = nn.BatchNorm1d(num_features=self.fc_1.out_features)
        self.act_2 = Swish()
        self.do = nn.Dropout(p=dropout_p)

        fc_2_kwargs = _get_split_fc_2_kwargs(
            in_features=self.fc_1.out_features,
            out_feature_sets=self.out_feature_sets,
            bias=False,
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
                bias=False,
                num_chunks=self.fc_2.out_features,
            )

        self.out_features = self.fc_2.out_features

        if self.zero_init_last_bn:
            nn.init.zeros_(self.bn_2.weight)

    def forward(self, x):
        out = self.bn_1(x)
        out = self.act_1(out)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.bn_2(out)
        out = self.act_2(out)
        out = self.do(out)
        out = self.fc_2(out)

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
