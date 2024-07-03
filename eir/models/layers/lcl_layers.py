import math
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torchvision.ops import StochasticDepth

from eir.models.layers.norm_layers import LayerScale
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


class LCL(nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_feature_sets: int,
        num_chunks: int = 10,
        kernel_size: Optional[int] = None,
        bias: Optional[bool] = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_feature_sets = out_feature_sets
        self.num_chunks = num_chunks

        if kernel_size:
            self.kernel_size = kernel_size
            self.num_chunks = int(math.ceil(in_features / kernel_size))
            logger.debug(
                "%s: Setting num chunks to %d as kernel size of %d was passed in.",
                self.__class__,
                self.num_chunks,
                kernel_size,
            )
        else:
            self.kernel_size = int(math.ceil(self.in_features / self.num_chunks))
            logger.debug(
                "%s :Setting kernel size to %d as number of "
                "chunks of %d was passed in.",
                self.__class__,
                self.kernel_size,
                self.num_chunks,
            )

        assert self.kernel_size is not None

        self.out_features = self.out_feature_sets * self.num_chunks

        self.padding = _find_lcl_padding_needed(
            input_size=self.in_features,
            kernel_size=self.kernel_size,
            num_chunks=self.num_chunks,
        )

        self.weight = Parameter(
            torch.Tensor(self.out_feature_sets, self.num_chunks, self.kernel_size),
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
            "in_features={}, num_chunks={}, kernel_size={}, "
            "out_feature_sets={}, out_features={}, bias={}".format(
                self.in_features,
                self.num_chunks,
                self.kernel_size,
                self.out_feature_sets,
                self.out_features,
                self.bias is not None,
            )
        )

    def forward(self, input: torch.Tensor):
        input_padded = F.pad(input=input, pad=[0, self.padding, 0, 0])

        input_reshaped = input_padded.reshape(
            input.shape[0], 1, self.num_chunks, self.kernel_size
        )

        out = calc_lcl_forward(input=input_reshaped, weight=self.weight, bias=self.bias)
        return out


def _find_lcl_padding_needed(input_size: int, kernel_size: int, num_chunks: int):
    return num_chunks * kernel_size - input_size


def calc_lcl_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    n: num samples
    c: num chunks (height)
    s: kernel size (width)
    o: output sets
    """

    summed = torch.einsum("nhw, ohw -> noh", input.squeeze(1), weight)
    flattened = summed.flatten(start_dim=1)

    final = flattened
    if bias is not None:
        final = flattened + bias

    return final


class LCLResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_feature_sets: int,
        kernel_size: int,
        dropout_p: float = 0.0,
        stochastic_depth_p: float = 0.0,
        full_preactivation: bool = False,
        reduce_both: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.kernel_size = kernel_size
        self.out_feature_sets = out_feature_sets

        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.reduce_both = reduce_both
        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.LayerNorm(normalized_shape=in_features)
        self.fc_1 = LCL(
            in_features=self.in_features,
            out_feature_sets=self.out_feature_sets,
            bias=True,
            kernel_size=self.kernel_size,
        )

        self.act_1 = nn.GELU()
        self.do = nn.Dropout(p=dropout_p)

        fc_2_kwargs = _get_lcl_2_kwargs(
            in_features=self.fc_1.out_features,
            out_feature_sets=self.out_feature_sets,
            bias=True,
            kernel_size=self.kernel_size,
            reduce_both=self.reduce_both,
        )
        self.fc_2 = LCL(**fc_2_kwargs)

        self.out_features = self.fc_2.out_features

        self.ls = LayerScale(dim=self.out_features, init_values=1.0)

        if in_features == self.out_features:
            self.downsample_identity = lambda x: x
        else:
            self.downsample_identity = LCL(
                in_features=self.in_features,
                out_feature_sets=1,
                bias=True,
                num_chunks=self.fc_2.out_features,
            )

        self.stochastic_depth = StochasticDepth(p=stochastic_depth_p, mode="batch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.act_1(out)
        out = self.do(out)
        out = self.fc_2(out)
        out = self.ls(out)

        out = self.stochastic_depth(out)

        return out + identity


def _get_lcl_2_kwargs(
    in_features: int,
    out_feature_sets: int,
    bias: bool,
    reduce_both: bool,
    kernel_size: int,
):
    common_kwargs = {
        "in_features": in_features,
        "out_feature_sets": out_feature_sets,
        "bias": bias,
    }

    if reduce_both:
        common_kwargs["kernel_size"] = kernel_size
    else:
        num_chunks = _calculate_num_chunks_for_equal_lcl_out_features(
            in_features=in_features,
            out_feature_sets=out_feature_sets,
        )
        common_kwargs["num_chunks"] = num_chunks

    return common_kwargs


def _calculate_num_chunks_for_equal_lcl_out_features(
    in_features: int,
    out_feature_sets: int,
) -> int:
    """
    Ensure total out features are equal to in features.
    """
    return in_features // out_feature_sets
