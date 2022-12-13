from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import List, Callable, Sequence, TYPE_CHECKING, Union

import torch
from aislib.misc_utils import get_logger
from torch import nn

from eir.models.layers import SplitLinear, SplitMLPResidualBlock

if TYPE_CHECKING:
    from eir.setup.input_setup import DataDimensions

logger = get_logger(__name__)


@dataclass
class SimpleLCLModelConfig:
    """
    :param fc_repr_dim:
        Controls the number of output sets in the first and only split layer. Analogous
        to channels in CNNs.
    :param split_mlp_num_splits:
        Controls the number of splits applied to the input. E.g. with a input with of
        800, using ``split_mlp_num_splits=100`` will result in a kernel width of 8,
        meaning 8 elements in the flattened input. If using a SNP inputs with a one-hot
        encoding of 4 possible values, this will result in 8/2 = 2 SNPs per locally
        connected area.
    :param l1:
        L1 regularization applied to the first and only locally connected layer.
    """

    fc_repr_dim: int = 12
    split_mlp_num_splits: int = 64
    l1: float = 0.00


class SimpleLCLModel(nn.Module):
    def __init__(
        self, model_config: SimpleLCLModelConfig, data_dimensions: "DataDimensions"
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions

        num_chunks = self.model_config.split_mlp_num_splits
        self.fc_0 = SplitLinear(
            in_features=self.fc_1_in_features,
            out_feature_sets=self.model_config.fc_repr_dim,
            num_chunks=num_chunks,
            bias=False,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    @property
    def num_out_features(self) -> int:
        return self.fc_0.out_features

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = flatten_h_w_fortran(x=input)

        out = self.fc_0(out)

        return out


@dataclass
class LCLModelConfig:
    """
    Note that when using the automatic network setup, kernel widths will get expanded
    to ensure that the feature representations become smaller as they are propagated
    through the network.

    :param layers:
        Controls the number of layers in the model. If set to ``None``, the model will
        automatically set up the number of layers according to the ``cutoff`` parameter
        value.

    :param kernel_width:
        With of the locally connected kernels. Note that this refers to the flattened
        input, meaning that if we have a one-hot encoding of 4 values (e.g. SNPs), 12
        refers to 12/4 = 3 SNPs per locally connected window. Can be set to ``None`` if
        the ``split_mlp_num_splits`` parameter is set, which means that the kernel width
        will be set automatically according to

    :param first_kernel_expansion:
        Factor to extend the first kernel. This value can both be positive or negative.
        For example in the case of ``kernel_width=12``, setting
        ``first_kernel_expansion=2`` means that the first kernel will have a width of
        24, whereas other kernels will have a width of 12. When using a negative value,
        divides the first kernel by the value instead of multiplying.

    :param channel_exp_base:
        Which power of 2 to use in order to set the number of channels/weight sets in
        the network. For example, setting ``channel_exp_base=3`` means that 2**3=8
        weight sets will be used.

    :param first_channel_expansion:
        Whether to expand / shrink the number of channels in the first layer as compared
        to other layers in the network. Works analogously to the
        ``first_kernel_expansion`` parameter.

    :param split_mlp_num_splits:
        Controls the number of splits applied to the input. E.g. with a input width of
        800, using ``split_mlp_num_splits=100`` will result in a kernel width of 8,
        meaning 8 elements in the flattened input. If using a SNP inputs with a one-hot
        encoding of 4 possible values, this will result in 8/2 = 2 SNPs per locally
        connected area.

    :param rb_do:
        Dropout in the residual blocks.

    :param stochastic_depth_p:
        Probability of dropping input.

    :param l1:
        L1 regularization applied to the first layer in the network.

    :param cutoff:
        Feature dimension cutoff where the automatic network setup stops adding layers.
    """

    layers: Union[None, List[int]] = None

    kernel_width: Union[None, int] = 16
    first_kernel_expansion: int = -2

    channel_exp_base: int = 2
    first_channel_expansion: int = 1

    split_mlp_num_splits: Union[None, int] = None

    rb_do: float = 0.10
    stochastic_depth_p: float = 0.00
    l1: float = 0.00

    cutoff: int = 1024


class LCLModel(nn.Module):
    def __init__(self, model_config: LCLModelConfig, data_dimensions: "DataDimensions"):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions

        fc_0_split_size = calc_value_after_expansion(
            base=self.model_config.kernel_width,
            expansion=self.model_config.first_kernel_expansion,
        )
        fc_0_channel_exponent = calc_value_after_expansion(
            base=self.model_config.channel_exp_base,
            expansion=self.model_config.first_channel_expansion,
        )
        self.fc_0 = SplitLinear(
            in_features=self.fc_1_in_features,
            out_feature_sets=2**fc_0_channel_exponent,
            split_size=fc_0_split_size,
            bias=False,
        )

        split_parameter_spec = LCParameterSpec(
            in_features=self.fc_0.out_features,
            kernel_width=self.model_config.kernel_width,
            channel_exp_base=self.model_config.channel_exp_base,
            dropout_p=self.model_config.rb_do,
            cutoff=self.model_config.cutoff,
            stochastic_depth_p=self.model_config.stochastic_depth_p,
        )
        self.split_blocks = _get_split_blocks(
            split_parameter_spec=split_parameter_spec,
            block_layer_spec=self.model_config.layers,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    @property
    def num_out_features(self) -> int:
        return self.split_blocks[-1].out_features

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = flatten_h_w_fortran(x=input)

        out = self.fc_0(out)
        out = self.split_blocks(out)

        return out


def flatten_h_w_fortran(x: torch.Tensor) -> torch.Tensor:
    """
    This is needed when e.g. flattening one-hot inputs that are ordered in a columns
    wise fashion (meaning that each column is a one-hot feature),
    and we want to make sure the first part of the flattened tensor is the first column,
    i.e. first one-hot element.
    """
    column_order_flattened = x.transpose(2, 3).flatten(1)
    return column_order_flattened


def calc_value_after_expansion(base: int, expansion: int, min_value: int = 0) -> int:
    if expansion > 0:
        return base * expansion
    elif expansion < 0:
        abs_expansion = abs(expansion)
        return max(min_value, base // abs_expansion)
    return base


@dataclass
class LCParameterSpec:
    in_features: int
    kernel_width: int
    channel_exp_base: int
    dropout_p: float
    stochastic_depth_p: float
    cutoff: int


def _get_split_blocks(
    split_parameter_spec: LCParameterSpec,
    block_layer_spec: Union[None, Sequence[int]],
) -> nn.Sequential:
    factory = _get_split_block_factory(block_layer_spec=block_layer_spec)

    blocks = factory(split_parameter_spec)

    return blocks


def _get_split_block_factory(
    block_layer_spec: Sequence[int],
) -> Callable[[LCParameterSpec], nn.Sequential]:
    if not block_layer_spec:
        return generate_split_resblocks_auto

    auto_factory = partial(
        _generate_split_blocks_from_spec, block_layer_spec=block_layer_spec
    )

    return auto_factory


def _generate_split_blocks_from_spec(
    split_parameter_spec: LCParameterSpec,
    block_layer_spec: List[int],
) -> nn.Sequential:

    s = split_parameter_spec
    block_layer_spec_copy = copy(block_layer_spec)

    first_block = SplitMLPResidualBlock(
        in_features=s.in_features,
        split_size=s.kernel_width,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=True,
    )

    block_modules = [first_block]
    block_layer_spec_copy[0] -= 1

    for cur_layer_index, block_dim in enumerate(block_layer_spec_copy):
        for block in range(block_dim):

            cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_layer_index)
            cur_kernel_width = s.kernel_width
            while cur_out_feature_sets >= cur_kernel_width:
                cur_kernel_width *= 2

            cur_size = block_modules[-1].out_features

            cur_block = SplitMLPResidualBlock(
                in_features=cur_size,
                split_size=cur_kernel_width,
                out_feature_sets=cur_out_feature_sets,
                dropout_p=s.dropout_p,
                stochastic_depth_p=s.stochastic_depth_p,
            )

            block_modules.append(cur_block)

    return nn.Sequential(*block_modules)


def generate_split_resblocks_auto(split_parameter_spec: LCParameterSpec):
    """
    TODO:   Create some over-engineered abstraction for this and
            ``_generate_split_blocks_from_spec`` if feeling bored.
    """

    s = split_parameter_spec

    first_block = SplitMLPResidualBlock(
        in_features=s.in_features,
        split_size=s.kernel_width,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=True,
    )

    block_modules = [first_block]

    while True:
        cur_no_blocks = len(block_modules)
        cur_index = cur_no_blocks // 2

        cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_index)
        cur_kernel_width = s.kernel_width
        while cur_out_feature_sets >= cur_kernel_width:
            cur_kernel_width *= 2

        cur_size = block_modules[-1].out_features
        if cur_size <= s.cutoff:
            break

        cur_block = SplitMLPResidualBlock(
            in_features=cur_size,
            split_size=cur_kernel_width,
            out_feature_sets=cur_out_feature_sets,
            dropout_p=s.dropout_p,
            stochastic_depth_p=s.stochastic_depth_p,
        )

        block_modules.append(cur_block)

    logger.info(
        "No SplitLinear residual blocks specified in CL arguments. Created %d "
        "blocks with final output dimension of %d.",
        len(block_modules),
        cur_size,
    )
    return nn.Sequential(*block_modules)
