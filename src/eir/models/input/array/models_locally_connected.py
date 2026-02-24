from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    Union,
    cast,
)

import torch
from torch import nn

from eir.models.input.sequence.transformer_models import PositionalEmbedding
from eir.models.layers.attention_layers import LinearAttention
from eir.models.layers.lcl_layers import LCL, LCLResidualBlock
from eir.models.layers.norm_layers import LayerScale
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.output.array.array_output_modules import LCLOutputModelConfig
    from eir.setup.input_setup_modules.common import DataDimensions

logger = get_logger(__name__)


class FlattenFunc(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass
class SimpleLCLModelConfig:
    """
    :param fc_repr_dim:
        Controls the number of output sets in the first and only split layer. Analogous
        to channels in CNNs.
    :param num_lcl_chunks:
        Controls the number of splits applied to the input. E.g. with a input with of
        800, using ``num_lcl_chunks=100`` will result in a kernel width of 8,
        meaning 8 elements in the flattened input. If using a SNP inputs with a one-hot
        encoding of 4 possible values, this will result in 8/2 = 2 SNPs per locally
        connected area.
    :param l1:
        L1 regularization applied to the first and only locally connected layer.
    """

    fc_repr_dim: int = 12
    num_lcl_chunks: int = 64
    l1: float = 0.00


class SimpleLCLModel(nn.Module):
    def __init__(
        self,
        model_config: SimpleLCLModelConfig,
        data_dimensions: "DataDimensions",
        flatten_fn: FlattenFunc,
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions
        self.flatten_fn = flatten_fn

        num_chunks = self.model_config.num_lcl_chunks
        self.fc_0 = LCL(
            in_features=self.fc_1_in_features,
            out_feature_sets=self.model_config.fc_repr_dim,
            num_chunks=num_chunks,
            bias=True,
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

    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.flatten_fn(x=input)

        out = self.fc_0(out)

        return out


@dataclass
class LCLModelConfig:
    """
    This is what the ``"genome-local-net"`` model refers to. See
    https://academic.oup.com/nar/article/51/12/e67/7177885 for more details on
    the model architecture.

    Note that when using the automatic network setup, kernel widths will get expanded
    to ensure that the feature representations become smaller as they are propagated
    through the network.

    :param patch_size:
        Controls the size of the patches used in the first layer. If set to ``None``,
        the input is flattened according to the torch ``flatten`` function. Note that
        when using this parameter, we generally want the kernel width to be set to
        the multiplication of the patch size. Order follows PyTorch convention, i.e.,
        [channels, height, width].

    :param layers:
        Controls the number of layers in the model. If set to ``None``, the model will
        automatically set up the number of layers according to the ``cutoff`` parameter
        value.

    :param kernel_width:
        With of the locally connected kernels. Note that in the context of genomic
        inputs this refers to the flattened input,
        meaning that if we have a one-hot encoding of 4 values (e.g. SNPs), 12
        refers to 12/4 = 3 SNPs per locally connected window. Can be set to ``None`` if
        the ``num_lcl_chunks`` parameter is set, which means that the kernel width
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

    :param num_lcl_chunks:
        Controls the number of splits applied to the input. E.g. with a input width of
        800, using ``num_lcl_chunks=100`` will result in a kernel width of 8,
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
        The 'auto' option is only supported when using the model for array *outputs*,
        and will set the cutoff to roughly the number of output features.

    :param direction:
        Whether to use a "down" or "up" network. "Down" means that the feature
        representation will get smaller as it is propagated through the network, whereas
        "up" means that the feature representation will get larger.

    :param attention_inclusion_cutoff:
        Cutoff to start including attention blocks in the network. If set to ``None``,
        no attention blocks will be included. The cutoff here refers to the "length"
        dimension of the input after reshaping according to the output_feature_sets
        in the preceding layer. For example, if we 1024 output features, and we have
        4 output feature sets, the length dimension will be 1024/4 = 256. With an
        attention cutoff >= 256, the attention block will be included.
    """

    patch_size: tuple[int, int, int] | None = None

    layers: None | list[int] = None

    kernel_width: int | Literal["patch"] = 16
    first_kernel_expansion: int = -2

    channel_exp_base: int = 2
    first_channel_expansion: int = 1

    num_lcl_chunks: None | int = None

    rb_do: float = 0.10
    stochastic_depth_p: float = 0.00
    l1: float = 0.00

    num_initial_maintaining_blocks: int = 0

    cutoff: int | Literal["auto"] = 1024
    direction: Literal["down", "up"] = "down"
    attention_inclusion_cutoff: int | None = None


class LCLModel(nn.Module):
    def __init__(
        self,
        model_config: Union[LCLModelConfig, "LCLOutputModelConfig"],
        data_dimensions: "DataDimensions",
        flatten_fn: FlattenFunc,
        dynamic_cutoff: int | None = None,
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions
        self.flatten_fn = flatten_fn

        kernel_width = parse_kernel_width(
            kernel_width=self.model_config.kernel_width,
            patch_size=self.model_config.patch_size,
        )

        fc_0_kernel_size = calc_value_after_expansion(
            base=kernel_width,
            expansion=self.model_config.first_kernel_expansion,
        )
        fc_0_out_feature_sets = calc_value_after_expansion(
            base=2**self.model_config.channel_exp_base,
            expansion=self.model_config.first_channel_expansion,
        )
        self.fc_0 = LCL(
            in_features=self.fc_1_in_features,
            out_feature_sets=fc_0_out_feature_sets,
            kernel_size=fc_0_kernel_size,
            bias=True,
        )
        self.act_0 = nn.GELU()

        cutoff = dynamic_cutoff or self.model_config.cutoff
        assert isinstance(cutoff, int)

        lcl_parameter_spec = LCParameterSpec(
            in_features=self.fc_0.out_features,
            kernel_width=kernel_width,
            channel_exp_base=self.model_config.channel_exp_base,
            dropout_p=self.model_config.rb_do,
            cutoff=cutoff,
            stochastic_depth_p=self.model_config.stochastic_depth_p,
            num_initial_maintaining_blocks=self.model_config.num_initial_maintaining_blocks,
            attention_inclusion_cutoff=self.model_config.attention_inclusion_cutoff,
            direction=self.model_config.direction,
        )
        self.lcl_blocks = _get_lcl_blocks(
            lcl_spec=lcl_parameter_spec,
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
        last_block = self.lcl_blocks[-1]
        return cast(int, last_block.out_features)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.flatten_fn(x=input)

        out = self.fc_0(out)
        out = self.act_0(out)
        out = self.lcl_blocks(out)

        return out


@dataclass
class LCLMoEModelConfig(LCLModelConfig):
    """
    :param num_experts:
        Number of parallel expert stacks. Each expert has its own ``fc_0`` and
        ``lcl_blocks``, allowing experts to specialize on different genomic patterns
        from the initial projection onward.
    """

    num_experts: int = 4


class ExpertTap(nn.Module):
    def __init__(self, expert_index: int, depth_name: str):
        """
        We gave this and the aggregation checkpoints to gather up all expert
        outputs at a given stage, and we need this little workaround here
        because they are run in parallel. So essentially, the point of this module,
        the AggregationCheckpoint module, and the ExpertAggregator module is to
        gather the outputs of all experts at a given stage - which can be
        used e.g. with the Tensor Broker to allow downstream modules to select
        among all expert outputs at different stages in the network.

        This ExpertTap module only has one job, to pull out *single* expert output at
        a given state and write it to the bound storage. Note that the storage
        is shared across all ExpertTap and AggregationCheckpoint modules, so that the
        each expert-stage combination must have it's own unique key/dict path written
        into the storage.
        """
        super().__init__()
        self.expert_index = expert_index
        self.depth_name = depth_name
        self._storage: dict[str, dict[int, torch.Tensor]] | None = None

    def bind_storage(self, storage: dict[str, dict[int, torch.Tensor]]) -> None:
        self._storage = storage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._storage is not None:
            if self.depth_name not in self._storage:
                self._storage[self.depth_name] = {}
            self._storage[self.depth_name][self.expert_index] = x
        return x


class AggregationCheckpoint(nn.Module):
    def __init__(self, num_experts: int, expert_dim: int):
        """
        This servers as to store all expert outputs at a given stage in the network
        as cached via the ExpertTap modules, and then make them available as a
        concatenated tensor. This maintains the tensor-out interface of a general
        module, but it also stores the boundaries that allows consumers of this
        checkpoint to know how to split the concatenated tensor back into individual
        expert outputs (for expert gating e.g. in the TB).
        """
        super().__init__()
        self._expert_boundaries: dict[str, int] = {
            "num_experts": num_experts,
            "expert_dim": expert_dim,
        }

    @property
    def expert_boundaries(self) -> dict[str, int]:
        return self._expert_boundaries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ExpertAggregator(nn.Module):
    def __init__(
        self,
        depth_configs: list[tuple[str, int]],
        num_experts: int,
    ):
        """
        This contains the actual logic to set up aggregation checkpoint and populate
        them with the concatenated outputs from the ExpertTap modules. This runs
        after each expert stack in the network (which means that the taps have
        actually already cached the individual expert outputs). Since e.g. the TB
        first calls each modules `forward` method to infer tensor dimensions,
        this will ensure the TB knowns the dimensions of the concatenated expert
        outputs at each stage.

        Then this aggregator is the actual entry point for the TBs, e.g. via:

        (aggregator): ExpertAggregator(
            (checkpoints): ModuleDict(
            (after_fc_0): AggregationCheckpoint()
            (after_lcl_block_0): AggregationCheckpoint()
            )
        )
        """
        super().__init__()
        self.depth_names = [name for name, _ in depth_configs]
        self.checkpoints = nn.ModuleDict()
        for depth_name, expert_dim in depth_configs:
            self.checkpoints[depth_name] = AggregationCheckpoint(
                num_experts=num_experts,
                expert_dim=expert_dim,
            )
        self._tap_storage: dict[str, dict[int, torch.Tensor]] | None = None

    def bind_storage(self, storage: dict[str, dict[int, torch.Tensor]]) -> None:
        self._tap_storage = storage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._tap_storage is not None
        for depth_name in self.depth_names:
            expert_dict = self._tap_storage[depth_name]
            expert_tensors = [expert_dict[k] for k in sorted(expert_dict.keys())]
            concatenated = torch.cat(expert_tensors, dim=1)
            self.checkpoints[depth_name](concatenated)
        return x


class LCLMoEModel(nn.Module):
    def __init__(
        self,
        model_config: LCLMoEModelConfig,
        data_dimensions: "DataDimensions",
        flatten_fn: FlattenFunc,
        dynamic_cutoff: int | None = None,
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions
        self.flatten_fn = flatten_fn
        self.num_experts = model_config.num_experts

        kernel_width = parse_kernel_width(
            kernel_width=self.model_config.kernel_width,
            patch_size=self.model_config.patch_size,
        )

        fc_0_kernel_size = calc_value_after_expansion(
            base=kernel_width,
            expansion=self.model_config.first_kernel_expansion,
        )
        fc_0_out_feature_sets = calc_value_after_expansion(
            base=2**self.model_config.channel_exp_base,
            expansion=self.model_config.first_channel_expansion,
        )

        cutoff = dynamic_cutoff or self.model_config.cutoff
        assert isinstance(cutoff, int)

        depth_configs: list[tuple[str, int]] = []
        self.expert_branches = nn.ModuleDict()
        for i in range(self.num_experts):
            fc_0 = LCL(
                in_features=self._fc_0_in_features,
                out_feature_sets=fc_0_out_feature_sets,
                kernel_size=fc_0_kernel_size,
                bias=True,
            )
            act_0 = nn.GELU()

            lcl_parameter_spec = LCParameterSpec(
                in_features=fc_0.out_features,
                kernel_width=kernel_width,
                channel_exp_base=self.model_config.channel_exp_base,
                dropout_p=self.model_config.rb_do,
                cutoff=cutoff,
                stochastic_depth_p=self.model_config.stochastic_depth_p,
                num_initial_maintaining_blocks=(
                    self.model_config.num_initial_maintaining_blocks
                ),
                attention_inclusion_cutoff=self.model_config.attention_inclusion_cutoff,
                direction=self.model_config.direction,
            )
            lcl_blocks = _get_lcl_blocks(
                lcl_spec=lcl_parameter_spec,
                block_layer_spec=self.model_config.layers,
            )

            modules: list[nn.Module] = [
                fc_0,
                act_0,
                ExpertTap(expert_index=i, depth_name="after_fc_0"),
            ]

            if i == 0:
                depth_configs.append(("after_fc_0", fc_0.out_features))

            for block_idx in range(len(lcl_blocks)):
                block = lcl_blocks[block_idx]
                depth_name = f"after_lcl_block_{block_idx}"
                modules.append(block)
                modules.append(ExpertTap(expert_index=i, depth_name=depth_name))

                if i == 0:
                    depth_configs.append((depth_name, cast(int, block.out_features)))

            self.expert_branches[f"expert_{i}"] = nn.Sequential(*modules)

        self._single_expert_out_features = depth_configs[-1][1]

        self.aggregator = ExpertAggregator(
            depth_configs=depth_configs,
            num_experts=self.num_experts,
        )

        self._tap_storage: dict[str, dict[int, torch.Tensor]] = {}
        self._bind_all_storage()
        self._init_weights()

    def _bind_all_storage(self) -> None:
        for expert in self.expert_branches.values():
            for module in expert.modules():
                if isinstance(module, ExpertTap):
                    module.bind_storage(storage=self._tap_storage)
        self.aggregator.bind_storage(storage=self._tap_storage)

    @property
    def _fc_0_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        weights: list[torch.Tensor] = []
        for expert in self.expert_branches.values():
            for module in expert.modules():
                if isinstance(module, LCL):
                    weights.append(module.weight)
                    break
        return torch.stack(weights)

    @property
    def num_out_features(self) -> int:
        return self.num_experts * self._single_expert_out_features

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    @property
    def expert_boundaries(self) -> dict[str, int]:
        return {
            "num_experts": self.num_experts,
            "expert_dim": self._single_expert_out_features,
        }

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.flatten_fn(x=input)

        self._tap_storage.clear()

        expert_outputs: list[torch.Tensor] = []
        for expert in self.expert_branches.values():
            expert_outputs.append(expert(out))

        final = torch.cat(expert_outputs, dim=1)
        final = self.aggregator(final)

        return final


def parse_kernel_width(
    kernel_width: int | Literal["patch"],
    patch_size: tuple[int, int, int] | None,
) -> int:
    if kernel_width == "patch":
        if patch_size is None:
            raise ValueError(
                "kernel_width set to 'patch', but no patch_size was specified."
            )
        kernel_width = patch_size[0] * patch_size[1] * patch_size[2]
    return kernel_width


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
    if expansion < 0:
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
    num_initial_maintaining_blocks: int = 0
    attention_inclusion_cutoff: int | None = None
    direction: Literal["down", "up"] = "down"


def _get_lcl_blocks(
    lcl_spec: LCParameterSpec,
    block_layer_spec: Sequence[int] | None,
) -> nn.Sequential:
    factory = _get_lcl_block_factory(block_layer_spec=block_layer_spec)

    blocks = factory(lcl_spec)

    return blocks


def _get_lcl_block_factory(
    block_layer_spec: Sequence[int] | None,
) -> Callable[[LCParameterSpec], nn.Sequential]:
    if not block_layer_spec:
        return generate_lcl_residual_blocks_auto

    auto_factory = partial(
        _generate_lcl_blocks_from_spec,
        block_layer_spec=block_layer_spec,
    )

    return auto_factory


def _generate_lcl_blocks_from_spec(
    lcl_parameter_spec: LCParameterSpec,
    block_layer_spec: Sequence[int],
) -> nn.Sequential:
    s = lcl_parameter_spec
    block_layer_spec_copy: list[int] = list(copy(block_layer_spec))

    block_modules: list[LCLResidualBlock | LCLAttentionBlock]
    block_modules = []

    maintaining_kernel_size = 2**s.channel_exp_base
    for i in range(s.num_initial_maintaining_blocks):
        cur_in_features = s.in_features if i == 0 else block_modules[-1].out_features
        maintaining_block = LCLResidualBlock(
            in_features=cur_in_features,
            kernel_size=maintaining_kernel_size,
            out_feature_sets=maintaining_kernel_size,
            dropout_p=s.dropout_p,
            stochastic_depth_p=s.stochastic_depth_p,
            full_preactivation=i == 0,
            reduce_both=False,
        )
        block_modules.append(maintaining_block)

    first_downsampling_block = LCLResidualBlock(
        in_features=block_modules[-1].out_features if block_modules else s.in_features,
        kernel_size=s.kernel_width,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=len(block_modules) == 0,
    )
    block_modules.append(first_downsampling_block)
    block_layer_spec_copy[0] -= 1

    for cur_layer_index, block_dim in enumerate(block_layer_spec_copy):
        for _block in range(block_dim):
            cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_layer_index)
            cur_kernel_width = s.kernel_width

            cur_out_feature_sets, cur_kernel_width = _adjust_auto_params(
                cur_out_feature_sets=cur_out_feature_sets,
                cur_kernel_width=cur_kernel_width,
                direction=s.direction,
            )

            cur_size = block_modules[-1].out_features

            cur_block = LCLResidualBlock(
                in_features=cur_size,
                kernel_size=cur_kernel_width,
                out_feature_sets=cur_out_feature_sets,
                dropout_p=s.dropout_p,
                stochastic_depth_p=s.stochastic_depth_p,
            )

            block_modules.append(cur_block)

    return nn.Sequential(*block_modules)


def _adjust_auto_params(
    cur_out_feature_sets: int, cur_kernel_width: int, direction: Literal["down", "up"]
) -> tuple[int, int]:
    """
    Down: increase kernel width until it is larger than the number of output feature
    sets.
    Up: increase number of output feature sets until it is larger than the kernel width.

    """
    if direction == "down":
        while cur_out_feature_sets >= cur_kernel_width:
            cur_kernel_width *= 2
    elif direction == "up":
        while cur_out_feature_sets <= cur_kernel_width:
            cur_out_feature_sets *= 2
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return cur_out_feature_sets, cur_kernel_width


def generate_lcl_residual_blocks_auto(lcl_parameter_spec: LCParameterSpec):
    """
    TODO:   Create some over-engineered abstraction for this and
            ``_generate_lcl_blocks_from_spec`` if feeling bored.
    """

    s = lcl_parameter_spec

    block_modules: list[LCLResidualBlock | LCLAttentionBlock]
    block_modules = []

    maintaining_kernel_size = 2**s.channel_exp_base
    for i in range(s.num_initial_maintaining_blocks):
        cur_in_features = s.in_features if i == 0 else block_modules[-1].out_features
        maintaining_block = LCLResidualBlock(
            in_features=cur_in_features,
            kernel_size=maintaining_kernel_size,
            out_feature_sets=maintaining_kernel_size,
            dropout_p=s.dropout_p,
            stochastic_depth_p=s.stochastic_depth_p,
            full_preactivation=i == 0,
            reduce_both=False,
        )
        block_modules.append(maintaining_block)

    first_downsampling_block = LCLResidualBlock(
        in_features=block_modules[-1].out_features if block_modules else s.in_features,
        kernel_size=s.kernel_width,
        out_feature_sets=2**s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=len(block_modules) == 0,
    )
    block_modules.append(first_downsampling_block)

    if _do_add_attention(
        attention_inclusion_cutoff=s.attention_inclusion_cutoff,
        in_features=first_downsampling_block.out_features,
        embedding_dim=first_downsampling_block.out_feature_sets,
    ):
        cur_attention_block = LCLAttentionBlock(
            embedding_dim=first_downsampling_block.out_feature_sets,
            in_features=first_downsampling_block.out_features,
        )
        block_modules.append(cur_attention_block)

    while True:
        cur_no_blocks = len(block_modules)
        cur_index = cur_no_blocks // 2

        cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_index)
        cur_kernel_width = s.kernel_width
        cur_out_feature_sets, cur_kernel_width = _adjust_auto_params(
            cur_out_feature_sets=cur_out_feature_sets,
            cur_kernel_width=cur_kernel_width,
            direction=s.direction,
        )

        cur_size = block_modules[-1].out_features

        if _should_break_auto(
            cur_size=cur_size,
            cutoff=s.cutoff,
            direction=s.direction,
        ):
            break

        cur_block = LCLResidualBlock(
            in_features=cur_size,
            kernel_size=cur_kernel_width,
            out_feature_sets=cur_out_feature_sets,
            dropout_p=s.dropout_p,
            stochastic_depth_p=s.stochastic_depth_p,
        )

        block_modules.append(cur_block)

        if _do_add_attention(
            attention_inclusion_cutoff=s.attention_inclusion_cutoff,
            in_features=cur_block.out_features,
            embedding_dim=cur_block.out_feature_sets,
        ):
            cur_attention_block = LCLAttentionBlock(
                embedding_dim=cur_block.out_feature_sets,
                in_features=cur_block.out_features,
            )
            block_modules.append(cur_attention_block)

    logger.debug(
        "No SplitLinear residual blocks specified in CL arguments. Created %d "
        "blocks with final output dimension of %d.",
        len(block_modules),
        cur_size,
    )
    return nn.Sequential(*block_modules)


def _should_break_auto(
    cur_size: int, cutoff: int, direction: Literal["up", "down"]
) -> bool:
    if direction == "down":
        return cur_size <= cutoff
    if direction == "up":
        return cur_size >= cutoff
    raise ValueError(f"Unknown direction: {direction}")


class LCLAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        in_features: int,
        num_heads: int | Literal["auto"] = "auto",
        dropout_p: float = 0.0,
        attention_type: Literal["full", "linear"] = "full",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.in_features = in_features
        self.dropout_p = dropout_p
        self.attention_type = attention_type
        self.out_features = in_features

        if num_heads == "auto":
            self.num_heads = embedding_dim
        else:
            self.num_heads = num_heads
        assert isinstance(self.num_heads, int)

        self.attention: nn.MultiheadAttention | LinearAttention
        if attention_type == "full":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=dropout_p,
            )
        elif attention_type == "linear":
            self.attention = LinearAttention(
                embed_dim=self.embedding_dim,
                heads=self.num_heads,
                dim_head=self.embedding_dim // self.num_heads,
            )

        self.norm = nn.RMSNorm(self.embedding_dim)
        self.pos_emb = PositionalEmbedding(
            embedding_dim=self.embedding_dim,
            max_length=self.in_features // self.embedding_dim,
            dropout=self.dropout_p,
            zero_init=True,
        )

        self.ls = LayerScale(dim=self.in_features, init_values=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.reshape(x.shape[0], -1, self.embedding_dim)
        out = self.pos_emb(out)

        if self.attention_type == "full":
            out = self.norm(out)
            attn_output, _ = self.attention(out, out, out)
            out = out + attn_output
        elif self.attention_type == "linear":
            out = self.norm(out)
            attn_output = self.attention(out)
            out = out + attn_output
        else:
            raise ValueError("attention_type must be either 'full' or 'linear'")

        out = torch.flatten(out, start_dim=1)
        out = self.ls(out)

        return x + out


def _do_add_attention(
    in_features: int, embedding_dim: int, attention_inclusion_cutoff: int | None
) -> bool:
    if attention_inclusion_cutoff is None:
        return False

    attention_sequence_length = in_features // embedding_dim
    return attention_sequence_length <= attention_inclusion_cutoff
