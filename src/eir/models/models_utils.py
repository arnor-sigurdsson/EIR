from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import torch
from torch import nn
from transformers import PreTrainedModel

from eir.models.input.sequence.transformer_models import get_hf_transformer_forward
from eir.models.layers.mlp_layers import MLPResidualBlock
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.output_setup_modules.tabular_output_setup import (
        al_num_outputs_per_target,
    )

logger = get_logger(name=__name__, tqdm_compatible=True)


def merge_module_dicts(module_dicts: tuple[nn.ModuleDict, ...]) -> nn.ModuleDict:
    def _check_inputs():
        assert all(i.keys() == module_dicts[0].keys() for i in module_dicts)

    _check_inputs()

    new_module_dicts = deepcopy(module_dicts)
    final_module_dict = nn.ModuleDict()

    keys = new_module_dicts[0].keys()
    for key in keys:
        final_module_dict[key] = nn.Sequential()

        for index, module_dict in enumerate(new_module_dicts):
            cur_module = module_dict[key]
            final_module_dict[key].add_module(str(index), cur_module)

    return final_module_dict


def construct_blocks(
    num_blocks: int,
    block_constructor: Callable,
    block_kwargs: dict,
) -> nn.Sequential:
    blocks = []
    for _i in range(num_blocks):
        cur_block = block_constructor(**block_kwargs)
        blocks.append(cur_block)
    return nn.Sequential(*blocks)


def create_multi_task_blocks_with_first_adaptor_block(
    num_blocks: int,
    branch_names: Sequence[str],
    block_constructor: Callable,
    block_constructor_kwargs: dict,
    first_layer_kwargs_overload: dict,
) -> nn.ModuleDict:
    adaptor_block = construct_multi_branches(
        branch_names=branch_names,
        branch_factory=construct_blocks,
        branch_factory_kwargs={
            "num_blocks": 1,
            "block_constructor": block_constructor,
            "block_kwargs": {**block_constructor_kwargs, **first_layer_kwargs_overload},
        },
    )

    if num_blocks == 1:
        return merge_module_dicts((adaptor_block,))

    blocks = construct_multi_branches(
        branch_names=branch_names,
        branch_factory=construct_blocks,
        branch_factory_kwargs={
            "num_blocks": num_blocks - 1,
            "block_constructor": block_constructor,
            "block_kwargs": {**block_constructor_kwargs},
        },
    )

    merged_blocks = merge_module_dicts(module_dicts=(adaptor_block, blocks))

    return merged_blocks


def assert_module_dict_uniqueness(module_dict: nn.ModuleDict):
    """
    We have this function as a safeguard to help us catch if we are reusing modules
    when they should not be (i.e. if splitting into multiple branches with same layers,
    one could accidentally reuse the instantiated nn.Modules across branches).
    """
    branch_ids = [id(sequential_branch) for sequential_branch in module_dict.values()]
    assert len(branch_ids) == len(set(branch_ids))

    module_ids = []
    for sequential_branch in module_dict.values():
        module_ids += [id(module) for module in sequential_branch.modules()]

    num_total_modules = len(module_ids)
    num_unique_modules = len(set(module_ids))
    assert num_unique_modules == num_total_modules


def construct_multi_branches(
    branch_names: Iterable[str],
    branch_factory: Callable[..., nn.Sequential],
    branch_factory_kwargs: dict[str, Any],
    extra_hooks: Sequence[Callable] = (),
) -> nn.ModuleDict:
    branched_module_dict = nn.ModuleDict()
    for name in branch_names:
        cur_branch = branch_factory(**branch_factory_kwargs)
        assert callable(cur_branch)
        branched_module_dict[name] = cur_branch

    for hook in extra_hooks:
        branched_module_dict = hook(branched_module_dict)

    assert_module_dict_uniqueness(module_dict=branched_module_dict)
    return branched_module_dict


def get_final_layer(
    in_features: int,
    num_outputs_per_target: "al_num_outputs_per_target",
    layer_type: Literal["linear"] | Literal["mlp_residual"] = "linear",
    layer_type_specific_kwargs: dict | None = None,
) -> nn.ModuleDict:
    final_module_dict = nn.ModuleDict()

    if layer_type_specific_kwargs is None:
        layer_type_specific_kwargs = {}

    for task, num_outputs in num_outputs_per_target.items():
        if layer_type == "mlp_residual":
            spec_init_kwargs = _get_mlp_residual_final_spec(
                in_features=in_features,
                num_outputs=num_outputs,
                dropout_p=layer_type_specific_kwargs.get("dropout_p", 0.0),
                stochastic_depth_p=layer_type_specific_kwargs.get(
                    "stochastic_depth_p", 0.0
                ),
            )
        else:
            spec_init_kwargs = _get_mlp_basic_final_spec(
                in_features=in_features,
                num_outputs=num_outputs,
                bias=layer_type_specific_kwargs.get("bias", True),
            )

        cur_spec = OrderedDict({"final": spec_init_kwargs})
        cur_module = initialize_modules_from_spec(spec=cur_spec)
        final_module_dict[task] = cur_module

    return final_module_dict


def _get_mlp_residual_final_spec(
    in_features: int,
    num_outputs: int,
    dropout_p: float,
    stochastic_depth_p: float,
) -> tuple[type[nn.Module], dict[str, Any]]:
    spec = (
        MLPResidualBlock,
        {
            "in_features": in_features,
            "out_features": num_outputs,
            "dropout_p": dropout_p,
            "stochastic_depth_p": stochastic_depth_p,
        },
    )

    return spec


def _get_mlp_basic_final_spec(
    in_features: int,
    num_outputs: int,
    bias: bool = True,
) -> tuple[type[nn.Module], dict[str, Any]]:
    spec = (
        nn.Linear,
        {
            "in_features": in_features,
            "out_features": num_outputs,
            "bias": bias,
        },
    )

    return spec


def initialize_modules_from_spec(
    spec: "OrderedDict[str, tuple[type[nn.Module], dict]]",
) -> nn.Sequential:
    module_dict = OrderedDict()
    for name, recipe in spec.items():
        module_class = recipe[0]
        module_args = recipe[1]

        module = initialize_module(module=module_class, module_args=module_args)

        module_dict[name] = module

    return nn.Sequential(module_dict)


def initialize_module(module: type[nn.Module], module_args: dict) -> nn.Module:
    return module(**module_args)


def calculate_module_dict_outputs(
    input_: torch.Tensor, module_dict: nn.ModuleDict
) -> "OrderedDict[str, torch.Tensor]":
    out = OrderedDict()
    for name, module in module_dict.items():
        out[name] = module(input_)

    return out


def get_output_dimensions_for_input(
    module: PreTrainedModel | nn.Module,
    input_shape: tuple[int, ...],
    pool: Literal["max"] | Literal["avg"] | None,
    hf_model: bool = False,
) -> torch.LongTensor:
    module_copy = deepcopy(module)

    with torch.no_grad():
        test_input = torch.randn(*input_shape)

        if hf_model:
            hf_forward = get_hf_transformer_forward(
                feature_extractor_=module_copy,
                input_length=input_shape[1],
                embedding_dim=input_shape[2],
                device="cpu",
                pool=pool,
            )
            test_output = hf_forward(input=test_input, feature_extractor=module_copy)
        else:
            test_output = module_copy(test_input)

        output_shape = test_output.shape
        logger.info(
            "Inferred output shape of %s from model %s.",
            output_shape,
            type(module).__name__,
        )
        return output_shape


def log_model(
    model: nn.Module,
    structure_file: Path | None,
    verbose: bool = False,
    parameter_file: str | None = None,
    context: str = "Starting Training",
) -> None:
    no_params = sum(p.numel() for p in model.parameters())
    no_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_summary = f"{context} with following model specifications:\n"
    model_summary += f"Total parameters: {format(no_params, ',.0f')}\n"
    model_summary += f"Trainable parameters: {format(no_trainable_params, ',.0f')}"

    logger.info(model_summary)

    if structure_file:
        with open(structure_file, "w") as structure_handle:
            structure_handle.write(str(model))

    layer_summary = ""
    if verbose:
        layer_summary = "\nModel layers:\n"
        for name, param in model.named_parameters():
            layer_summary += (
                f"Layer: {name}, "
                f"Shape: {list(param.size())}, "
                f"Parameters: {param.numel()}, "
                f"Trainable: {param.requires_grad}\n"
            )
        logger.info(layer_summary)

    if parameter_file:
        with open(parameter_file, "w") as f:
            f.write(model_summary)
            if verbose:
                f.write(layer_summary)
