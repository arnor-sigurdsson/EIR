from dataclasses import dataclass
from difflib import get_close_matches
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_utils import Batch
from eir.models.meta.meta_utils import (
    al_fusion_modules,
    al_input_modules,
    al_output_modules,
    run_meta_forward,
)
from eir.models.tensor_broker.tensor_broker_fusion_layers import (
    get_fusion_layer_wrapper,
)
from eir.models.tensor_broker.tensor_broker_projection_layers import (
    get_projection_layer,
)
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.schemas import FusionConfig, InputConfig, OutputConfig
from eir.train_utils.step_logic import (
    al_dataloader_getitem_batch,
    prepare_base_batch_default,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(name=__name__)


@dataclass
class CachedTensor:
    tensor: torch.Tensor
    shape: torch.Size
    layer_path: str


def prepare_example_test_batch(
    input_objects: al_input_objects_as_dict,
    output_objects: al_output_objects_as_dict,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 2,
) -> Batch:
    imputed_inputs = impute_missing_modalities_wrapper(
        inputs_values={}, inputs_objects=input_objects
    )

    loader_batch: al_dataloader_getitem_batch = (imputed_inputs, {}, list())
    batch_as_list = [loader_batch] * batch_size
    loader_batch_collated = default_collate(batch=batch_as_list)

    batch = prepare_base_batch_default(
        loader_batch=loader_batch_collated,
        input_objects=input_objects,
        output_objects=output_objects,
        model=model,
        device=device,
    )

    return batch


def create_tensor_shapes_estimation_hook(
    output_shapes: Dict[str, torch.Size],
    input_shapes: Dict[str, torch.Size],
    layer_path: str,
) -> Callable:

    def hook(
        module: nn.Module,
        args: Tuple[torch.Tensor, ...],
        kwargs: Dict[str, torch.Tensor],
        output: torch.Tensor | Dict[str, torch.Tensor],
    ):
        if isinstance(output, torch.Tensor):
            output_shapes[layer_path] = output.shape[1:]
        elif isinstance(output, dict):
            cur_output_shapes = {k: v.shape[1:] for k, v in output.items()}
            for k, v in cur_output_shapes.items():
                output_shapes[f"{layer_path}.{k}"] = v

        if len(args) > 0:
            input_tensor = args[0]
        else:
            x = kwargs.get("x")
            input_key = kwargs.get("input")
            if x is not None:
                input_tensor = x
            elif input_key is not None:
                input_tensor = input_key
            else:
                raise ValueError("No input tensor found in args or kwargs.")

        if isinstance(input_tensor, torch.Tensor):
            input_shapes[layer_path] = input_tensor.shape[1:]
        elif isinstance(input_tensor, dict):
            cur_input_shapes = {k: v.shape[1:] for k, v in input_tensor.items()}
            for k, v in cur_input_shapes.items():
                input_shapes[f"{layer_path}.{k}"] = v

    return hook


def estimate_tensor_shapes(
    module_storage: nn.Module,
    fusion_to_output_mapping: Dict[str, Literal["computed", "pass-through"]],
    example_input_data: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Size], Dict[str, torch.Size]]:
    output_shapes: Dict[str, torch.Size] = {}
    input_shapes: Dict[str, torch.Size] = {}

    hooks = []

    for name, module in module_storage.named_modules():
        cur_hook = create_tensor_shapes_estimation_hook(
            output_shapes=output_shapes,
            input_shapes=input_shapes,
            layer_path=name,
        )
        hooks.append(module.register_forward_hook(cur_hook, with_kwargs=True))

    with torch.no_grad():
        run_meta_forward(
            input_modules=module_storage.input_modules,
            fusion_modules=module_storage.fusion_modules,
            output_modules=module_storage.output_modules,
            fusion_to_output_mapping=fusion_to_output_mapping,
            inputs=example_input_data,
        )

    for h in hooks:
        h.remove()

    return output_shapes, input_shapes


class ModuleStorage(nn.Module):
    def __init__(
        self,
        input_modules: al_input_modules,
        fusion_modules: al_fusion_modules,
        output_modules: al_output_modules,
    ):
        """
        We use this to imitate the future initialised Meta model, following the same
        structure, forward etc.
        """
        super().__init__()
        self.input_modules = input_modules
        self.fusion_modules = fusion_modules
        self.output_modules = output_modules

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return run_meta_forward(
            input_modules=self.input_modules,
            fusion_modules=self.fusion_modules,
            output_modules=self.output_modules,
            fusion_to_output_mapping={},
            inputs=inputs,
        )


def attach_store_forward_hook(
    module: nn.Module,
    cache: Dict[str, CachedTensor],
    layer_path: str,
    layer_cache_target: Literal["input", "output"],
) -> Callable[[], None]:
    def hook(module: nn.Module, args: Tuple[torch.Tensor, ...], output: torch.Tensor):

        if layer_cache_target == "input":
            cache[layer_path] = CachedTensor(
                tensor=args[0],
                shape=args[0].shape,
                layer_path=layer_path,
            )
        else:
            cache[layer_path] = CachedTensor(
                tensor=output,
                shape=output.shape,
                layer_path=layer_path,
            )

    handle = module.register_forward_hook(hook)

    def remove_hook():
        handle.remove()

    return remove_hook


def attach_tensor_broker_module_injection(
    target_module: nn.Module,
    tensor_broker_module: nn.Module,
    tensor_cache: Dict[str, CachedTensor],
    tensor_cache_key: str,
) -> Callable[[], None]:
    def hook(
        module: nn.Module,
        args: Tuple[torch.Tensor, ...],
        kwargs: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        if len(args) > 0:
            input_tensor = args[0]
        else:
            x = kwargs.get("x")
            input_key = kwargs.get("input")
            if x is not None:
                input_tensor = x
            elif input_key is not None:
                input_tensor = input_key
            else:
                raise ValueError("No input tensor found in args or kwargs.")

        cached_tensor = tensor_cache[tensor_cache_key].tensor
        tensor_broker_out = tensor_broker_module(input_tensor, cached_tensor)

        if len(args) > 0:
            new_args = (tensor_broker_out,) + args[1:]
            return new_args, kwargs
        else:
            new_kwargs = kwargs.copy()
            new_kwargs["x" if "x" in kwargs else "input"] = tensor_broker_out
            return args, new_kwargs

    handle = target_module.register_forward_pre_hook(hook, with_kwargs=True)

    def remove_hook():
        handle.remove()

    return remove_hook


def get_tensor_broker(
    input_objects: al_input_objects_as_dict,
    output_objects: al_output_objects_as_dict,
    input_modules: al_input_modules,
    fusion_modules: al_fusion_modules,
    output_modules: al_output_modules,
    fusion_to_output_mapping: Dict[str, Literal["computed", "pass-through"]],
    input_configs: Sequence[InputConfig],
    fusion_configs: Sequence[FusionConfig],
    output_configs: Sequence[OutputConfig],
    device: str,
) -> nn.ModuleDict:

    tensor_broker_modules = nn.ModuleDict()
    all_configs = list(input_configs) + list(fusion_configs) + list(output_configs)

    any_tensor_broker = any(
        config.tensor_broker_config is not None for config in all_configs
    )
    if not any_tensor_broker:
        return tensor_broker_modules

    module_storage = ModuleStorage(
        input_modules=input_modules,
        fusion_modules=fusion_modules,
        output_modules=output_modules,
    )
    module_storage = module_storage.to(device=device)

    example_batch = prepare_example_test_batch(
        input_objects=input_objects,
        output_objects=output_objects,
        model=module_storage,
        device=device,
    )

    output_shapes, input_shapes = estimate_tensor_shapes(
        module_storage=module_storage,
        fusion_to_output_mapping=fusion_to_output_mapping,
        example_input_data=example_batch.inputs,
    )

    tensor_cache: Dict[str, CachedTensor] = {}
    all_named_modules: dict[str, nn.Module] = dict(module_storage.named_modules())

    have_been_cached_mapping: dict[str, tuple[str, str]] = {}
    for config in all_configs:
        if not config.tensor_broker_config:
            continue

        for tmc in config.tensor_broker_config.message_configs:
            layer_path = tmc.layer_path
            layer_cache_target = tmc.layer_cache_target

            if tmc.cache_tensor:
                msg = (
                    f"When attempting to cache '{tmc.name}', "
                    f"layer path '{layer_path}' was not found in module storage. "
                    f"please check the layer path in the tensor broker config."
                )
                module = fuzzy_dict_lookup(
                    d=all_named_modules,
                    key=layer_path,
                    custom_prefix_message=msg,
                )

                attach_store_forward_hook(
                    module=module,
                    cache=tensor_cache,
                    layer_path=layer_path,
                    layer_cache_target=layer_cache_target,
                )
                have_been_cached_mapping[tmc.name] = (layer_path, layer_cache_target)

    have_been_used_from_cache = set()
    for config in all_configs:
        if not config.tensor_broker_config:
            continue

        for tmc in config.tensor_broker_config.message_configs:
            to_path = tmc.layer_path
            to_name = tmc.name

            if tmc.use_from_cache:
                for from_name in tmc.use_from_cache:

                    from_path, cache_target = have_been_cached_mapping[from_name]
                    message_name = f"{to_name}: {from_path}>>>{to_path}"
                    # . is not allowed in layer names in Torch
                    message_name = message_name.replace(".", "--")

                    if cache_target == "output":
                        from_shape_no_batch = output_shapes[from_path]
                    else:
                        from_shape_no_batch = input_shapes[from_path]

                    msg = (
                        f"When setting up message '{message_name}', "
                        f"the destination path "
                        f"'{to_path}' was not found as a module in the model."
                    )
                    to_shape_no_batch = fuzzy_dict_lookup(
                        d=input_shapes,
                        key=to_path,
                        custom_prefix_message=msg,
                    )

                    projection_layer, projected_shape = get_projection_layer(
                        from_shape_no_batch=from_shape_no_batch,
                        to_shape_no_batch=to_shape_no_batch,
                        cache_fusion_type=tmc.cache_fusion_type,
                        projection_type=tmc.projection_type,
                    )
                    have_been_used_from_cache.add(from_name)

                    fusion_layer = get_fusion_layer_wrapper(
                        projected_shape=projected_shape,
                        target_shape=to_shape_no_batch,
                        cache_fusion_type=tmc.cache_fusion_type,
                        projection_layer=projection_layer,
                        device=device,
                    )

                    tensor_broker_modules[message_name] = fusion_layer

                    attach_tensor_broker_module_injection(
                        target_module=all_named_modules[to_path],
                        tensor_broker_module=fusion_layer,
                        tensor_cache=tensor_cache,
                        tensor_cache_key=from_path,
                    )

    tensor_broker_modules = tensor_broker_modules.to(device=device)
    return tensor_broker_modules


def fuzzy_dict_lookup(
    d: Dict[str, Any],
    key: str,
    num_suggestions: int = 3,
    custom_prefix_message: Optional[str] = None,
) -> Any:
    if key in d:
        return d[key]

    close_matches = get_close_matches(key, d.keys(), n=num_suggestions, cutoff=0.6)

    error_parts = []

    if custom_prefix_message:
        error_parts.append(f"{custom_prefix_message}")

    error_parts.append(f"Key not found: '{key}'")

    if close_matches:
        suggestions = "\n".join(f"  - {match}" for match in close_matches)
        error_parts.append(f"Did you mean one of these?\n{suggestions}")
    else:
        error_parts.append("No close matches found.")

    sample_keys = list(d.keys())[:5]
    if sample_keys:
        examples = "\n".join(f"  - {k}" for k in sample_keys)
        error_parts.append(f"Example keys from the dictionary:\n{examples}")
        if len(d) > 5:
            error_parts.append(f"  ... ({len(d) - 5} more keys)")

    error_message = "\n".join(error_parts)
    logger.error(error_message)

    raise KeyError(f"Key '{key}' not found. See logs for details.")
