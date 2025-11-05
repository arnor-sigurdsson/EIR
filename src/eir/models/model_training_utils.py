from collections.abc import Callable, Generator, Iterable, Sequence
from copy import copy, deepcopy
from difflib import get_close_matches
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypedDict,
)

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

from eir.setup.output_setup import ComputedSurvivalOutputInfo, ComputedTabularOutputInfo
from eir.setup.schemas import SurvivalOutputTypeConfig
from eir.train_utils.utils import call_hooks_stage_iterable
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.train import Experiment  # noqa: F401
    from eir.train_utils.step_logic import (
        al_ids,
        al_input_batch,
        al_training_labels_target,
    )

# Aliases
al_dataloader_gathered_predictions = tuple[
    dict[str, dict[str, torch.Tensor]], Optional["al_training_labels_target"], list[str]
]
al_dataloader_gathered_raw = tuple[
    dict[str, torch.Tensor], "al_training_labels_target", Sequence[str]
]
al_lr_find_results = dict[str, list[float | list[float]]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def predict_on_batch(
    model: Module,
    inputs: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    model_device = next(model.parameters()).device
    device_as_str = str(model_device)
    inputs = recursive_to_device(obj=inputs, device=device_as_str)

    assert not model.training
    with torch.inference_mode():
        batch_outputs = model(inputs=inputs)

    return batch_outputs


class ColumnType(Enum):
    CON = "con"
    CAT = "cat"


def prepare_all_targets(
    output_objects: "al_output_objects_as_dict",
    device: str,
    labels: "al_training_labels_target",
) -> "al_training_labels_target":
    labels_prepared = parse_tabular_target_labels(
        output_objects=output_objects,
        device=device,
        labels=labels,
    )

    return labels_prepared


def recursive_to_device(
    obj: Any,
    device: str,
) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {
            key: recursive_to_device(obj=value, device=device)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [recursive_to_device(obj=value, device=device) for value in obj]
    return obj


def parse_tabular_target_labels(
    output_objects: "al_output_objects_as_dict",
    device: str,
    labels: "al_training_labels_target",
) -> "al_training_labels_target":
    labels_casted = copy(labels)

    def handle_tabular_object(
        output_object_: "ComputedTabularOutputInfo",
        output_name_: str,
    ) -> None:
        if output_name_ not in labels_casted:
            labels_casted[output_name_] = {}

        target_columns = output_object_.target_columns
        for column_type, list_of_cols_of_this_type in target_columns.items():
            for column_name in list_of_cols_of_this_type:
                cur_labels = labels[output_name_][column_name]

                if column_type == ColumnType.CON.value:
                    labels_casted[output_name_][column_name] = cur_labels.to(
                        dtype=torch.float,
                    )
                elif column_type == ColumnType.CAT.value:
                    cur_labels = replace_nan_and_cast_to_long(
                        cur_labels=cur_labels.to(dtype=torch.float),
                    )
                    labels_casted[output_name_][column_name] = cur_labels.to(
                        dtype=torch.long,
                    )

    def handle_survival_object(
        output_object_: "ComputedSurvivalOutputInfo",
        output_name_: str,
    ) -> None:
        """
        Mainly needed for MPS as to ensure we have float32 (as the often the default
        is float64, which is not supported by MPS).
        """

        output_type_info = output_object_.output_config.output_type_info
        assert isinstance(output_type_info, SurvivalOutputTypeConfig)

        time_column = output_type_info.time_column
        event_column = output_type_info.event_column

        cur_labels_time = labels[output_name_][time_column]

        loss_function = output_type_info.loss_function
        model_type = "cox" if loss_function == "CoxPHLoss" else "discrete"

        if model_type == "cox":
            labels_casted[output_name_][time_column] = cur_labels_time.to(
                dtype=torch.float,
            )
        else:
            cur_labels_time = replace_nan_and_cast_to_long(
                cur_labels=cur_labels_time.to(dtype=torch.float),
            )
            labels_casted[output_name_][time_column] = cur_labels_time.to(
                dtype=torch.float,
            )

        cur_label_event = labels[output_name_][event_column]
        cur_label_event = replace_nan_and_cast_to_long(
            cur_labels=cur_label_event.to(dtype=torch.float),
        )
        labels_casted[output_name_][event_column] = cur_label_event.to(
            dtype=torch.long,
        )

    for output_name, output_object in output_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                handle_tabular_object(
                    output_object_=output_object,
                    output_name_=output_name,
                )
            case ComputedSurvivalOutputInfo():
                handle_survival_object(
                    output_object_=output_object,
                    output_name_=output_name,
                )

    return labels_casted


def replace_nan_and_cast_to_long(
    cur_labels: torch.Tensor,
    replacement_value: int = -1,
) -> torch.Tensor:
    """
    Replace NaN values in a PyTorch tensor and cast it to a long integer
    (torch.long) type.

    When casting a tensor from a floating-point type (like float64) to an
    integer type (like long), NaN values need to be handled specially. In PyTorch,
    the behavior of casting NaN values to integers can be inconsistent
    across different environments, leading to unexpected results.

    For example:
    - On a local machine (e.g., Mac), NaN values may be converted to 0.
    - On some servers (e.g., GitHub Actions server), NaN values may be converted to
      -9223372036854775808, which is the minimum value for a 64-bit integer.

    This function explicitly replaces NaN values in the tensor with a defined integer
    before casting. This ensures consistent and defined behavior across
    different platforms.

    Example:
    >>> cur_labels = torch.tensor([nan, nan, 2., nan, 1.], dtype=torch.float32)
    >>> cur_labels.to(dtype=torch.long)
    tensor([0, 0, 2, 0, 1])
    or
    tensor([-9223372036854775808, -9223372036854775808, 2, -9223372036854775808, 1])
    >>> cur_labels = torch.tensor([nan, nan, 2., nan, 1.], dtype=torch.float32)
    >>> replace_nan_and_cast_to_long(cur_labels)
    tensor([-1, -1,  2, -1,  1])

    Note:
    The choice of `replacement_value` should be made carefully, in the case of
    categories, they are encoded from 0 to n-1, so using a negative value is
    potentially a reasonable choice.
    """

    replacement_tensor = torch.tensor(
        replacement_value,
        dtype=torch.float32,
    )

    cur_labels = cur_labels

    cur_labels = cur_labels.where(~cur_labels.isnan(), replacement_tensor)
    return cur_labels.to(dtype=torch.long)


def get_prediction_outputs_generator(
    data_loader: DataLoader,
    batch_prep_hook: Iterable[Callable],
    batch_prep_hook_kwargs: dict[str, Any],
    model: Module,
    with_labels: bool = True,
) -> Generator[al_dataloader_gathered_predictions]:
    assert not model.training
    for loader_batch in data_loader:
        state = call_hooks_stage_iterable(
            hook_iterable=batch_prep_hook,
            common_kwargs={"loader_batch": loader_batch, **batch_prep_hook_kwargs},
            state=None,
        )
        batch = state["batch"]

        inputs = batch.inputs
        target_labels = batch.target_labels
        ids = batch.ids

        outputs = predict_on_batch(model=model, inputs=inputs)

        target_labels_copy: al_training_labels_target | None
        target_labels_copy = deepcopy(target_labels) if with_labels else None
        ids_copy: list[str] = deepcopy(ids)

        yield outputs, target_labels_copy, ids_copy


def gather_data_loader_samples(
    data_loader: DataLoader,
    batch_prep_hook: Iterable[Callable],
    batch_prep_hook_kwargs: dict[str, Any],
    n_samples: int | None = None,
) -> al_dataloader_gathered_raw:
    all_input_batches = []
    all_label_batches = []
    ids_total = []

    for loader_batch in data_loader:
        state = call_hooks_stage_iterable(
            hook_iterable=batch_prep_hook,
            common_kwargs={"loader_batch": loader_batch, **batch_prep_hook_kwargs},
            state=None,
        )
        batch = state["batch"]

        inputs: al_input_batch = batch.inputs
        target_labels: al_training_labels_target = batch.target_labels
        ids: al_ids = batch.ids

        all_input_batches.append(inputs)
        all_label_batches.append(target_labels)
        ids_total += list(ids)

        if n_samples and len(ids_total) >= n_samples:
            ids_total = ids_total[:n_samples]
            break

    all_input_batches_stacked = _stack_list_of_batch_dicts(
        list_of_batch_dicts=all_input_batches
    )
    all_target_label_batches_stacked = stack_list_of_output_target_dicts(
        list_of_target_batch_dicts=all_label_batches
    )

    if n_samples:
        inputs_final: al_input_batch = {}
        for input_name in all_input_batches_stacked:
            input_subset = all_input_batches_stacked[input_name][:n_samples]
            inputs_final[input_name] = input_subset

        target_labels_final: al_training_labels_target = {}
        for output_name in all_target_label_batches_stacked:
            target_labels_final[output_name] = {}

            cur_output = all_target_label_batches_stacked[output_name]
            for target_name in cur_output:
                target_subset = cur_output[target_name][:n_samples]
                target_labels_final[output_name][target_name] = target_subset

    else:
        inputs_final, target_labels_final = (
            all_input_batches_stacked,
            all_target_label_batches_stacked,
        )

    return inputs_final, target_labels_final, ids_total


def stack_list_of_output_target_dicts(
    list_of_target_batch_dicts: list["al_training_labels_target"],
) -> "al_training_labels_target":
    """
    Spec:
        [batch_1, batch_2, batch_3]

        batch_1 =   {
                        'Output_Name_1': {'Target_Column_1': torch.Tensor(...)},
                        'Output_Name_2': {'Target_Column_2': torch.Tensor(...)},
                    }
                                                            with obs as rows in tensors
    """
    if not list_of_target_batch_dicts:
        return {}

    output_names = list(list_of_target_batch_dicts[0].keys())

    aggregated_batches: dict[str, dict[str, list[torch.Tensor]]] = {
        output_name: {} for output_name in output_names
    }

    for output_name in output_names:
        for target_name in list_of_target_batch_dicts[0].get(output_name, {}):
            aggregated_batches[output_name][target_name] = []

    for batch in list_of_target_batch_dicts:
        for output_name, output_dict in batch.items():
            for target_name, target_tensor in output_dict.items():
                if target_name in aggregated_batches[output_name]:
                    aggregated_batches[output_name][target_name].append(target_tensor)

    stacked_outputs: al_training_labels_target = {}
    for output_name, tensors_dict in aggregated_batches.items():
        cur_stacked_outputs: dict[str, torch.Tensor] = {}
        for key, list_of_tensors in tensors_dict.items():
            if list_of_tensors:
                cur_stacked_outputs[key] = torch.cat(list_of_tensors, dim=0)

        if cur_stacked_outputs:
            stacked_outputs[output_name] = cur_stacked_outputs

    return stacked_outputs


def _stack_list_of_batch_dicts(
    list_of_batch_dicts: list["al_input_batch"],
) -> "al_input_batch":
    """
    Spec:
        [batch_1, batch_2, batch_3]
        batch_1 =   {
                        'input_1': torch.Tensor(...), # with obs as rows
                        'input_2': torch.Tensor(...),
                    }
    """
    if not list_of_batch_dicts:
        return {}

    input_names = list(list_of_batch_dicts[0].keys())
    aggregated_batches: dict[str, list[torch.Tensor]] = {key: [] for key in input_names}

    for batch in list_of_batch_dicts:
        for input_name, tensor in batch.items():
            if input_name in aggregated_batches:
                aggregated_batches[input_name].append(tensor)

    stacked_inputs = {
        key: torch.cat(list_of_tensors, dim=0)
        for key, list_of_tensors in aggregated_batches.items()
        if list_of_tensors
    }

    return stacked_inputs


class ParamGroup(TypedDict, total=False):
    params: list[nn.Parameter]
    weight_decay: float
    force_adamw: bool


def _get_embedding_param_ids(model: nn.Module) -> set[int]:
    embedding_param_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters():
                embedding_param_ids.add(id(param))
    return embedding_param_ids


def add_wd_to_model_params(model: nn.Module, wd: float) -> list[ParamGroup]:
    """
    We want to skip adding weight decay to learnable activation parameters so as
    not to bias them towards 0.

    Parameters with dimensionality >= 2 (weight matrices, embeddings) will have
    weight decay applied, while parameters with dimensionality < 2 (biases,
    normalization parameters) will not.

    Embedding parameters are marked with force_adamw=True to ensure they use
    AdamW in composite optimizers like MuonAdamW.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    embedding_param_ids = _get_embedding_param_ids(model=model)

    decay_params = []
    embedding_params = []
    no_decay_params = []

    for _name, param in param_dict.items():
        if id(param) in embedding_param_ids:
            embedding_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_list = [
        ParamGroup(params=decay_params, weight_decay=wd),
        ParamGroup(params=embedding_params, weight_decay=wd, force_adamw=True),
        ParamGroup(params=no_decay_params, weight_decay=0.0),
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_embedding_params = sum(p.numel() for p in embedding_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    if wd > 0.0:
        logger.debug(
            f"Number of weight-decayed parameters: {num_decay_params:,} "
            f"({len(decay_params)} tensors)"
        )
        if embedding_params:
            logger.debug(
                f"Number of embedding parameters: "
                f"{num_embedding_params:,} ({len(embedding_params)} tensors)"
            )
        logger.debug(
            f"Number of non-decayed parameters: {num_no_decay_params:,} "
            f"({len(no_decay_params)} tensors)"
        )

    return param_list


def get_module_from_path(
    all_named_modules: dict[str, nn.Module],
    layer_path: str,
    custom_error_message: str | None = None,
    num_suggestions: int = 3,
) -> nn.Module:
    if layer_path in all_named_modules:
        return all_named_modules[layer_path]

    common_prefixes = ["_forward_module.", "module.", "_orig_mod."]
    for prefix in common_prefixes:
        prefixed_path = prefix + layer_path
        if prefixed_path in all_named_modules:
            logger.debug(
                f"Found layer with wrapper prefix: '{prefixed_path}' "
                f"(originally searched for '{layer_path}')"
            )
            return all_named_modules[prefixed_path]

    close_matches = get_close_matches(
        layer_path, all_named_modules.keys(), n=num_suggestions, cutoff=0.6
    )

    error_parts = []

    if custom_error_message:
        error_parts.append(f"{custom_error_message}")

    error_parts.append(f"Layer path not found: '{layer_path}'")

    if close_matches:
        suggestions = "\n".join(f"  - {match}" for match in close_matches)
        error_parts.append(f"Did you mean one of these?\n{suggestions}")
    else:
        error_parts.append("No close matches found.")

    sample_keys = list(all_named_modules.keys())[:5]
    if sample_keys:
        examples = "\n".join(f"  - {k}" for k in sample_keys)
        error_parts.append(f"Example layer paths:\n{examples}")
        if len(all_named_modules) > 5:
            error_parts.append(f"  ... ({len(all_named_modules) - 5} more paths)")

    error_message = "\n".join(error_parts)
    logger.error(error_message)

    raise KeyError(f"Layer path '{layer_path}' not found. See logs for details.")


def attach_caching_hook(
    module: nn.Module,
    cache: dict[str, torch.Tensor],
    cache_key: str,
    cache_target: Literal["input", "output"] = "output",
    detach: bool = True,
) -> Callable[[], None]:
    def hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor):
        tensor = args[0] if cache_target == "input" else output

        if detach:
            tensor = tensor.detach()

        cache[cache_key] = tensor

    handle = module.register_forward_hook(hook)

    def remove_hook():
        handle.remove()

    return remove_hook
