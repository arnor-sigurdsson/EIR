from copy import deepcopy, copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    TYPE_CHECKING,
    Callable,
    Any,
    Sequence,
)

import matplotlib.pyplot as plt
import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from ignite.engine import Engine
from ignite.handlers.lr_finder import FastaiLRFinder
from torch import nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from eir.models.meta.meta import MetaModel
from eir.setup.output_setup import (
    ComputedTabularOutputInfo,
)
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from eir.train import (  # noqa: F401
        Experiment,
    )
    from eir.train_utils.step_logic import (
        al_training_labels_target,
        al_input_batch,
        al_ids,
    )
    from eir.setup.output_setup import (
        al_output_objects_as_dict,
    )
    from eir.models.model_setup_modules.meta_setup import al_meta_model

# Aliases
al_dataloader_gathered_predictions = Tuple[
    Dict[str, Dict[str, torch.Tensor]], "al_training_labels_target", List[str]
]
al_dataloader_gathered_raw = Tuple[
    Dict[str, torch.Tensor], "al_training_labels_target", Sequence[str]
]
al_lr_find_results = Dict[str, List[Union[float, List[float]]]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def predict_on_batch(
    model: Module, inputs: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, torch.Tensor]]:
    assert not model.training
    with torch.no_grad():
        val_outputs = model(inputs=inputs)

    return val_outputs


class ColumnType(Enum):
    CON = "con"
    CAT = "cat"


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
                        dtype=torch.float, device=device
                    )
                elif column_type == ColumnType.CAT.value:
                    labels_casted[output_name_][column_name] = cur_labels.to(
                        dtype=torch.long, device=device
                    )

    for output_name, output_object in output_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                handle_tabular_object(
                    output_object_=output_object,
                    output_name_=output_name,
                )

    return labels_casted


def gather_prediction_outputs_from_dataloader(
    data_loader: DataLoader,
    batch_prep_hook: Sequence[Callable],
    batch_prep_hook_kwargs: Dict[str, Any],
    model: Module,
    with_labels: bool = True,
) -> al_dataloader_gathered_predictions:
    """
    Used to gather predictions from a dataloader, normally for evaluation – hence the
    assertion that we are in eval mode.

    Why the deepcopy when appending labels? See:
    https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189

    TODO: Use hook model forward here.
    """
    all_output_batches = []
    all_label_batches = []
    ids_total = []

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

        all_output_batches.append(outputs)

        ids_total += [i for i in deepcopy(ids)]

        if with_labels:
            all_label_batches.append(deepcopy(target_labels))

        del inputs
        del target_labels
        del ids

    all_label_batches_stacked = {}
    if with_labels:
        all_label_batches_stacked = _stack_list_of_output_target_dicts(
            list_of_target_batch_dicts=all_label_batches
        )

    all_output_batches_stacked = _stack_list_of_output_target_dicts(
        list_of_target_batch_dicts=all_output_batches
    )

    return all_output_batches_stacked, all_label_batches_stacked, ids_total


def gather_data_loader_samples(
    data_loader: DataLoader,
    batch_prep_hook: Sequence[Callable],
    batch_prep_hook_kwargs: Dict[str, Any],
    n_samples: Union[int, None] = None,
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

        inputs: "al_input_batch" = batch.inputs
        target_labels: "al_training_labels_target" = batch.target_labels
        ids: "al_ids" = batch.ids

        all_input_batches.append(inputs)
        all_label_batches.append(target_labels)
        ids_total += [i for i in ids]

        if n_samples:
            if len(ids_total) >= n_samples:
                ids_total = ids_total[:n_samples]
                break

    all_input_batches_stacked = _stack_list_of_batch_dicts(
        list_of_batch_dicts=all_input_batches
    )
    all_target_label_batches_stacked = _stack_list_of_output_target_dicts(
        list_of_target_batch_dicts=all_label_batches
    )

    if n_samples:
        inputs_final: "al_input_batch" = {}
        for input_name in all_input_batches_stacked.keys():
            input_subset = all_input_batches_stacked[input_name][:n_samples]
            inputs_final[input_name] = input_subset

        target_labels_final: "al_training_labels_target" = {}
        for output_name in all_target_label_batches_stacked.keys():
            target_labels_final[output_name] = {}

            cur_output = all_target_label_batches_stacked[output_name]
            for target_name in cur_output.keys():
                target_subset = cur_output[target_name][:n_samples]
                target_labels_final[output_name][target_name] = target_subset

    else:
        inputs_final, target_labels_final = (
            all_input_batches_stacked,
            all_target_label_batches_stacked,
        )

    return inputs_final, target_labels_final, ids_total


def _stack_list_of_output_target_dicts(
    list_of_target_batch_dicts: List["al_training_labels_target"],
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

    output_names = list_of_target_batch_dicts[0].keys()
    aggregated_batches: dict[str, dict[str, list[torch.Tensor]]] = {
        output_name: {} for output_name in output_names
    }
    for output_name in output_names:
        cur_output_targets = list_of_target_batch_dicts[0][output_name]
        for target_name in cur_output_targets.keys():
            aggregated_batches[output_name][target_name] = []

    for batch in list_of_target_batch_dicts:
        assert set(batch.keys()) == output_names

        for output_name in batch.keys():
            cur_output_batch = batch[output_name]
            for target_name in cur_output_batch.keys():
                cur_column_batch = cur_output_batch[target_name]
                cur_batch_value = [i for i in cur_column_batch]

                aggregated_batches[output_name][target_name] += cur_batch_value

    stacked_outputs: "al_training_labels_target" = {}
    for output_name, output_dict in aggregated_batches.items():
        cur_stacked_outputs = {
            key: _do_stack(list_of_elements=list_of_elements)
            for key, list_of_elements in output_dict.items()
        }
        stacked_outputs[output_name] = cur_stacked_outputs

    return stacked_outputs


def _stack_list_of_batch_dicts(
    list_of_batch_dicts: List["al_input_batch"],
) -> "al_input_batch":
    """
    Spec:
        [batch_1, batch_2, batch_3]
        batch_1 =   {
                        'input_1': torch.Tensor(...), # with obs as rows
                        'input_2': torch.Tensor(...),
                    }
    """

    target_columns = list_of_batch_dicts[0].keys()
    aggregated_batches: dict[str, list[torch.Tensor]] = {
        key: [] for key in target_columns
    }

    for batch in list_of_batch_dicts:
        assert set(batch.keys()) == target_columns

        for column in batch.keys():
            cur_column_batch = batch[column]
            aggregated_batches[column] += [i for i in cur_column_batch]

    stacked_inputs = {
        key: _do_stack(list_of_elements=list_of_elements)
        for key, list_of_elements in aggregated_batches.items()
    }

    return stacked_inputs


def _do_stack(
    list_of_elements: List[torch.Tensor],
) -> torch.Tensor:
    # check that they're all the same type
    list_types = set(type(i) for i in list_of_elements)
    assert len(list_types) == 1

    return torch.stack(list_of_elements)


def add_wd_to_model_params(
    model: nn.Module, wd: float
) -> List[Dict[str, nn.Parameter | float]]:
    """
    We want to skip adding weight decay to learnable activation parameters so as
    not to bias them towards 0.

    TODO:   Split this function in two, one to get the parameters and one to add the
            WD to them. Possibly we have to do it in-place here, not copy as we have
            tensors.

    Note: Since we are adding the weight decay manually here, the optimizer does not
    touch the parameter group weight decay at initialization.
    """
    _check_named_modules(model=model)

    param_list = []
    for name, param in model.named_parameters():
        cur_dict: dict[str, nn.Parameter | float]
        cur_dict = {"params": param}

        if "act_" in name:
            cur_dict["weight_decay"] = 0.0
        else:
            cur_dict["weight_decay"] = wd

        param_list.append(cur_dict)

    return param_list


def _check_named_modules(model: nn.Module):
    """
    We have this function as a safeguard to check that activations that have learnable
    parameters are named correctly (so that WD is not applied to them). Also, we want
    to make sure we don't have modules that are named 'incorrectly' and have the WD
    skipped when they should have it.
    """

    for name, module in model.named_modules():
        if name.startswith("act_"):
            assert isinstance(module, (Swish, nn.PReLU)), (name, module)

        if isinstance(module, (Swish, nn.PReLU)):
            assert "act_" in name, name


def run_lr_find(
    trainer_engine: Engine,
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    output_folder: Path,
):
    lr_find_results = get_lr_range_results(
        trainer_engine=trainer_engine,
        train_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
    )

    plot_lr_find_results(
        lr_values=lr_find_results.lr_history,
        loss_values=lr_find_results.loss_history,
        lr_suggestion=lr_find_results.lr_suggestion,
        output_folder=output_folder,
    )


@dataclass
class LRFindResults:
    lr_history_groups: list[list[float]]
    lr_history: list[float]
    loss_history: list[float]
    lr_suggestion: float


def get_lr_range_results(
    trainer_engine: Engine,
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    num_iter: int = 500,
) -> LRFindResults:
    """
    We do a little hack with max_epochs and epoch_length because we don't pass that
    to the original trainer when running a normal training instance, which uses the
    default of max_epochs = 1. This is normally not a problem, as 1 epoch is
    generally more than for example 300 iterations, which should be enough for the LR
    test.

    However, in the cases where we have a small epoch length, the LR find will not
    run to completion as it only runs for one epoch. Hence, we need to patch the
    max_epochs here for the test.
    """
    lr_finder = FastaiLRFinder()

    def _extract_loss(x: dict[str, dict[str, dict[str, float]]]) -> float:
        return x["average"]["average"]["loss-average"]

    to_save = {"optimizer": optimizer, "model": model}
    with lr_finder.attach(
        trainer_engine,
        to_save=to_save,
        output_transform=_extract_loss,
        num_iter=num_iter,
        start_lr=1e-7,
    ) as trainer_with_lr_finder:
        logger.info("Running LR range test for max %d iterations.", num_iter)

        default_max_epochs = trainer_with_lr_finder.state.max_epochs
        trainer_with_lr_finder.state.max_epochs = 100
        trainer_with_lr_finder.state.epoch_length = len(train_dataloader)

        assert (
            trainer_with_lr_finder.state.max_epochs
            * trainer_with_lr_finder.state.epoch_length
        ) >= num_iter

        trainer_with_lr_finder.run(train_dataloader)
        trainer_with_lr_finder.state.max_epochs = default_max_epochs

        lr_results = deepcopy(lr_finder.get_results())
        lr_suggestions_all_groups = deepcopy(lr_finder.lr_suggestion())
        lr_suggestion_single = lr_suggestions_all_groups[0]

        logger.info(
            "LR Find: The model has %d parameter groups which can have different "
            "learning rates depending on implementation. For the LR find, "
            "we are using the LR defined in group 0, so make sure that is taken "
            "into consideration when analysing LR find results, the supplied LR "
            "for training might have to be scaled accordingly.",
            len(lr_results["lr"][0]),
        )

        return LRFindResults(
            lr_history_groups=lr_results["lr"],
            lr_history=[i[0] for i in lr_results["lr"]],
            loss_history=lr_results["loss"],
            lr_suggestion=lr_suggestion_single,
        )


def _parse_out_lr_find_multiple_param_groups_suggestion(
    lr_finder_suggestion: List[float],
) -> float:
    return lr_finder_suggestion[0]


def plot_lr_find_results(
    lr_values: list[float],
    loss_values: list[float],
    lr_suggestion: float,
    output_folder: Path,
) -> None:
    plt.plot(lr_values, loss_values)
    plt.xscale("log")
    plt.title(f"Learning Rate Search\nLR Suggestion: {lr_suggestion:.2e}")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.axvline(x=lr_suggestion, color="red", linewidth=1)

    plt.savefig(str(output_folder / "lr_search.pdf"))


def trace_eir_model(
    meta_model: "al_meta_model", example_inputs: Dict[str, Any]
) -> nn.Module:
    """
    Optimally we would like to trace the whole meta_model in one go, but since torch
    currently does not like / support nested dict outputs when tracing a model,
    we'll opt for tracing the individual modules like below for now (assuming it's
    better than nothing).
    """

    if not isinstance(meta_model, MetaModel):
        raise ValueError("Tracing is only supported for MetaModel instances")

    meta_model.eval()

    for name, module in meta_model.named_modules():
        if hasattr(module, "script_submodules_for_tracing"):
            module.script_submodules_for_tracing()

    with torch.no_grad():
        traced_input_modules = nn.ModuleDict()
        feature_extractors_out = {}
        for module_name, module_input in example_inputs.items():
            module = meta_model.input_modules[module_name]
            traced_input_module = torch.jit.trace_module(
                mod=module,
                inputs={"forward": module_input},
                strict=False,
            )
            traced_input_modules[module_name] = traced_input_module
            feature_extractors_out[module_name] = module(module_input)

        traced_fusion_modules = nn.ModuleDict()
        fused_features = {}
        for fusion_name, fusion_module in meta_model.fusion_modules.items():
            traced_fusion_module = torch.jit.trace_module(
                mod=fusion_module,
                inputs={"forward": feature_extractors_out},
                strict=False,
            )
            traced_fusion_modules[fusion_name] = traced_fusion_module
            fused_features[fusion_name] = fusion_module(feature_extractors_out)

        traced_output_modules = nn.ModuleDict()
        for output_module_name, output_module in meta_model.output_modules.items():
            cur_fusion_target = meta_model.fusion_to_output_mapping[output_module_name]
            corresponding_fused_features = fused_features[cur_fusion_target]

            traced_output_module = torch.jit.trace_module(
                mod=output_module,
                inputs={"forward": corresponding_fused_features},
                strict=False,
            )
            traced_output_modules[output_module_name] = traced_output_module

    traced_meta_model = meta_model.__class__(
        input_modules=traced_input_modules,
        fusion_modules=traced_fusion_module,
        output_modules=traced_output_modules,
        fusion_to_output_mapping=meta_model.fusion_to_output_mapping,
    )

    return traced_meta_model
