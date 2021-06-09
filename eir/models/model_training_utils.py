from copy import deepcopy, copy
from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    overload,
    TYPE_CHECKING,
    Callable,
    Any,
    Sequence,
)

import plotly.express as px
import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from ignite.contrib.handlers import FastaiLRFinder
from ignite.engine import Engine
from torch import nn
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from eir.data_load.data_utils import get_target_columns_generator
from eir.data_load.label_setup import al_target_columns
from eir.train_utils.metrics import (
    calculate_prediction_losses,
    aggregate_losses,
)
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from eir.train import (  # noqa: F401
        Config,
        al_training_labels_batch,
        al_training_labels_target,
        al_training_labels_extra,
        Batch,
    )

# Aliases
al_dloader_gathered_preds = Tuple[
    Dict[str, torch.Tensor], Union[List[str], Dict[str, torch.Tensor]], List[str]
]
al_dloader_gathered_raw = Tuple[
    Dict[str, torch.Tensor], "al_training_labels_target", Sequence[str]
]
al_lr_find_results = Dict[str, List[Union[float, List[float]]]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def predict_on_batch(
    model: Module, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    assert not model.training
    with torch.no_grad():
        val_outputs = model(inputs)

    return val_outputs


def parse_target_labels(
    target_columns: al_target_columns, device: str, labels: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    target_columns_gen = get_target_columns_generator(target_columns=target_columns)

    labels_casted = {}
    for column_type, column_name in target_columns_gen:
        cur_labels = labels[column_name]
        cur_labels = cur_labels.to(device=device)
        if column_type == "con":
            labels_casted[column_name] = cur_labels.to(dtype=torch.float)
        elif column_type == "cat":
            labels_casted[column_name] = cur_labels.to(dtype=torch.long)

    return labels_casted


def gather_pred_outputs_from_dloader(
    data_loader: DataLoader,
    batch_prep_hook: Sequence[Callable],
    batch_prep_hook_kwargs: Dict[str, Any],
    model: Module,
    with_labels: bool = True,
) -> al_dloader_gathered_preds:
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

    if with_labels:
        all_label_batches = _stack_list_of_tensor_dicts(
            list_of_batch_dicts=all_label_batches
        )

    all_output_batches = _stack_list_of_tensor_dicts(
        list_of_batch_dicts=all_output_batches
    )

    return all_output_batches, all_label_batches, ids_total


def gather_dloader_samples(
    data_loader: DataLoader,
    batch_prep_hook: Sequence[Callable],
    batch_prep_hook_kwargs: Dict[str, Any],
    n_samples: Union[int, None] = None,
) -> al_dloader_gathered_raw:

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

        inputs: Dict[str, torch.Tensor] = batch.inputs
        target_labels: Dict[str, torch.Tensor] = batch.target_labels
        ids: Sequence[str] = batch.ids

        all_input_batches.append(inputs)
        all_label_batches.append(target_labels)
        ids_total += [i for i in ids]

        if n_samples:
            if len(ids_total) >= n_samples:
                ids_total = ids_total[:n_samples]
                break

    all_label_batches = _stack_list_of_tensor_dicts(
        list_of_batch_dicts=all_label_batches
    )
    all_input_batches = _stack_list_of_tensor_dicts(
        list_of_batch_dicts=all_input_batches
    )

    if n_samples:
        inputs_final, target_labels_final = {}, {}

        for input_name in all_input_batches.keys():
            input_subset = all_input_batches[input_name][:n_samples]
            inputs_final[input_name] = input_subset

        for target_name in all_label_batches.keys():
            target_subset = all_label_batches[target_name][:n_samples]
            target_labels_final[target_name] = target_subset

    else:
        inputs_final, target_labels_final = all_input_batches, all_label_batches

    return inputs_final, target_labels_final, ids_total


@overload
def _stack_list_of_tensor_dicts(
    list_of_batch_dicts: List["al_training_labels_target"],
) -> "al_training_labels_target":
    ...


@overload
def _stack_list_of_tensor_dicts(
    list_of_batch_dicts: List["al_training_labels_extra"],
) -> "al_training_labels_extra":
    ...


def _stack_list_of_tensor_dicts(list_of_batch_dicts):
    """
    Spec:
        [batch_1, batch_2, batch_3]

        batch_1 =   {
                        'Target_Column_1': torch.Tensor(...), # with obs as rows
                        'Target_Column_2': torch.Tensor(...),
                    }
    """

    def _do_stack(
        list_of_elements: List[Union[torch.Tensor, torch.LongTensor, str]]
    ) -> Union[torch.Tensor, List[str]]:
        # check that they're all the same type
        list_types = set(type(i) for i in list_of_elements)
        assert len(list_types) == 1

        are_tensors = isinstance(list_of_elements[0], (torch.Tensor, torch.LongTensor))
        if are_tensors:
            return torch.stack(list_of_elements)

        return list_of_elements

    target_columns = list_of_batch_dicts[0].keys()
    aggregated_batches = {key: [] for key in target_columns}

    for batch in list_of_batch_dicts:
        assert set(batch.keys()) == target_columns

        for column in batch.keys():
            cur_column_batch = batch[column]
            aggregated_batches[column] += [i for i in cur_column_batch]

    stacked_outputs = {
        key: _do_stack(list_of_elements)
        for key, list_of_elements in aggregated_batches.items()
    }

    return stacked_outputs


def add_wd_to_model_params(
    model: nn.Module, wd: float
) -> List[Dict[str, Union[str, float]]]:
    """
    We want to skip adding weight decay to learnable activation parameters so as
    not to bias them towards 0.

    TODO:   Split this function in two, one to get the parameters and one to add the
            WD to them. Possibly we have to do it in-place here, not copy as we have
            tensors.

    Note: Since we are adding the weight decay manually here, the optimizer does not
    touch the parameter group weight decay at initialization.
    """
    _check_named_modules(model)

    params = []
    for name, param in model.named_parameters():
        cur_dict = {"params": param}

        if "act_" in name:
            cur_dict["weight_decay"] = 0.0
        else:
            cur_dict["weight_decay"] = wd

        params.append(cur_dict)

    return params


def _check_named_modules(model: nn.Module):
    """
    We have this function as a safeguard to check that activations that have learnable
    parameters are named correctly (so that WD is not applied to them). Also, we want
    to make sure we don't have modules that are named 'incorrectly' and have the WD
    skipped when they should have it.
    """

    for name, module in model.named_modules():
        if "act_" in name:
            assert isinstance(module, (Swish, nn.PReLU))

        if isinstance(module, (Swish, nn.PReLU)):
            assert "act_" in name, name


def run_lr_find(
    trainer_engine: Engine,
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    output_folder: Path,
):
    lr_range_results, lr_suggestion = get_lr_range_results(
        trainer_engine=trainer_engine,
        train_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
    )

    lr_range_results_parsed = _parse_out_lr_find_multiple_param_groups(
        lr_find_results=lr_range_results
    )
    lr_suggestion_parsed = _parse_out_lr_find_multiple_param_groups_suggestion(
        lr_finder_suggestion=lr_suggestion
    )

    plot_lr_find_results(
        lr_find_results=lr_range_results_parsed,
        lr_suggestion=lr_suggestion_parsed,
        outfolder=output_folder,
    )


def get_lr_range_results(
    trainer_engine: Engine,
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: Optimizer,
    num_iter: int = 500,
) -> Tuple[al_lr_find_results, List[float]]:
    """
    We do a little hack with max_epochs and epoch_length because we don't pass that
    to the original trainer when running a normal training instance, which uses the
    default of max_epochs = 1. This is normally not a problem, as 1 epoch is
    generally more than for example 300 iterations, which should be enough for the LR
    test.

    However, in the cases where we have a small epoch length, the LR find will not
    run to completion as it only runs for one epoch. Hence we need to patch the
    max_epochs here for the test.
    """
    lr_finder = FastaiLRFinder()

    to_save = {"optimizer": optimizer, "model": model}
    with lr_finder.attach(
        trainer_engine,
        to_save=to_save,
        output_transform=lambda x: x["average"]["loss-average"],
        num_iter=num_iter,
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

        return deepcopy(lr_finder.get_results()), deepcopy(lr_finder.lr_suggestion())


def _parse_out_lr_find_multiple_param_groups(lr_find_results: al_lr_find_results):
    lr_find_results_copy = copy(lr_find_results)
    lr_steps = lr_find_results["lr"]
    logger.info(
        "LR Find: The model has %d parameter groups which can have different "
        "learning rates depending on implementation. For the LR find, "
        "we are using the LR defined in group 0, so make sure that is taken "
        "into consideration when analysing LR find results, the supplied LR "
        "for training might have to be scaled accordingly.",
        len(lr_steps[0]),
    )
    first_param_groups_lrs = [i[0] for i in lr_steps]

    lr_find_results_copy["lr"] = first_param_groups_lrs
    return lr_find_results_copy


def _parse_out_lr_find_multiple_param_groups_suggestion(
    lr_finder_suggestion: List[float],
):
    return lr_finder_suggestion[0]


def plot_lr_find_results(
    lr_find_results: al_lr_find_results,
    lr_suggestion: float,
    outfolder: Path,
):
    lr_values = copy(lr_find_results["lr"])
    loss_values = copy(lr_find_results["loss"])

    fig = px.line(
        x=lr_values,
        y=loss_values,
        log_x=True,
        title="Learning Rate Search",
    )
    fig.update_layout(
        xaxis_title="Learning Rate ",
        yaxis_title="Loss",
    )

    fig.add_shape(
        dict(
            type="line",
            x0=lr_suggestion,
            y0=0,
            x1=lr_suggestion,
            y1=max(loss_values),
            line=dict(color="Red", width=1),
        )
    )

    fig.write_html(str(outfolder / "lr_search.html"))


def _calculate_losses_and_average(criterions, outputs, labels) -> torch.Tensor:
    all_losses = calculate_prediction_losses(
        criterions=criterions, targets=labels, inputs=outputs
    )
    average_loss = aggregate_losses(all_losses)

    return average_loss
