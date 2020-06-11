from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union, Dict, Callable, overload, TYPE_CHECKING

import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.data_load.label_setup import al_target_columns
from human_origins_supervised.models.extra_inputs_module import get_extra_inputs
from human_origins_supervised.train_utils.metrics import (
    calculate_losses,
    aggregate_losses,
)
from human_origins_supervised.train_utils.utils import get_run_folder

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from human_origins_supervised.train import (  # noqa: F401
        Config,
        al_training_labels_batch,
        al_training_labels_target,
        al_training_labels_extra,
    )

# Aliases
al_dloader_gathered_preds = Tuple[
    Dict[str, torch.Tensor], Union[List[str], Dict[str, torch.Tensor]], List[str]
]
al_dloader_gathered_raw = Tuple[torch.Tensor, "al_training_labels_batch", List[str]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def find_no_resblocks_needed(
    width: int, stride: int, first_stride_expansion: int
) -> List[int]:
    """
    Used in order to calculate / set up residual blocks specifications as a list
    automatically when they are not passed in as CL args, based on the minimum
    width after the resblock convolutions.

    We have 2 resblocks per channel depth until we have a total of 8 blocks,
    then the rest is put in the third depth index (following resnet convention).

    That is with a base channel depth of 32, we have these depths in the list:
    [32, 64, 128, 256].

    Examples
    ------
    3 blocks --> [2, 1]
    7 blocks --> [2, 2, 2, 1]
    10 blocks --> [2, 2, 4, 2]
    """

    min_size = 8 * stride
    # account for first conv
    cur_width = width // (stride * first_stride_expansion)

    resblocks = [0] * 4
    while cur_width >= min_size:
        cur_no_blocks = sum(resblocks)

        if cur_no_blocks >= 8:
            resblocks[2] += 1
        else:
            cur_index = cur_no_blocks // 2
            resblocks[cur_index] += 1

        cur_width = cur_width // stride

    return [i for i in resblocks if i != 0]


def predict_on_batch(
    model: Module, inputs: Tuple[torch.Tensor, ...]
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        val_outputs = model(*inputs)

    return val_outputs


def parse_target_labels(
    target_columns: al_target_columns, device: str, labels: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    target_columns_gen = get_target_columns_generator(target_columns)

    labels_casted = {}
    for column_type, column_name in target_columns_gen:
        cur_labels = labels[column_name]
        cur_labels = cur_labels.to(device=device)
        if column_type == "con":
            labels_casted[column_name] = cur_labels.to(dtype=torch.float).unsqueeze(1)
        elif column_type == "cat":
            labels_casted[column_name] = cur_labels.to(dtype=torch.long)

    return labels_casted


def gather_pred_outputs_from_dloader(
    data_loader: DataLoader,
    cl_args: Namespace,
    model: Module,
    device: str,
    with_labels: bool = True,
) -> al_dloader_gathered_preds:
    """
    Used to gather predictions from a dataloader, normally for evaluation – hence the
    assertion that we are in eval mode.
    """
    all_output_batches = []
    all_label_batches = []
    ids_total = []

    assert not model.training
    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device)
        inputs = inputs.to(dtype=torch.float32)

        extra_inputs = get_extra_inputs(
            cl_args=cl_args, model=model, labels=labels["extra_labels"]
        )

        outputs = predict_on_batch(model=model, inputs=(inputs, extra_inputs))

        all_output_batches.append(outputs)

        ids_total += [i for i in ids]

        if with_labels:
            all_label_batches.append(labels["target_labels"])

    if with_labels:
        all_label_batches = _stack_list_of_tensor_dicts(
            list_of_batch_dicts=all_label_batches
        )

    all_output_batches = _stack_list_of_tensor_dicts(
        list_of_batch_dicts=all_output_batches
    )

    return all_output_batches, all_label_batches, ids_total


def gather_dloader_samples(
    data_loader: DataLoader, device: str, n_samples: Union[int, None] = None
) -> al_dloader_gathered_raw:
    inputs_total = []
    all_label_batches = {"target_labels": [], "extra_labels": []}
    ids_total = []

    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)

        inputs_total += [i for i in inputs]
        ids_total += [i for i in ids]
        all_label_batches["target_labels"].append(labels["target_labels"])
        all_label_batches["extra_labels"].append(labels["extra_labels"])

        if n_samples:
            if len(inputs_total) >= n_samples:
                inputs_total = inputs_total[:n_samples]
                ids_total = ids_total[:n_samples]
                break

    all_label_batches["target_labels"] = _stack_list_of_tensor_dicts(
        all_label_batches["target_labels"]
    )
    all_label_batches["extra_labels"] = _stack_list_of_tensor_dicts(
        all_label_batches["extra_labels"]
    )

    if n_samples:
        for label_type in all_label_batches:
            for label_name in all_label_batches[label_type]:
                label_subset = all_label_batches[label_type][label_name][:n_samples]
                all_label_batches[label_type][label_name] = label_subset

    return torch.stack(inputs_total), all_label_batches, ids_total


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


def get_model_params(model: nn.Module, wd: float) -> List[Dict[str, Union[str, int]]]:
    """
    We want to skip adding weight decay to learnable activation parameters so as
    not to bias them towards 0.
    """
    _check_named_modules(model)

    params = []
    for name, param in model.named_parameters():
        cur_dict = {"params": param}

        if "act_" in name:
            cur_dict["weight_decay"] = 0
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


def test_lr_range(config: "Config") -> None:

    custom_lr_finder_objects = construct_lr_finder_custom_objects(config)

    logger.info(
        "Running learning rate range test and exiting, results will be " "saved to %s.",
        custom_lr_finder_objects.plot_output_path,
    )

    lr_finder = LRFinder(
        model=config.model,
        optimizer=config.optimizer,
        device=config.cl_args.device,
        lr_finder_custom_objects=custom_lr_finder_objects,
    )
    lr_finder.range_test(config.train_loader, end_lr=10, num_iter=300)
    lr_finder.plot()


@dataclass
class LRFinderCustomObjects:
    label_casting_func: Callable
    loss_func: Callable
    extra_inputs_hook: Callable
    plot_output_path: Path


def construct_lr_finder_custom_objects(config) -> LRFinderCustomObjects:
    c = config

    p_label_casting = partial(
        parse_target_labels, target_columns=c.target_columns, device=c.cl_args.device
    )

    p_extra_inputs_hook = partial(get_extra_inputs, cl_args=c.cl_args, model=c.model)

    p_calculate_losses = partial(_calculate_losses_and_average, criterions=c.criterions)

    plot_path = get_run_folder(c.cl_args.run_name) / "lr_search.png"

    custom_lr_finder_objects = LRFinderCustomObjects(
        label_casting_func=p_label_casting,
        loss_func=p_calculate_losses,
        extra_inputs_hook=p_extra_inputs_hook,
        plot_output_path=plot_path,
    )

    return custom_lr_finder_objects


def _calculate_losses_and_average(criterions, outputs, labels) -> torch.Tensor:
    all_losses = calculate_losses(criterions=criterions, labels=labels, outputs=outputs)
    average_loss = aggregate_losses(all_losses)

    return average_loss
