from argparse import Namespace
from functools import partial
from typing import List, Tuple, Union, Dict, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from aislib.misc_utils import get_logger
from human_origins_supervised.data_load.label_setup import al_label_dict
from human_origins_supervised.data_load.datasets import al_target_columns
from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.models.extra_inputs_module import get_extra_inputs
from human_origins_supervised.train_utils.utils import get_run_folder

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

# Aliases
al_dloader_outputs = Tuple[
    Dict[str, torch.Tensor], Union[List[str], Dict[str, torch.Tensor]], List[str]
]

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


def predict_on_batch(model: Module, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    with torch.no_grad():
        val_outputs = model(*inputs)

    return val_outputs


def cast_labels(
    target_columns: al_target_columns, device: str, labels: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:

    target_columns_gen = get_target_columns_generator(target_columns)

    labels_casted = {}
    for column_type, column_name in target_columns_gen:
        cur_labels = labels[column_name]
        cur_labels.to(device=device)
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
    labels_dict: al_label_dict,
    with_labels: bool = True,
) -> al_dloader_outputs:
    """
    Used to gather predictions from a dataloader, normally for evaluation – hence the
    assertion that we are in eval mode.
    """
    all_output_batches = []
    all_label_batches = []
    ids_total = []

    assert not model.training
    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)

        extra_inputs = get_extra_inputs(cl_args, ids, labels_dict, model)

        outputs = predict_on_batch(model, (inputs, extra_inputs))

        # TODO: Update this, outputs is now dict.
        all_output_batches.append(outputs)

        ids_total += [i for i in ids]

        if with_labels:
            breakpoint()
            # TODO: Make sure this gets called after updating.
            # labels = labels.to(device=device)
            all_label_batches.append(labels)

    breakpoint()
    if with_labels:
        all_label_batches = torch.stack(all_label_batches)

    # TODO: Add stacking call at the end of this func.
    # We need to merge all the dictionaries of outputs and labels here
    # So the final will be the correctly ordered outputs and labels according to ids.
    return torch.stack(all_output_batches), all_label_batches, ids_total


def stack_list_of_tensor_dicts(
    list_of_dicts: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Spec:
        [batch_1, batch_2, batch_3]

        batch_1 =   {
                        'Target_Column_1': torch.Tensor(...),
                        'Target_Column_2': torch.Tensor(...),
                    }
    """
    stacked_outputs = {}
    for key in list_of_dicts[0].keys():
        aggregated_key_values = [small_dict[key] for small_dict in list_of_dicts]
        stacked_outputs[key] = torch.stack(aggregated_key_values)

    return stacked_outputs


def gather_dloader_samples(
    data_loader: DataLoader, device: str, n_samples: Union[int, None] = None
) -> al_dloader_outputs:
    inputs_total = []
    labels_total = []
    ids_total = []

    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device)

        inputs_total += [i for i in inputs]
        labels_total += [i for i in labels]
        ids_total += [i for i in ids]

        if n_samples:
            if len(inputs_total) >= n_samples:
                inputs_total = inputs_total[:n_samples]
                labels_total = labels_total[:n_samples]
                ids_total = ids_total[:n_samples]
                break

    return torch.stack(inputs_total), labels_total, ids_total


def get_model_params(model: nn.Module, wd: float) -> List[Dict[str, Union[str, int]]]:
    """
    We want to skip adding weight decay to learnable activation parameters so as
    not to bias them towards 0.
    """
    params = []
    for name, param in model.named_parameters():
        cur_dict = {"params": param}

        if "act_" in name:
            cur_dict["weight_decay"] = 0
        else:
            cur_dict["weight_decay"] = wd

        params.append(cur_dict)

    return params


def test_lr_range(config: "Config") -> None:

    c = config

    extra_inputs_hook = partial(
        get_extra_inputs, cl_args=c.cl_args, labels_dict=c.labels_dict, model=c.model
    )
    plot_path = get_run_folder(c.cl_args.run_name) / "lr_search.png"

    logger.info(
        "Running learning rate range test and exiting, results will be " "saved to %s.",
        plot_path,
    )

    lr_finder = LRFinder(
        model=c.model,
        optimizer=c.optimizer,
        criterion=c.criterions,
        device=c.cl_args.device,
        extra_inputs_hook=extra_inputs_hook,
        plot_output_path=plot_path,
    )
    lr_finder.range_test(c.train_loader, end_lr=10, num_iter=300)
    lr_finder.plot()
