from typing import List, Tuple, Union, Dict
from argparse import Namespace

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader

from human_origins_supervised.models import embeddings
from human_origins_supervised.data_load.label_setup import al_label_dict

al_dloader_outputs = Tuple[torch.Tensor, Union[List[str], torch.LongTensor], List[str]]


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


def cast_labels(model_task: str, labels: torch.Tensor) -> torch.Tensor:
    if model_task == "reg":
        return labels.to(dtype=torch.float).unsqueeze(1)
    return labels.to(dtype=torch.long)


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
    outputs_total = []
    labels_total = []
    ids_total = []

    assert not model.training
    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)

        extra_inputs = embeddings.get_extra_inputs(cl_args, ids, labels_dict, model)

        outputs = predict_on_batch(model, (inputs, extra_inputs))

        outputs_total += [i for i in outputs]
        ids_total += [i for i in ids]

        if with_labels:
            labels = labels.to(device=device)
            labels_total += [i for i in labels]

    if with_labels:
        labels_total = torch.stack(labels_total)

    return torch.stack(outputs_total), labels_total, ids_total


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
    params = []
    for name, param in model.named_parameters():
        cur_dict = {"params": param}

        if "act_" in name:
            cur_dict["weight_decay"] = 0
        else:
            cur_dict["weight_decay"] = wd

        params.append(cur_dict)

    return params
