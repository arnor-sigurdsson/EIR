from typing import List, Tuple, Union

import torch


def find_no_resblocks_needed(width: int, stride: int, min_size: int = 32) -> List[int]:
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
    # account for first resblock
    cur_width = width // stride

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


def predict_on_batch(model, inputs):
    with torch.no_grad():
        val_outputs = model(inputs)

    return val_outputs


def cast_labels(model_task, labels):
    if model_task == "reg":
        return labels.to(dtype=torch.float)
    return labels.to(dtype=torch.long)


def gather_pred_outputs_from_dloader(
    data_loader, model, device, with_labels=True
) -> Tuple[torch.Tensor, Union[List[str], torch.LongTensor], List[str]]:
    model.eval()
    outputs_total = []
    labels_total = []
    ids_total = []

    for inputs, labels, ids in data_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)

        outputs = predict_on_batch(model, inputs)

        outputs_total += [i for i in outputs]
        ids_total += [i for i in ids]

        if with_labels:
            labels = labels.to(device=device)
            labels_total += [i for i in labels]

    if with_labels:
        labels_total = torch.stack(labels_total)

    model.train()
    return torch.stack(outputs_total), labels_total, ids_total


def gather_dloader_samples(data_loader, device, n_samples=None):
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

    return torch.stack(inputs_total), torch.stack(labels_total), ids_total
