from typing import Dict, Union, Tuple

import torch


def split_target_and_extra_labels(
    all_labels_batch: Dict[str, Union[torch.tensor, str]], label_column: str
) -> Tuple[torch.Tensor, Dict[str, str]]:
    target_labels = all_labels_batch[label_column]
    extra_labels = {k: v for k, v in all_labels_batch.items() if k != label_column}

    return target_labels, extra_labels


def get_extra_labels_from_ids(labels_dict, cur_ids, label_column):
    """
    Returns a batch in same order as cur_ids.
    """
    extra_labels = []
    for sample_id in cur_ids:
        cur_labels_all = labels_dict.get(sample_id)
        cur_labels_extra = {
            k: v for k, v in cur_labels_all.items() if k != label_column
        }
        extra_labels.append(cur_labels_extra)

    return extra_labels
