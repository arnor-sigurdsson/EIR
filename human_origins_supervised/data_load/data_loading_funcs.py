from collections import Counter
from typing import TYPE_CHECKING, List, Tuple, Union, Dict

import numpy as np
import torch
from aislib.misc_utils import get_logger
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from human_origins_supervised.data_load.datasets import ArrayDatasetBase

logger = get_logger(name=__name__, tqdm_compatible=True)


def get_weighted_random_sampler(
    train_dataset: "ArrayDatasetBase", target_columns: List[str]
):
    """
    Labels spec:

    {
        {
        ID1:
            {
                Label Column: Target Value,
                Extra Column 1: Extra Column 1 Value
                Extra Column 2: Extra Column 2 Value}
            }
        },
        ID2: {...}
    }

    The list comprehension is going over all the label dicts associated with the IDs,
    then just parsing the label (converting to int in the case of classification).
    """
    samples_weighted, num_sample_per_epoch = _aggregate_column_sampling_weights(
        train_dataset=train_dataset, target_columns=target_columns
    )

    logger.debug("Num samples per epoch: %d", num_sample_per_epoch)
    sampler = WeightedRandomSampler(
        samples_weighted, num_samples=num_sample_per_epoch, replacement=True
    )

    return sampler


def _aggregate_column_sampling_weights(
    train_dataset: "ArrayDatasetBase", target_columns: List[str]
) -> Tuple[torch.Tensor, int]:
    all_target_columns = {}
    for column in target_columns:
        cur_weight_dict = _get_column_sample_weights(
            train_dataset=train_dataset, target_column=column
        )
        all_target_columns[column] = cur_weight_dict

    all_weights = torch.stack(
        [i["samples_weighted"] for i in all_target_columns.values()], dim=1
    )
    all_weights_summed = all_weights.sum(dim=1)
    samples_per_epoch = sum(min(i["label_counts"]) for i in all_target_columns.values())

    return all_weights_summed, samples_per_epoch


def _get_column_sample_weights(
    train_dataset: "ArrayDatasetBase", target_column: str
) -> Dict[str, Union[torch.Tensor, List[int]]]:
    """
    TODO:   Optimize so we do just one pass over `train_dataset.samples` if this becomes
            a bottleneck.
    """

    labels = list(
        (i.labels["target_labels"][target_column] for i in train_dataset.samples)
    )

    label_counts = [i[1] for i in sorted(Counter(labels).items())]

    logger.debug("Label counts in column %s:  %s", target_column, label_counts)

    weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    samples_weighted = weights[labels]

    output_dict = {"samples_weighted": samples_weighted, "label_counts": label_counts}
    return output_dict


def make_random_snps_missing(
    array: torch.Tensor, percentage: float = 0.05, probability: float = 1.0
) -> torch.Tensor:
    random_draw = np.random.uniform()
    if random_draw > probability:
        return array

    n_snps = array.shape[2]
    n_to_drop = (int(n_snps * percentage),)
    random_to_drop = np.random.choice(n_snps, n_to_drop, replace=False)
    random_to_drop = torch.tensor(random_to_drop, dtype=torch.long)

    missing_arr = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float).reshape(-1, 1)
    array[:, :, random_to_drop] = missing_arr

    return array
