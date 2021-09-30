from collections import Counter
from statistics import mean
from typing import TYPE_CHECKING, List, Tuple, Union, Dict, Iterable

import torch
from aislib.misc_utils import get_logger
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from eir.data_load.datasets import (  # noqa: F401
        DatasetBase,
        Sample,
    )

logger = get_logger(name=__name__, tqdm_compatible=True)


# Type Aliases
al_sample_weight_and_counts = Dict[str, Union[torch.Tensor, List[int]]]


def get_weighted_random_sampler(samples: Iterable["Sample"], target_columns: List[str]):
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
    all_column_weights = _gather_column_sampling_weights(
        samples=samples, target_columns=target_columns
    )

    samples_weighted, num_sample_per_epoch = _aggregate_column_sampling_weights(
        all_target_columns_weights_and_counts=all_column_weights
    )

    logger.debug(
        "Num samples per epoch according to average target class counts in %s: %d",
        target_columns,
        num_sample_per_epoch,
    )
    sampler = WeightedRandomSampler(
        weights=samples_weighted, num_samples=num_sample_per_epoch, replacement=True
    )

    return sampler


def _gather_column_sampling_weights(
    samples: Iterable["Sample"], target_columns: Iterable[str]
) -> Dict[str, al_sample_weight_and_counts]:
    all_target_label_weight_dicts = {}

    for column in target_columns:
        cur_label_iterable = (i.target_labels[column] for i in samples)
        cur_weight_dict = _get_column_label_weights_and_counts(
            label_iterable=cur_label_iterable
        )

        logger.debug(
            "Label counts in column %s:  %s", column, cur_weight_dict["label_counts"]
        )

        all_target_label_weight_dicts[column] = cur_weight_dict

    return all_target_label_weight_dicts


def _get_column_label_weights_and_counts(
    label_iterable: Iterable[int],
) -> al_sample_weight_and_counts:
    """
    We have the assertion to make sure we have a unique integer for each label, starting
    with 0 as we use it to index into the weights directly.

    TODO:   Optimize so we do just one pass over `train_dataset.samples` if this becomes
            a bottleneck.
    """

    def _check_labels(label_list: List[int]):
        labels_set = set(label_list)
        assert sorted(list(labels_set)) == list(range(len(labels_set)))

    labels = list(label_iterable)
    _check_labels(label_list=labels)

    label_counts = [i[1] for i in sorted(Counter(labels).items())]

    weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    samples_weighted = weights[labels]

    output_dict = {"samples_weighted": samples_weighted, "label_counts": label_counts}
    return output_dict


def _aggregate_column_sampling_weights(
    all_target_columns_weights_and_counts: Dict[str, al_sample_weight_and_counts]
) -> Tuple[torch.Tensor, int]:
    """
    We sum up the normalized weights for each target column to create the final sampling
    weights.

    As for the samples per epoch, we take the average of the class counts per target,
    then sum those up.
    """

    all_weights = torch.stack(
        [i["samples_weighted"] for i in all_target_columns_weights_and_counts.values()],
        dim=1,
    )
    all_weights_summed = all_weights.sum(dim=1)

    samples_per_epoch = int(
        mean(
            mean(i["label_counts"])
            for i in all_target_columns_weights_and_counts.values()
        )
    )
    samples_per_epoch = min(len(all_weights_summed), samples_per_epoch)

    return all_weights_summed, samples_per_epoch
