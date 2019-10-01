from collections import Counter
from typing import TYPE_CHECKING

import torch
import numpy as np
from aislib.misc_utils import get_logger
from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from human_origins_supervised.data_load.datasets import ArrayDatasetBase

logger = get_logger(__name__)


def get_weighted_random_sampler(train_dataset: "ArrayDatasetBase"):
    """
    TODO: Use label column here after we add additional columns in dataset label dict.

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
    label_parser = train_dataset.parse_label
    labels = [
        label_parser(label).item() for label in train_dataset.labels_dict.values()
    ]

    label_counts = [i[1] for i in sorted(Counter(labels).items())]

    logger.debug("Using weighted sampling with label counts %s", label_counts)

    weights = 1.0 / torch.tensor(label_counts, dtype=torch.float32)
    samples_weighted = weights[labels]

    num_sample_per_epoch = min(label_counts) * len(weights)
    logger.debug("Num samples per epoch: %d", num_sample_per_epoch)
    sampler = WeightedRandomSampler(
        samples_weighted, num_samples=num_sample_per_epoch, replacement=False
    )

    return sampler


def make_random_snps_missing(array, percentage=0.05):
    n_snps = array.shape[2]
    n_to_drop = (int(n_snps * percentage),)
    random_to_drop = np.random.choice(n_snps, n_to_drop, replace=False)
    random_to_drop = torch.tensor(random_to_drop, dtype=torch.long)

    missing_arr = torch.tensor([0, 0, 0, 1], dtype=torch.uint8).reshape(-1, 1)
    array[:, :, random_to_drop] = missing_arr

    return array
