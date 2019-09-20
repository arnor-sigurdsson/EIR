from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler

from aislib.misc_utils import get_logger
from human_origins_supervised.data_load import datasets

logger = get_logger(__name__)


def get_weighted_random_sampler(train_dataset: datasets.ArrayDatasetBase, label_column):
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

    sampler = WeightedRandomSampler(
        samples_weighted, num_samples=len(train_dataset), replacement=True
    )

    return sampler
