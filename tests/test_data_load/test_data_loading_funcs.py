from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
from hypothesis import given
from hypothesis.strategies import lists, integers

from human_origins_supervised.data_load import data_loading_funcs


def test_get_weighted_random_sampler():
    pass


def test_aggregate_column_sampling_weights():
    pass


def test_gather_column_sampling_weights():
    pass


@given(
    test_labels=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: x + list(range(10)))
)
def test_get_column_sample_weights(test_labels):
    """
    Why do the `map` to make sure that the list contains at least one copy of all
    unique elements, which
    """
    label_weight_dict = data_loading_funcs._get_column_sample_weights(
        label_iterable=test_labels
    )

    counts = Counter(test_labels).most_common()
    most_common = counts[0][1]
    least_common = counts[-1][1]

    assert min(label_weight_dict["samples_weighted"]) == 1.0 / most_common
    assert max(label_weight_dict["samples_weighted"]) == 1.0 / least_common

    assert sum(label_weight_dict["label_counts"]) == len(test_labels)


def test_make_random_snps_missing_some():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.float)
    test_array[:, 0, :] = 1

    patch_target = (
        "human_origins_supervised.data_load.data_loading_funcs.np.random.choice"
    )
    with patch(patch_target, autospec=True) as mock_target:
        mock_return = np.array([1, 2, 3, 4, 5])
        mock_target.return_value = mock_return

        array = data_loading_funcs.make_random_snps_missing(test_array)

        # check that all columns have one filled value
        assert (array.sum(1) != 1).sum() == 0

        expected_missing = torch.tensor([1] * 5, dtype=torch.float)
        assert (array[:, 3, mock_return] == expected_missing).all()


def test_make_random_snps_missing_all():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.float)
    test_array[:, 0, :] = 1

    array = data_loading_funcs.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=1.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 1).all()


def test_make_random_snps_missing_none():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.float)
    test_array[:, 0, :] = 1

    array = data_loading_funcs.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=0.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 0).all()
