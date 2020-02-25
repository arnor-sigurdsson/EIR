from unittest.mock import patch

import torch
import numpy as np

from human_origins_supervised.data_load import data_loading_funcs


def test_make_random_snps_missing_some():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.uint8)
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

        expected_missing = torch.tensor([1] * 5, dtype=torch.uint8)
        assert (array[:, 3, mock_return] == expected_missing).all()


def test_make_random_snps_missing_all():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.uint8)
    test_array[:, 0, :] = 1

    array = data_loading_funcs.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=1.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 1).all()


def test_make_random_snps_missing_none():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.uint8)
    test_array[:, 0, :] = 1

    array = data_loading_funcs.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=0.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 0).all()
