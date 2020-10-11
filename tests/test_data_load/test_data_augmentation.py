from unittest.mock import patch

import numpy as np
import torch

from snp_pred.data_load import data_augmentation


def test_get_mix_data_hook():
    pass


def test_hook_mix_data():
    pass


def test_hook_mix_loss():
    pass


def test_mixup_snp_data():
    pass


def test_get_random_index_for_mixing():
    pass


def test_mixup_tensor():
    pass


def test_mixup_input():
    pass


def test_block_cutmix_input():
    pass


def test_get_block_cutmix_indices():
    pass


def test_uniform_cutmix_input():
    pass


def test_get_uniform_cutmix_indices():
    pass


def test_mixup_all_targets():
    pass


def test_mixup_targets():
    pass


def test_calc_all_mixed_losses():
    pass


def test_calc_mixed_loss():
    pass


def test_make_random_snps_missing_some():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    patch_target = "snp_pred.data_load.data_augmentation.np.random.choice"
    with patch(patch_target, autospec=True) as mock_target:
        mock_return = np.array([1, 2, 3, 4, 5])
        mock_target.return_value = mock_return

        array = data_augmentation.make_random_snps_missing(test_array)

        # check that all columns have one filled value
        assert (array.sum(1) != 1).sum() == 0

        expected_missing = torch.tensor([1] * 5, dtype=torch.bool)
        assert (array[:, 3, mock_return] == expected_missing).all()


def test_make_random_snps_missing_all():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    array = data_augmentation.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=1.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 1).all()


def test_make_random_snps_missing_none():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    array = data_augmentation.make_random_snps_missing(
        array=test_array, percentage=1.0, probability=0.0
    )

    assert (array.sum(1) != 1).sum() == 0
    assert (array[:, 3, :] == 0).all()
