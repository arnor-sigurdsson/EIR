from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import lists, integers, floats
from torch import nn

from snp_pred.data_load import data_augmentation
from tests.conftest import _set_up_base_test_array


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


@given(
    patched_indices=lists(
        elements=integers(min_value=0, max_value=999), min_size=10, max_size=1000
    )
)
def test_uniform_cutmix_input(patched_indices):
    """
    Here we explicitly cut from 1 --> 0 and vice versa.

    While we expect the patches to match up (where we have base_0, mixed_0, etc), we
    expect all the arrays themselves to be different.
    """
    test_arrays = []
    for i in range(2):
        test_array, *_ = _set_up_base_test_array(n_snps=1000)
        test_array = torch.tensor(test_array).unsqueeze(0)
        test_arrays.append(test_array)

    test_batch = torch.stack(test_arrays)

    # Needed since mixing overwrites input
    test_batch_original = test_batch.clone()

    indices_for_mixing = torch.LongTensor([1, 0])

    # Ensure that we have at least 10 unique, otherwise e.g. if we only have 1
    # value, it's quite likely that the arrays can be the same in that once place
    patched_indices_tensor = torch.tensor(patched_indices + list(range(10))).unique()
    with patch(
        "snp_pred.data_load.data_augmentation.get_uniform_cutmix_indices",
        return_value=patched_indices_tensor,
        autospec=True,
    ):
        uniform_cutmixed_test_arrays = data_augmentation.uniform_cutmix_input(
            input_batch=test_batch,
            lambda_=1.0,
            random_batch_indices_to_mix=indices_for_mixing,
        )

    base_0 = test_batch_original[0, ..., patched_indices_tensor]
    base_1 = test_batch_original[1, ..., patched_indices_tensor]
    mixed_0 = uniform_cutmixed_test_arrays[0, ..., patched_indices_tensor]
    mixed_1 = uniform_cutmixed_test_arrays[1, ..., patched_indices_tensor]

    assert (base_0 == mixed_1).all()
    assert (base_1 == mixed_0).all()
    assert not (base_0 == mixed_0).all()
    assert not (base_1 == mixed_1).all()

    all_arrays = torch.cat((test_batch_original, uniform_cutmixed_test_arrays))
    for tensor_1, tensor_2 in combinations(all_arrays, r=2):
        assert not (tensor_1 == tensor_2).all()


@given(
    test_lambda=floats(min_value=0.0, max_value=1.0),
    test_num_snps=integers(min_value=100, max_value=int(1e4)),
)
def test_get_uniform_cutmix_indices(test_lambda, test_num_snps):
    test_random_indices = data_augmentation.get_uniform_cutmix_indices(
        input_length=test_num_snps, lambda_=test_lambda
    )
    assert len(test_random_indices.unique()) == len(test_random_indices)


@given(
    test_targets=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: torch.tensor(x))
)
def test_mixup_all_targets(test_targets):
    target_columns = {
        "con": ["test_target_1", "test_target_2"],
        "cat": ["test_target_3"],
    }
    random_indices = torch.randperm(len(test_targets))
    all_target_columns = target_columns["con"] + target_columns["cat"]
    targets = {c: test_targets for c in all_target_columns}

    all_mixed_targets = data_augmentation.mixup_all_targets(
        targets=targets,
        random_index_for_mixing=random_indices,
        target_columns=target_columns,
    )
    for _, targets_permuted in all_mixed_targets.items():
        assert set(test_targets.tolist()) == set(targets_permuted.tolist())


@given(
    test_targets=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: torch.tensor(x))
)
def test_mixup_targets(test_targets):
    random_indices = torch.randperm(len(test_targets))
    targets_permuted = data_augmentation.mixup_targets(
        targets=test_targets, random_index_for_mixing=random_indices
    )
    assert set(test_targets.tolist()) == set(targets_permuted.tolist())


def _get_mixed_loss_test_cases_for_parametrization():
    return [  # Case 1: All correct, mixed 50%
        (
            dict(
                outputs=torch.ones(5),
                targets=torch.ones(5),
                targets_permuted=torch.ones(5),
                lambda_=0.5,
            ),
            0.0,
        ),
        # Case 2: Only base fully correct, but lambda 1.0 (base is 100%)
        (
            dict(
                outputs=torch.ones(5),
                targets=torch.ones(5),
                targets_permuted=torch.zeros(5),
                lambda_=1.0,
            ),
            0.0,
        ),
        # Case 3: All wrong, lambda 0.0 (permuted is 100%)
        (
            dict(
                outputs=torch.ones(5),
                targets=torch.ones(5),
                targets_permuted=torch.zeros(5),
                lambda_=0.0,
            ),
            1.0,
        ),
        # Case 4: 50% mix of correct and incorrect, weighted equally, meaning we
        # have a mean of 0.5 loss, weighted down by 0.5 = 0.25
        (
            dict(
                outputs=torch.ones(6),
                targets=torch.ones(6),
                targets_permuted=torch.tensor([0, 0, 0, 1, 1, 1]),
                lambda_=0.5,
            ),
            0.25,
        ),
    ]


@pytest.mark.parametrize(
    "test_inputs,expected_output", _get_mixed_loss_test_cases_for_parametrization(),
)
def test_calc_all_mixed_losses(test_inputs, expected_output):
    target_columns = {
        "con": ["test_target_1", "test_target_2"],
        "cat": ["test_target_3"],
    }
    all_target_columns = target_columns["con"] + target_columns["cat"]

    targets = {c: test_inputs["targets"] for c in all_target_columns}
    targets_permuted = {c: test_inputs["targets_permuted"] for c in all_target_columns}
    mixed_object = data_augmentation.MixupOutput(
        inputs=torch.zeros(0),
        targets=targets,
        targets_permuted=targets_permuted,
        lambda_=test_inputs["lambda_"],
        permuted_indexes=[0],
    )

    test_criterions = {c: nn.MSELoss() for c in all_target_columns}
    outputs = {c: test_inputs["outputs"] for c in all_target_columns}
    all_losses = data_augmentation.calc_all_mixed_losses(
        target_columns=target_columns,
        criterions=test_criterions,
        outputs=outputs,
        mixed_object=mixed_object,
    )
    for _, loss in all_losses.items():
        assert loss.item() == expected_output


@pytest.mark.parametrize(
    "test_inputs,expected_output", _get_mixed_loss_test_cases_for_parametrization(),
)
def test_calc_mixed_loss(test_inputs, expected_output):
    criterion = nn.MSELoss()

    mixed_loss = data_augmentation.calc_mixed_loss(criterion=criterion, **test_inputs)
    assert mixed_loss.item() == expected_output


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
