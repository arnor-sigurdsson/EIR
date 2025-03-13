from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, lists
from torch.nn import functional as F

from eir.data_load import data_augmentation
from eir.train_utils.criteria import vectorized_con_loss
from tests.setup_tests.setup_modelling_test_data.setup_omics_test_data import (
    _set_up_base_test_omics_array,
)


def test_get_mix_data_hook():
    pass


def test_hook_mix_data():
    pass


def test_sample_lambda():
    no_alpha = data_augmentation._sample_lambda(mixing_alpha=0)
    assert no_alpha == 1.0

    alpha = data_augmentation._sample_lambda(mixing_alpha=1.0)
    assert alpha != 1.0


def test_hook_mix_loss():
    pass


def test_mixup_omics_data():
    pass


@given(test_batch_size=integers(min_value=8, max_value=128))
@settings(deadline=500)
def test_get_random_batch_indices_to_mix(test_batch_size):
    random_indices = data_augmentation.get_random_batch_indices_to_mix(
        batch_size=test_batch_size
    )

    indices_as_list = random_indices.tolist()

    assert len(set(indices_as_list)) == test_batch_size
    assert sorted(indices_as_list) == list(range(test_batch_size))


@given(
    input_length=integers(min_value=10, max_value=int(1e3)),
    input_height=integers(min_value=1, max_value=10),
    lambda_=floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=500)
def test_mixup_tensor(input_length: int, input_height: int, lambda_: float) -> None:
    tensor_1_one_hot_indices = torch.randint(0, input_height, (input_length,))
    tensor_2_one_hot_indices = torch.randint(0, input_height, (input_length,))

    guaranteed_indices = torch.arange(0, input_height).to(dtype=torch.long)

    # We need to make sure each one-hot index appears at least once, otherwise
    # the one-hot dimensions will not match
    tensor_1_one_hot_indices = torch.cat((tensor_1_one_hot_indices, guaranteed_indices))
    tensor_2_one_hot_indices = torch.cat((tensor_2_one_hot_indices, guaranteed_indices))

    tensor_1 = F.one_hot(tensor_1_one_hot_indices).T
    tensor_2 = F.one_hot(tensor_2_one_hot_indices).T

    assert tensor_1.shape == tensor_2.shape

    tensor_1_w_batch_dim = tensor_1.unsqueeze(0)
    tensor_2_w_batch_dim = tensor_2.unsqueeze(0)

    test_batch = torch.cat((tensor_1_w_batch_dim, tensor_2_w_batch_dim))
    batch_indices_for_mixing = torch.LongTensor([1, 0])

    mixed_tensor = data_augmentation.mixup_tensor(
        tensor=test_batch,
        lambda_=lambda_,
        random_batch_indices_to_mix=batch_indices_for_mixing,
    )

    assert (mixed_tensor.sum(dim=1) == 1.0).all()


@given(
    patched_indices=lists(
        elements=integers(min_value=0, max_value=999),
        min_size=2,
        max_size=2,
        unique=True,
    ).map(lambda x: sorted(x))
)
@settings(deadline=500)
def test_block_cutmix_omics_input(patched_indices: list[int]) -> None:
    """
    We need it to be sorted here to avoid indexing with e.g. [521:3]
    """
    test_arrays = []
    for _i in range(2):
        test_array, *_ = _set_up_base_test_omics_array(n_snps=1000)
        test_array = torch.tensor(test_array).unsqueeze(0)
        test_arrays.append(test_array)

    test_batch = torch.stack(test_arrays)

    # Needed since mixing overwrites input
    test_batch_original = test_batch.detach().clone()

    batch_indices_for_mixing = torch.LongTensor([1, 0])

    with patch(
        "eir.data_load.data_augmentation.get_block_cutmix_indices",
        return_value=patched_indices,
        autospec=True,
    ):
        block_cutmixed_test_arrays = data_augmentation.block_cutmix_omics_input(
            tensor=test_batch,
            lambda_=1.0,
            random_batch_indices_to_mix=batch_indices_for_mixing,
        )

    patched_start, patched_end = patched_indices
    base_0 = test_batch_original[0, ..., patched_start:patched_end]
    base_1 = test_batch_original[1, ..., patched_start:patched_end]
    mixed_0 = block_cutmixed_test_arrays[0, ..., patched_start:patched_end]
    mixed_1 = block_cutmixed_test_arrays[1, ..., patched_start:patched_end]

    assert (base_0 == mixed_1).all()
    assert (base_1 == mixed_0).all()

    # NOTE: Currently we only have a probabilistic guarantee for the code below to pass,
    # as e.g. if we only have 1 SNP, it can be quite likely that they match only by
    # change. This is less likely if we have >2, but at some point we should probably
    # make this more concrete.
    if patched_end - patched_start > 2:
        assert not (base_0 == mixed_0).all()
        assert not (base_1 == mixed_1).all()

        all_arrays = torch.cat((test_batch_original, block_cutmixed_test_arrays))
        for tensor_1, tensor_2 in combinations(all_arrays, r=2):
            assert not (tensor_1 == tensor_2).all()


@given(
    input_length=integers(min_value=10, max_value=int(1e4)),
    lambda_=floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=500)
def test_get_block_cutmix_indices(input_length: int, lambda_: float):
    random_index_start, random_index_end = data_augmentation.get_block_cutmix_indices(
        input_length=input_length, lambda_=lambda_
    )

    assert random_index_start <= random_index_end

    num_snps_in_mixed_block = random_index_end - random_index_start
    num_snps_from_original = input_length - num_snps_in_mixed_block
    assert num_snps_from_original + num_snps_in_mixed_block == input_length

    expected_original_no_snps = int(round(lambda_ * input_length))
    expected_diff = abs(num_snps_from_original - expected_original_no_snps)
    assert 0 <= expected_diff <= 1


@given(
    patched_indices=lists(
        elements=integers(min_value=0, max_value=999), min_size=10, max_size=1000
    )
)
@settings(deadline=500)
def test_uniform_cutmix_omics_input(patched_indices: list[int]):
    """
    Here we explicitly cut from 1 --> 0 and vice versa.

    While we expect the patches to match up (where we have base_0, mixed_0, etc), we
    expect all the arrays themselves to be different.
    """
    test_arrays = []
    for _i in range(2):
        test_array, *_ = _set_up_base_test_omics_array(n_snps=1000)
        test_array = torch.tensor(test_array).unsqueeze(0)
        test_arrays.append(test_array)

    test_batch = torch.stack(test_arrays)

    # Needed since mixing overwrites input
    test_batch_original = test_batch.detach().clone()

    batch_indices_for_mixing = torch.LongTensor([1, 0])

    # Ensure that we have at least 10 unique, otherwise e.g. if we only have 1
    # value, it's quite likely that the arrays can be the same in that once place
    patched_indices_tensor = torch.tensor(patched_indices + list(range(10))).unique()
    with patch(
        "eir.data_load.data_augmentation.get_uniform_cutmix_indices",
        return_value=patched_indices_tensor,
        autospec=True,
    ):
        uniform_cutmixed_test_arrays = data_augmentation.uniform_cutmix_omics_input(
            tensor=test_batch,
            lambda_=1.0,
            random_batch_indices_to_mix=batch_indices_for_mixing,
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
    lambda_=floats(min_value=0.0, max_value=1.0),
    input_length=integers(min_value=100, max_value=int(1e4)),
)
@settings(deadline=500)
def test_get_uniform_cutmix_indices(lambda_, input_length):
    test_random_indices = data_augmentation.get_uniform_cutmix_indices(
        input_length=input_length, lambda_=lambda_
    )
    assert len(test_random_indices.unique()) == len(test_random_indices)

    num_mixed_snps = len(test_random_indices)
    num_snps_from_original = input_length - num_mixed_snps
    assert num_snps_from_original + num_mixed_snps == input_length

    expected_original_no_snps = int(round(lambda_ * input_length))
    expected_diff = abs(num_snps_from_original - expected_original_no_snps)
    assert 0 <= expected_diff <= 1


@given(
    test_targets=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: torch.tensor(x))
)
@settings(deadline=500)
def test_mixup_all_targets(test_targets):
    target_columns = {
        "con": ["test_target_1", "test_target_2"],
        "cat": ["test_target_3"],
    }
    random_indices = torch.randperm(len(test_targets)).to(dtype=torch.long)
    all_target_columns = target_columns["con"] + target_columns["cat"]
    targets = {"test_output_tabular": dict.fromkeys(all_target_columns, test_targets)}

    def _target_columns_gen():
        for con_column in target_columns["con"]:
            yield "test_output_tabular", "con", con_column
        for cat_column in target_columns["cat"]:
            yield "test_output_tabular", "cat", cat_column

    all_mixed_targets = data_augmentation.mixup_all_targets(
        targets=targets,
        permuted_indices_for_mixing=random_indices,
        target_columns_gen=_target_columns_gen(),
    )
    for _, targets_permuted in all_mixed_targets["test_output_tabular"].items():
        assert set(test_targets.tolist()) == set(targets_permuted.tolist())

        # Probabilistic guarantee here (i.e. would fail if all were the same)
        if len(set(test_targets.tolist())) > 3:
            assert test_targets.tolist() != targets_permuted.tolist()


@given(
    test_targets=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: torch.tensor(x))
)
@settings(deadline=500)
def test_mixup_targets(test_targets):
    random_indices = torch.randperm(len(test_targets))
    targets_permuted = data_augmentation.mixup_targets(
        targets=test_targets, permuted_indices_for_mixing=random_indices
    )
    assert set(test_targets.tolist()) == set(targets_permuted.tolist())

    # Probabilistic guarantee here (i.e. would fail if all were the same)
    if len(set(test_targets.tolist())) > 3:
        assert test_targets.tolist() != targets_permuted.tolist()


def _get_mixed_loss_test_cases_for_parametrization():
    return [  # Case 1: All correct, mixed 50%
        (
            {
                "outputs": torch.ones(5),
                "targets": torch.ones(5),
                "targets_permuted": torch.ones(5),
                "lambda_": 0.5,
            },
            0.0,
        ),
        # Case 2: Only base fully correct, but lambda 1.0 (base is 100%)
        (
            {
                "outputs": torch.ones(5),
                "targets": torch.ones(5),
                "targets_permuted": torch.zeros(5),
                "lambda_": 1.0,
            },
            0.0,
        ),
        # Case 3: All wrong, lambda 0.0 (permuted is 100%)
        (
            {
                "outputs": torch.ones(5),
                "targets": torch.ones(5),
                "targets_permuted": torch.zeros(5),
                "lambda_": 0.0,
            },
            1.0,
        ),
        # Case 4: 50% mix of correct and incorrect, weighted equally, meaning we
        # have a mean of 0.5 loss, weighted down by 0.5 = 0.25
        (
            {
                "outputs": torch.ones(6),
                "targets": torch.ones(6),
                "targets_permuted": torch.tensor([0, 0, 0, 1, 1, 1]),
                "lambda_": 0.5,
            },
            0.25,
        ),
    ]


@pytest.mark.parametrize(
    "test_inputs,expected_output",
    _get_mixed_loss_test_cases_for_parametrization(),
)
def test_calc_all_mixed_losses(test_inputs, expected_output):
    target_columns = {
        "con": ["test_target_1", "test_target_2"],
        "cat": ["test_target_3"],
    }
    all_target_columns = target_columns["con"] + target_columns["cat"]

    targets = {
        "test_output_tabular": dict.fromkeys(all_target_columns, test_inputs["targets"])
    }
    targets_permuted = {
        "test_output_tabular": dict.fromkeys(
            all_target_columns, test_inputs["targets_permuted"]
        )
    }

    n_samples = len(test_inputs["outputs"])
    ids = tuple(str(i) for i in range(n_samples))

    permuted_indexes = torch.LongTensor(list(range(n_samples)))
    mixed_object = data_augmentation.MixingObject(
        ids=ids,
        targets=targets,
        targets_permuted=targets_permuted,
        lambda_=test_inputs["lambda_"],
        permuted_indexes=permuted_indexes,
    )

    test_criteria = {"test_output_tabular": vectorized_con_loss}
    outputs = {
        "test_output_tabular": dict.fromkeys(all_target_columns, test_inputs["outputs"])
    }

    all_losses = data_augmentation.calc_all_mixed_losses(
        criteria=test_criteria,
        outputs=outputs,
        mixed_object=mixed_object,
    )
    for _, loss in all_losses["test_output_tabular"].items():
        assert loss.item() == expected_output


@pytest.mark.parametrize(
    "test_inputs,expected_output",
    _get_mixed_loss_test_cases_for_parametrization(),
)
def test_calc_mixed_loss(test_inputs, expected_output):
    criterion = vectorized_con_loss

    outputs_dict = {"test_output_tabular": test_inputs["outputs"]}
    targets_dict = {"test_output_tabular": test_inputs["targets"]}
    targets_permuted_dict = {"test_output_tabular": test_inputs["targets_permuted"]}
    lambda_ = test_inputs["lambda_"]

    mixed_loss = data_augmentation.calc_mixed_loss(
        criterion=criterion,
        outputs=outputs_dict,
        targets=targets_dict,
        targets_permuted=targets_permuted_dict,
        lambda_=lambda_,
    )
    assert mixed_loss["test_output_tabular"].item() == expected_output


def test_make_random_snps_missing_some():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    patch_target = "eir.data_load.data_augmentation.torch.randperm"
    with patch(patch_target, autospec=True) as mock_target:
        mock_return = torch.tensor(np.array([1, 2, 3, 4, 5]))
        mock_target.return_value = mock_return

        array = data_augmentation.make_random_omics_columns_missing(
            omics_array=test_array,
            na_augment_alpha=99.0,
            na_augment_beta=1.0,
        )

        assert (array.sum(1) != 1).sum() == 0

        expected_missing = torch.tensor([True] * 5, dtype=torch.bool)
        assert (array[:, 3, mock_return] == expected_missing).all()


def test_make_random_snps_missing_uniform_distribution():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    array = data_augmentation.make_random_omics_columns_missing(
        omics_array=test_array,
        na_augment_alpha=1.0,
        na_augment_beta=1.0,
    )

    assert array[:, 0, :].any(), "Expected some SNPs to be set to the default state."
    assert not array[:, 1, :].any(), "Expected no SNPs in this state."
    assert array[:, 3, :].any(), "Expected some SNPs to be set to the missing state."


def test_make_random_snps_missing_maximal():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    array = data_augmentation.make_random_omics_columns_missing(
        omics_array=test_array,
        na_augment_alpha=100.0,
        na_augment_beta=1.0,
    )

    missing_percentage = (array[:, 3, :].sum() / 1000).item()

    assert missing_percentage > 0.925, (
        "Expected a high percentage of SNPs to be set to missing."
    )


def test_make_random_snps_missing_minimal():
    test_array = torch.zeros((1, 4, 1000), dtype=torch.bool)
    test_array[:, 0, :] = True

    array = data_augmentation.make_random_omics_columns_missing(
        omics_array=test_array,
        na_augment_alpha=1.0,
        na_augment_beta=100.0,
    )

    missing_percentage = (array[:, 3, :].sum() / 1000).item()
    assert missing_percentage < 0.075, "Expected minimal SNPs to be set to missing."


def test_shuffle_columns_some():
    test_array = torch.tensor(
        [
            [
                [True, False, True, False, True, False, False, False, False, False],
                [False, True, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, True, False, False, True, True],
                [False, False, False, True, False, False, True, True, False, False],
            ]
        ],
        dtype=torch.bool,
    )

    with patch("torch.rand", autospec=True) as mock_rand:
        mock_rand.return_value = torch.tensor([0.2])

        shuffled_array = data_augmentation.shuffle_random_omics_columns(
            omics_array=test_array.clone(),
            shuffle_augment_alpha=100.0,
            shuffle_augment_beta=1.0,
        )

        assert not torch.equal(shuffled_array, test_array)


def test_shuffle_columns_one_hot():
    test_array = torch.tensor(
        [
            [
                [True, False, True],
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ]
        ],
        dtype=torch.bool,
    )

    with patch("torch.rand", autospec=True) as mock_rand:
        mock_rand.return_value = torch.tensor([0.2])

        shuffled_array = data_augmentation.shuffle_random_omics_columns(
            omics_array=test_array.clone(),
            shuffle_augment_alpha=1.0,
            shuffle_augment_beta=1.0,
        )
        assert (shuffled_array.sum(dim=1) == 1).all()
