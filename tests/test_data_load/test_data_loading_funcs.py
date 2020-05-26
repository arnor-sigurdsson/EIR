from math import isclose
from collections import Counter
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis.strategies import lists, integers

from human_origins_supervised.data_load import data_loading_funcs
from human_origins_supervised.data_load.datasets import Sample
from human_origins_supervised.train import get_dataloaders


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "model_type": "cnn",
                "target_cat_columns": ["Origin", "OriginExtraCol"],
                "extra_con_columns": ["ExtraTarget"],
                "target_con_columns": ["Height"],
                "run_name": "extra_inputs",
            }
        }
    ],
    indirect=True,
)
def test_get_weighted_random_sampler(
    create_test_cl_args, create_test_data, create_test_datasets
):
    cl_args = create_test_cl_args
    target_columns = cl_args.target_cat_columns
    train_dataset, valid_dataset = create_test_datasets

    patched_train_dataset = patch_dataset_to_be_unbalanced(dataset=train_dataset)
    random_sampler = data_loading_funcs.get_weighted_random_sampler(
        train_dataset=patched_train_dataset, target_columns=target_columns
    )

    assert random_sampler.replacement

    train_dataloader, valid_dataloader = get_dataloaders(
        train_dataset=patched_train_dataset,
        train_sampler=random_sampler,
        valid_dataset=valid_dataset,
        batch_size=64,
    )

    label_counts = _gather_dataloader_label_distributions(
        dataloader=train_dataloader, target_col="Origin"
    )
    are_close = _check_if_all_numbers_close(
        list_of_numbers=list(label_counts.values()), abs_tol=100
    )
    assert are_close

    # Assert failure when we don't use weighted random sampler
    train_dataloader, valid_dataloader = get_dataloaders(
        train_dataset=patched_train_dataset,
        train_sampler=None,
        valid_dataset=valid_dataset,
        batch_size=64,
    )

    label_counts_imbalanced = _gather_dataloader_label_distributions(
        dataloader=train_dataloader, target_col="Origin"
    )
    are_close_imb = _check_if_all_numbers_close(
        list_of_numbers=list(label_counts_imbalanced.values()), abs_tol=100
    )
    assert not are_close_imb


def _check_if_all_numbers_close(list_of_numbers, abs_tol):
    are_close = np.array(
        [
            isclose(i, v, abs_tol=abs_tol)
            for i in list_of_numbers
            for v in list_of_numbers
        ]
    )
    return are_close.all()


def patch_dataset_to_be_unbalanced(dataset):
    max_values = 100
    new_samples = []
    cur_values = 0
    for sample in dataset.samples:

        if sample.labels["target_labels"]["Origin"] == 1:
            if cur_values < max_values:
                new_samples.append(sample)
                cur_values += 1

        else:
            new_samples.append(sample)

    dataset.samples = new_samples
    return dataset


def _gather_dataloader_label_distributions(
    dataloader, target_col: str, num_epochs: int = 5
):
    total_counts = {}  # above step function

    for epoch in range(num_epochs):

        for batch in dataloader:
            _, labels, _ = batch

            counts = Counter(labels["target_labels"][target_col].numpy())
            for key, item in counts.items():
                if key not in total_counts:
                    total_counts[key] = item
                else:
                    total_counts[key] += item

    return total_counts


@given(
    test_labels=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: x + list(range(10)))
)
def test_gather_column_sampling_weights(test_labels):
    test_target_columns = ["Origin", "HairColor"]
    test_samples = generate_test_samples(
        test_labels=test_labels, target_columns=test_target_columns
    )
    all_target_weights_test_dict = data_loading_funcs._gather_column_sampling_weights(
        samples=test_samples, target_columns=test_target_columns
    )

    for cur_label_weight_dict in all_target_weights_test_dict.values():
        _check_label_weights_and_counts(
            test_labels=test_labels, label_weight_dict=cur_label_weight_dict
        )


def generate_test_samples(test_labels: List[int], target_columns: List[str]):
    test_samples = []
    for idx, label in enumerate(test_labels):
        cur_label_dict = {
            "target_labels": {column_name: label for column_name in target_columns},
            "extra_labels": {},
        }
        cur_test_sample = Sample(
            sample_id=str(idx), array=f"fake_path_{idx}.npy", labels=cur_label_dict
        )
        test_samples.append(cur_test_sample)

    return test_samples


def _check_label_weights_and_counts(test_labels, label_weight_dict):
    counts = Counter(test_labels).most_common()
    most_common = counts[0][1]
    least_common = counts[-1][1]

    assert min(label_weight_dict["samples_weighted"]) == 1.0 / most_common
    assert max(label_weight_dict["samples_weighted"]) == 1.0 / least_common

    assert sum(label_weight_dict["label_counts"]) == len(test_labels)


@given(
    test_labels=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: x + list(range(10)))
)
def test_get_column_sample_weights(test_labels):
    """
    We do the `map` to make sure that the list contains at least one copy of all
    unique elements, which is what is expected from the LabelEncoder class used to
    generate the labels.
    """
    label_weight_dict = data_loading_funcs._get_column_label_weights_and_counts(
        label_iterable=test_labels
    )

    _check_label_weights_and_counts(
        test_labels=test_labels, label_weight_dict=label_weight_dict
    )


@given(
    test_labels=lists(
        elements=integers(min_value=0, max_value=9), min_size=10, max_size=1000
    ).map(lambda x: x + list(range(10)))
)
def test_aggregate_column_sampling_weights_auto(test_labels):
    """
    Note: This test currently works with the assumption that the labels are equally
    distributed for all target columns, further work means making this more realistic
    by having different labels generated for different label columns.
    """
    target_columns = ["Origin", "HairColor"]
    test_samples = generate_test_samples(
        test_labels=test_labels, target_columns=target_columns
    )

    gather_func = data_loading_funcs._gather_column_sampling_weights
    test_all_label_weights_and_counts = gather_func(
        samples=test_samples, target_columns=target_columns
    )
    agg_func = data_loading_funcs._aggregate_column_sampling_weights
    test_weights, test_samples_per_epoch = agg_func(
        all_target_columns_weights_and_counts=test_all_label_weights_and_counts
    )
    origin_weights = test_all_label_weights_and_counts["Origin"]["samples_weighted"]
    expected_weights = origin_weights * len(target_columns)

    origin_counts = test_all_label_weights_and_counts["Origin"]["label_counts"]
    expected_samples_per_epoch = min(origin_counts) * len(target_columns)

    assert (test_weights == expected_weights).all()
    assert test_samples_per_epoch == expected_samples_per_epoch


def test_aggregate_column_sampling_weights_manual():
    pass


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
