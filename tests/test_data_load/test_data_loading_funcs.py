from collections import Counter
from math import isclose
from typing import List

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, integers

from eir.data_load import data_loading_funcs
from eir.data_load.datasets import Sample
from eir.setup.config import get_all_targets
from eir.train import get_dataloaders


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "run_name": "extra_inputs",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin", "OriginExtraCol"],
                    "target_con_columns": ["Height"],
                },
            },
        },
    ],
    indirect=True,
)
def test_get_weighted_random_sampler(
    create_test_config, create_test_data, create_test_datasets
):
    test_config = create_test_config
    targets_object = get_all_targets(targets_configs=test_config.target_configs)
    train_dataset, valid_dataset = create_test_datasets

    patched_train_dataset = patch_dataset_to_be_unbalanced(dataset=train_dataset)
    random_sampler = data_loading_funcs.get_weighted_random_sampler(
        samples=patched_train_dataset.samples, target_columns=targets_object.cat_targets
    )

    assert random_sampler.replacement

    train_dataloader, valid_dataloader = get_dataloaders(
        train_dataset=patched_train_dataset,
        train_sampler=random_sampler,
        valid_dataset=valid_dataset,
        batch_size=64,
        num_workers=0,
    )

    label_counts = _gather_dataloader_target_label_distributions(
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
        num_workers=0,
    )

    label_counts_imbalanced = _gather_dataloader_target_label_distributions(
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

        if sample.target_labels["Origin"] == 1:
            if cur_values < max_values:
                new_samples.append(sample)
                cur_values += 1

        else:
            new_samples.append(sample)

    dataset.samples = new_samples
    return dataset


def _gather_dataloader_target_label_distributions(
    dataloader, target_col: str, num_epochs: int = 5
):
    total_counts = {}  # above step function

    for epoch in range(num_epochs):

        for batch in dataloader:
            _, labels, _ = batch

            counts = Counter(labels[target_col].numpy())
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
@settings(deadline=500)
def test_gather_column_sampling_weights(test_labels):
    """
    We have the .map here to ensure that all possible values exist at least once.
    """
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
        cur_label_dict = {column_name: label for column_name in target_columns}
        cur_inputs = {"omics_test": f"fake_path_{idx}.npy"}
        cur_test_sample = Sample(
            sample_id=str(idx), inputs=cur_inputs, target_labels=cur_label_dict
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
@settings(deadline=500)
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
@settings(deadline=500)
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
