from collections import Counter
from math import isclose
from statistics import mean
from typing import List

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, lists

from eir.data_load import data_loading_funcs
from eir.data_load.datasets import al_datasets
from eir.setup.config import get_all_tabular_targets
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
                    "basic_experiment": {
                        "output_folder": "extra_inputs",
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin", "OriginExtraCol"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_weighted_random_sampler(
    create_test_config, create_test_data, create_test_datasets
):
    test_config = create_test_config
    targets_object = get_all_tabular_targets(output_configs=test_config.output_configs)
    train_dataset, valid_dataset = create_test_datasets

    patched_train_dataset = patch_dataset_to_be_unbalanced(dataset=train_dataset)
    columns_to_sample = [
        "test_output_tabular__" + i
        for i in targets_object.cat_targets["test_output_tabular"]
    ]
    random_sampler = data_loading_funcs.get_weighted_random_sampler(
        target_df=patched_train_dataset.target_labels_df,
        columns_to_sample=columns_to_sample,
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
        dataloader=train_dataloader,
        target_col="Origin",
        output_name="test_output_tabular",
    )
    are_close = _check_if_all_numbers_close(
        list_of_numbers=list(label_counts.values()), abs_tol=200
    )
    assert are_close, label_counts.values()

    # Assert failure when we don't use weighted random sampler
    train_dataloader, valid_dataloader = get_dataloaders(
        train_dataset=patched_train_dataset,
        train_sampler=None,
        valid_dataset=valid_dataset,
        batch_size=64,
        num_workers=0,
    )

    label_counts_imbalanced = _gather_dataloader_target_label_distributions(
        dataloader=train_dataloader,
        target_col="Origin",
        output_name="test_output_tabular",
    )
    are_close_imb = _check_if_all_numbers_close(
        list_of_numbers=list(label_counts_imbalanced.values()), abs_tol=100
    )
    assert not are_close_imb, label_counts_imbalanced.values()


def _check_if_all_numbers_close(list_of_numbers, abs_tol):
    are_close = np.array(
        [
            isclose(i, v, abs_tol=abs_tol)
            for i in list_of_numbers
            for v in list_of_numbers
        ]
    )
    return are_close.all()


def patch_dataset_to_be_unbalanced(dataset: al_datasets) -> al_datasets:
    """
    Makes dataset unbalanced by limiting samples with Origin=1 to max_values=100
    """
    max_values = 100

    df_targets = dataset.target_labels_df
    df_inputs = dataset.input_df
    origin_1_df = df_targets.filter(pl.col("test_output_tabular__Origin") == 1).head(
        max_values
    )
    other_df = df_targets.filter(pl.col("test_output_tabular__Origin") != 1)

    dataset.target_labels_df = pl.concat([origin_1_df, other_df])
    dataset.input_df = df_inputs.filter(
        pl.col("ID").is_in(dataset.target_labels_df["ID"])
    )

    return dataset


def _gather_dataloader_target_label_distributions(
    dataloader, target_col: str, output_name: str, num_epochs: int = 5
):
    total_counts = {}  # above step function

    for epoch in range(num_epochs):
        for batch in dataloader:
            _, labels, _ = batch

            counts = Counter(labels[output_name][target_col].numpy())
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
    test_input_df, test_target_df = generate_test_data(
        test_labels=test_labels,
        target_columns=test_target_columns,
        output_name="test_output_tabular",
    )
    all_target_weights_test_dict = data_loading_funcs._gather_column_sampling_weights(
        target_df=test_target_df,
        columns_to_sample=test_target_columns,
        output_name="test_output_tabular",
    )

    for cur_label_weight_dict in all_target_weights_test_dict.values():
        _check_label_weights_and_counts(
            test_labels=test_labels, label_weight_dict=cur_label_weight_dict
        )


def generate_test_data(
    test_labels: List[int], target_columns: List[str], output_name: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    input_df = pl.DataFrame(
        {
            "ID": [str(i) for i in range(len(test_labels))],
            "omics_test": [f"fake_path_{i}.npy" for i in range(len(test_labels))],
        }
    )

    target_data = {
        "ID": [str(i) for i in range(len(test_labels))],
        **{col: test_labels for col in target_columns},
    }
    target_df = pl.DataFrame(target_data)

    return input_df, target_df


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
    test_input_df, test_target_df = generate_test_data(
        test_labels=test_labels,
        target_columns=target_columns,
        output_name="test_output_tabular",
    )

    gather_func = data_loading_funcs._gather_column_sampling_weights
    test_all_label_weights_and_counts = gather_func(
        target_df=test_target_df,
        columns_to_sample=target_columns,
        output_name="test_output_tabular",
    )
    agg_func = data_loading_funcs._aggregate_column_sampling_weights
    test_weights, test_samples_per_epoch = agg_func(
        all_target_columns_weights_and_counts=test_all_label_weights_and_counts
    )
    origin_weights = test_all_label_weights_and_counts["Origin"]["samples_weighted"]
    expected_weights = origin_weights * len(target_columns)

    origin_counts = test_all_label_weights_and_counts["Origin"]["label_counts"]
    hair_counts = test_all_label_weights_and_counts["HairColor"]["label_counts"]
    expected_samples_per_epoch = int(mean([mean(origin_counts), mean(hair_counts)]))

    assert np.allclose(test_weights, expected_weights, rtol=1e-5, atol=1e-8)
    assert test_samples_per_epoch == expected_samples_per_epoch
