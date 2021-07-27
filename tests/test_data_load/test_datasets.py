import csv
from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np
import pytest
import torch

from eir import train
from eir.setup.config import get_all_targets, Configs
from eir.data_load import datasets
from eir.data_load.datasets import al_datasets

if TYPE_CHECKING:
    from ..conftest import TestDataConfig


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "mlp"},
                    }
                ],
            },
        }
    ],
    indirect=True,
)
def test_set_up_datasets(
    create_test_config,
    create_test_data,
    parse_test_cl_args,
):

    test_data_config = create_test_data
    n_classes = len(test_data_config.target_classes)

    test_configs = create_test_config

    all_array_ids = train.gather_all_array_target_ids(
        target_configs=test_configs.target_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    target_labels_info = train.get_tabular_target_file_infos(
        target_configs=test_configs.target_configs
    )

    target_labels = train.set_up_target_labels_wrapper(
        tabular_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    inputs = train.set_up_inputs(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=test_configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
    )

    assert (
        len(train_dataset) + len(valid_dataset)
        == test_data_config.n_per_class * n_classes
    )

    valid_ids = [i.sample_id for i in valid_dataset.samples]
    train_ids = [i.sample_id for i in train_dataset.samples]

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


def _set_up_bad_arrays_for_testing(n_snps: int, output_folder: Path):
    # try setting up some labels and arrays that should not be included
    for i in range(10):
        random_arr = np.random.rand(4, n_snps)
        np.save(Path(output_folder, f"SampleIgnoreFILE_{i}.npy"), random_arr)


def _set_up_bad_label_file_for_testing(label_file: Path):
    with open(str(label_file), "a") as csv_file:
        bad_label_writer = csv.writer(csv_file, delimiter=",")

        for i in range(10):
            bad_label_writer.writerow([f"SampleIgnoreLABEL_{i}", "BadLabel"])


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
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
                        "input_type_info": {"model_type": "mlp"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_cat_columns": [],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        }
    ],
    indirect=True,
)
def test_construct_dataset_init_params_from_cl_args(
    create_test_config, create_test_data, create_test_labels
):

    test_configs = create_test_config

    all_array_ids = train.gather_all_array_target_ids(
        target_configs=test_configs.target_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    target_labels_info = train.get_tabular_target_file_infos(
        target_configs=test_configs.target_configs
    )

    target_labels = train.set_up_target_labels_wrapper(
        tabular_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    inputs = train.set_up_inputs(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    targets = get_all_targets(targets_configs=test_configs.target_configs)
    constructed_args = datasets.construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels.train_labels,
        targets=targets,
        inputs=inputs,
    )

    assert len(constructed_args) == 3
    assert len(constructed_args.get("inputs")) == 2

    gotten_input_names = set(constructed_args.get("inputs").keys())
    expected_input_names = {"omics_test_genotype", "tabular_test_tabular"}
    assert gotten_input_names == expected_input_names

    assert (
        inputs["omics_test_genotype"]
        == constructed_args["inputs"]["omics_test_genotype"]
    )

    expected_target_cols = {"con": ["Height"], "cat": ["Origin"]}
    assert constructed_args["target_columns"] == expected_target_cols


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {
                            "model_type": "mlp",
                            "na_augment_perc": 0.05,
                        },
                    }
                ],
            },
        }
    ],
    indirect=True,
)
def test_datasets(
    dataset_type: str,
    create_test_data: "TestDataConfig",
    create_test_config: Configs,
    parse_test_cl_args,
):
    """
    We set `na_augment_perc` here to 0.0 as a safety guard against it having be set
    in the defined args setup. This is because the `check_dataset` currently assumes
    no SNPs have been dropped out.
    """
    c = create_test_data
    test_configs = create_test_config
    gc = test_configs.global_config
    classes_tested = sorted(list(c.target_classes.keys()))

    if dataset_type == "disk":
        gc.memory_dataset = False

    train_no_samples = int(len(classes_tested) * c.n_per_class * (1 - gc.valid_size))
    valid_no_sample = int(len(classes_tested) * c.n_per_class * gc.valid_size)

    all_array_ids = train.gather_all_array_target_ids(
        target_configs=test_configs.target_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    target_labels_info = train.get_tabular_target_file_infos(
        target_configs=test_configs.target_configs
    )

    target_labels = train.set_up_target_labels_wrapper(
        tabular_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    inputs = train.set_up_inputs(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=test_configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
    )

    check_dataset(
        dataset=train_dataset,
        exp_no_sample=train_no_samples,
        classes_tested=classes_tested,
        target_transformers=target_labels.label_transformers,
    )
    check_dataset(
        dataset=valid_dataset,
        exp_no_sample=valid_no_sample,
        classes_tested=classes_tested,
        target_transformers=target_labels.label_transformers,
    )


def check_dataset(
    dataset: al_datasets,
    exp_no_sample: int,
    classes_tested: List[str],
    target_transformers,
    target_column="Origin",
) -> None:

    assert len(dataset) == exp_no_sample

    transformed_values_in_dataset = set(
        i.target_labels[target_column] for i in dataset.samples
    )
    expected_transformed_values = set(range(len(classes_tested)))
    assert transformed_values_in_dataset == expected_transformed_values

    tt_it = target_transformers[target_column].inverse_transform

    assert (tt_it(range(len(classes_tested))) == classes_tested).all()

    test_inputs, target_labels, test_id = dataset[0]
    test_genotype = test_inputs["omics_test_genotype"]

    assert (test_genotype.sum(1) == 1).all()
    assert target_labels[target_column] in expected_transformed_values
    assert test_id == dataset.samples[0].sample_id


def test_prepare_genotype_array():
    test_array = torch.zeros((1, 4, 100), dtype=torch.bool)

    prepared_array = datasets.prepare_one_hot_omics_data(
        genotype_array=test_array,
        na_augment_perc=1.0,
        na_augment_prob=1.0,
    )

    assert (prepared_array[:, -1, :] == 1).all()
