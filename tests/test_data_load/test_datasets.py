import csv
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from eir import train
from eir.data_load import datasets
from eir.data_load.datasets import al_datasets
from eir.data_load.label_setup import al_label_transformers
from eir.setup.config import Configs
from eir.setup.output_setup import set_up_outputs_for_training
from eir.target_setup.target_label_setup import (
    gather_all_ids_from_output_configs,
    set_up_all_targets_wrapper,
)
from eir.train_utils.utils import get_run_folder

if TYPE_CHECKING:
    from ..setup_tests.fixtures_create_data import TestDataConfig


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
            "manual_test_data_creator": lambda: "test_dataset_binary",
        },
        {
            "task_type": "multi",
            "manual_test_data_creator": lambda: "test_dataset_multi",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
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

    _add_bad_arrays_and_targets_for_dataset_testing(
        test_configs=test_configs, test_cl_args=parse_test_cl_args
    )

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    # Check that corrupted arrays / labels were added successfully
    assert len(all_array_ids) > test_data_config.n_per_class * n_classes

    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids,
        valid_size=test_configs.gc.be.valid_size,
    )

    run_folder = get_run_folder(output_folder=test_configs.gc.be.output_folder)

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_configs=test_configs.output_configs,
        run_folder=run_folder,
        hooks=None,
    )

    inputs = train.set_up_inputs_for_training(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=test_configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
        outputs_as_dict=outputs_as_dict,
    )

    assert (
        len(train_dataset) + len(valid_dataset)
        == test_data_config.n_per_class * n_classes
    )

    valid_ids = tuple(valid_dataset.input_storage.get_ids())
    train_ids = tuple(train_dataset.input_storage.get_ids())

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


def _add_bad_arrays_and_targets_for_dataset_testing(
    test_configs: Configs, test_cl_args: dict
) -> None:
    test_inputs = test_configs.input_configs
    assert len(test_inputs) == 1
    test_omics_folder = Path(test_inputs[0].input_info.input_source)

    _set_up_bad_arrays_for_testing(
        n_snps=test_cl_args["n_snps"], output_folder=test_omics_folder
    )

    output_configs = test_configs.output_configs
    assert len(output_configs) == 1
    test_target_label_file = Path(output_configs[0].output_info.output_source)
    _set_up_bad_label_file_for_testing(label_file=test_target_label_file)


def _set_up_bad_arrays_for_testing(n_snps: int, output_folder: Path) -> None:
    # try setting up some labels and arrays that should not be included
    for i in range(10):
        random_arr = np.random.rand(4, n_snps)
        outpath = Path(output_folder, f"SampleIgnoreFILE_{i}.npy")
        np.save(str(outpath), random_arr)


def _set_up_bad_label_file_for_testing(label_file: Path) -> None:
    with open(str(label_file), "a") as csv_file:
        bad_label_writer = csv.writer(csv_file, delimiter=",")

        for i in range(10):
            row = [
                f"SampleIgnoreLABEL_{i}",  # ID
                "BadLabel",  # Origin
                0.0,  # Height
                "BadLabel",  # OriginExtraCol
                0.0,  # ExtraTarget
                0.0,  # SparseHeight
                "BadLabel",  # SparseOrigin
                0,  # BinaryOrigin
                1000.0,  # Time
            ]
            bad_label_writer.writerow(row)


@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "dataset_fail_config",
    [
        {
            "corrupt_arrays": True,
        },
        {
            "corrupt_labels": True,
        },
        {
            "corrupt_arrays": True,
            "corrupt_labels": True,
        },
        {
            "corrupt_ids": True,
        },
        {
            "corrupt_inputs": True,
        },
    ],
)
@pytest.mark.parametrize(
    "create_test_data",
    [{"task_type": "binary", "manual_test_data_creator": uuid4}],
    indirect=True,
)
def test_set_up_datasets_fails(
    dataset_fail_config: dict,
    create_test_config: Configs,
    create_test_data,
):
    test_configs = create_test_config

    configs_copy = deepcopy(test_configs)

    if dataset_fail_config.get("corrupt_arrays"):
        _corrupt_arrays_for_testing(
            test_configs=configs_copy,
            n=dataset_fail_config.get("corrupt_arrays_n"),
        )

    if dataset_fail_config.get("corrupt_labels"):
        _corrupt_label_file_for_testing(
            test_configs=configs_copy,
            n=dataset_fail_config.get("corrupt_labels_n"),
        )

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.gc.be.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.gc.be.output_folder)

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_configs=test_configs.output_configs,
        run_folder=run_folder,
        hooks=None,
    )

    inputs = train.set_up_inputs_for_training(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs,
        target_transformers=target_labels.label_transformers,
    )

    if set(dataset_fail_config.keys()).issubset({"corrupt_labels", "corrupt_arrays"}):
        with pytest.raises(ValueError):
            datasets.set_up_datasets_from_configs(
                configs=configs_copy,
                target_labels=target_labels,
                inputs_as_dict=inputs,
                outputs_as_dict=outputs_as_dict,
            )
    if set(dataset_fail_config.keys()).issubset({"corrupt_ids", "corrupt_inputs"}):
        train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
            configs=configs_copy,
            target_labels=target_labels,
            inputs_as_dict=inputs,
            outputs_as_dict=outputs_as_dict,
        )
        if dataset_fail_config.get("corrupt_ids"):
            assert train_dataset.input_storage.string_columns[0].name == "ID"
            assert valid_dataset.input_storage.string_columns[0].name == "ID"
            train_dataset.input_storage.string_data[0, 0] = np.nan
            valid_dataset.input_storage.string_data[0, 0] = np.nan
            with pytest.raises(ValueError):
                train_dataset.check_samples()
            with pytest.raises(ValueError):
                valid_dataset.check_samples()

        if dataset_fail_config.get("corrupt_inputs"):
            assert train_dataset.input_storage.path_columns[0].name == "test_genotype"
            assert valid_dataset.input_storage.path_columns[0].name == "test_genotype"
            train_dataset.input_storage.path_data[0, 0] = np.nan
            valid_dataset.input_storage.path_data[0, 0] = np.nan
            with pytest.raises(ValueError):
                train_dataset.check_samples()
            with pytest.raises(ValueError):
                valid_dataset.check_samples()


def _corrupt_label_file_for_testing(
    test_configs: Configs, n: int | None = None
) -> None:
    output_configs = test_configs.output_configs
    assert len(output_configs) == 1
    test_target_label_file = Path(output_configs[0].output_info.output_source)

    df_labels = pd.read_csv(test_target_label_file)
    if n:
        sample_row_idxs = df_labels.sample(n).index
        df_labels.loc[sample_row_idxs, "ID"] = df_labels.loc[
            sample_row_idxs, "ID"
        ].apply(lambda x: uuid4())
    else:
        df_labels["ID"] = df_labels["ID"].apply(lambda x: uuid4())

    df_labels.to_csv(test_target_label_file, index=0)


def _corrupt_arrays_for_testing(test_configs: Configs, n: int | None = None) -> None:
    test_inputs = test_configs.input_configs
    assert len(test_inputs) == 1
    test_omics_folder = Path(test_inputs[0].input_info.input_source)

    for idx, f in enumerate(test_omics_folder.iterdir()):
        if n and idx >= n:
            break

        f.rename(Path(f.parent, str(uuid4()) + ".npy"))


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "extra_inputs",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": [],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_construct_dataset_init_params_from_cl_args(
    create_test_config, create_test_data, create_test_labels
):
    test_configs = create_test_config

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.gc.be.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.gc.be.output_folder)

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_configs=test_configs.output_configs,
        run_folder=run_folder,
        hooks=None,
    )

    inputs = train.set_up_inputs_for_training(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs,
        target_transformers=target_labels.label_transformers,
    )

    constructed_args = datasets.construct_default_dataset_kwargs_from_cl_args(
        target_labels_df=target_labels.train_labels,
        outputs=outputs_as_dict,
        inputs=inputs,
        test_mode=False,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    assert len(constructed_args) == 6
    assert len(constructed_args.get("inputs")) == 2

    gotten_input_names = set(constructed_args.get("inputs").keys())
    expected_input_names = {"test_genotype", "test_tabular"}
    assert gotten_input_names == expected_input_names

    assert inputs["test_genotype"] == constructed_args["inputs"]["test_genotype"]

    assert constructed_args["outputs"] == outputs_as_dict


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
                            "na_augment_alpha": 0.0,
                            "na_augment_beta": 0.0,
                        },
                        "model_config": {
                            "model_type": "linear",
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
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
    parse_test_cl_args: dict,
):
    """
    We set `na_augment_alpha` here to 0.0 as a safety guard against it having be set
    in the defined args setup. This is because the `check_dataset` currently assumes
    no SNPs have been dropped out.
    """
    c = create_test_data
    test_configs = create_test_config
    gc = test_configs.global_config
    classes_tested = sorted(c.target_classes.keys())

    if dataset_type == "disk":
        gc.be.memory_dataset = False
    elif dataset_type == "memory":
        gc.be.memory_dataset = True

    train_no_samples = int(len(classes_tested) * c.n_per_class * (1 - gc.be.valid_size))
    valid_no_sample = int(len(classes_tested) * c.n_per_class * gc.be.valid_size)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.gc.be.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.gc.be.output_folder)

    target_labels = set_up_all_targets_wrapper(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_configs=test_configs.output_configs,
        run_folder=run_folder,
        hooks=None,
    )

    inputs = train.set_up_inputs_for_training(
        inputs_configs=test_configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=test_configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
        outputs_as_dict=outputs_as_dict,
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
    classes_tested: list[str],
    target_transformers: dict[str, al_label_transformers],
    check_targets: bool = True,
    output_name: str = "test_output_tabular",
    target_column="Origin",
) -> None:
    assert len(dataset) == exp_no_sample

    test_inputs, target_labels, test_id = dataset[0]

    if check_targets:
        transformed_values_in_dataset = set(
            dataset.target_labels_df[f"{output_name}__{target_column}"]
        )
        expected_transformed_values = set(range(len(classes_tested)))
        assert transformed_values_in_dataset == expected_transformed_values

        tt_it = target_transformers[output_name][target_column].inverse_transform

        assert (tt_it(range(len(classes_tested))) == classes_tested).all()

        assert target_labels[output_name][target_column] in expected_transformed_values

    test_genotype = test_inputs["test_genotype"]
    assert (test_genotype.sum(1) == 1).all()

    assert test_id == dataset.target_labels_storage.get_row(0)["ID"]
    assert test_id == dataset.input_storage.get_row(0)["ID"]
