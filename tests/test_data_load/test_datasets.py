import csv
from copy import deepcopy
from pathlib import Path
from typing import List, TYPE_CHECKING, Dict, Union, Any, Sequence
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from eir import train
from eir.data_load import datasets
from eir.data_load.data_preparation_modules import common
from eir.data_load.data_preparation_modules import imputation
from eir.data_load.data_preparation_modules import prepare_bytes
from eir.data_load.data_preparation_modules import prepare_image
from eir.data_load.data_preparation_modules import prepare_omics
from eir.data_load.data_preparation_modules import prepare_sequence
from eir.data_load.data_source_modules import local_ops
from eir.data_load.datasets import al_datasets
from eir.data_load.label_setup import al_label_transformers
from eir.setup.config import Configs
from eir.setup.input_setup_modules.setup_image import (
    ImageNormalizationStats,
    get_image_transforms,
)
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
                        "output_info": {"output_name": "test_output"},
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
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.global_config.output_folder)

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

    valid_ids = [i.sample_id for i in valid_dataset.samples]
    train_ids = [i.sample_id for i in train_dataset.samples]

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


def _add_bad_arrays_and_targets_for_dataset_testing(
    test_configs: Configs, test_cl_args: Dict
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
            row = [f"SampleIgnoreLABEL_{i}", "BadLabel", 0.0, "BadLabel", 0.0]
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
                        "output_info": {"output_name": "test_output"},
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
    dataset_fail_config: Dict,
    create_test_config: Configs,
    create_test_data,
):
    test_configs = create_test_config

    configs_copy = deepcopy(test_configs)

    if dataset_fail_config.get("corrupt_arrays", None):
        _corrupt_arrays_for_testing(
            test_configs=configs_copy,
            n=dataset_fail_config.get("corrupt_arrays_n", None),
        )

    if dataset_fail_config.get("corrupt_labels", None):
        _corrupt_label_file_for_testing(
            test_configs=configs_copy,
            n=dataset_fail_config.get("corrupt_labels_n", None),
        )

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.global_config.output_folder)

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
        if dataset_fail_config.get("corrupt_ids", None):
            train_dataset.samples[0].sample_id = ""
            valid_dataset.samples[0].sample_id = ""
            with pytest.raises(ValueError):
                train_dataset.check_samples()
            with pytest.raises(ValueError):
                valid_dataset.check_samples()

        if dataset_fail_config.get("corrupt_inputs", None):
            train_dataset.samples[0].inputs = {}
            valid_dataset.samples[0].inputs = {}
            with pytest.raises(ValueError):
                train_dataset.check_samples()
            with pytest.raises(ValueError):
                valid_dataset.check_samples()


def _corrupt_label_file_for_testing(
    test_configs: Configs, n: Union[int, None] = None
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


def _corrupt_arrays_for_testing(
    test_configs: Configs, n: Union[int, None] = None
) -> None:
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
                    "output_folder": "extra_inputs",
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
                        "output_info": {"output_name": "test_output"},
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
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.global_config.output_folder)

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
        target_labels_dict=target_labels.train_labels,
        outputs=outputs_as_dict,
        inputs=inputs,
        test_mode=False,
    )

    assert len(constructed_args) == 5
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
                            "na_augment_perc": 0.05,
                        },
                        "model_config": {
                            "model_type": "linear",
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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
    elif dataset_type == "memory":
        gc.memory_dataset = True

    train_no_samples = int(len(classes_tested) * c.n_per_class * (1 - gc.valid_size))
    valid_no_sample = int(len(classes_tested) * c.n_per_class * gc.valid_size)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_configs.global_config.valid_size
    )

    run_folder = get_run_folder(output_folder=test_configs.global_config.output_folder)

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
    classes_tested: List[str],
    target_transformers: Dict[str, al_label_transformers],
    check_targets: bool = True,
    output_name: str = "test_output",
    target_column="Origin",
) -> None:
    assert len(dataset) == exp_no_sample

    test_inputs, target_labels, test_id = dataset[0]

    if check_targets:
        transformed_values_in_dataset = set(
            i.target_labels[output_name][target_column] for i in dataset.samples
        )
        expected_transformed_values = set(range(len(classes_tested)))
        assert transformed_values_in_dataset == expected_transformed_values

        tt_it = target_transformers[output_name][target_column].inverse_transform

        assert (tt_it(range(len(classes_tested))) == classes_tested).all()

        assert target_labels[output_name][target_column] in expected_transformed_values

    test_genotype = test_inputs["test_genotype"]
    assert (test_genotype.sum(1) == 1).all()

    assert test_id == dataset.samples[0].sample_id


def test_prepare_genotype_array_train_mode():
    test_array = torch.zeros((4, 100), dtype=torch.uint8).detach().numpy()
    test_array_copy = deepcopy(test_array)

    prepared_array_train = prepare_omics.prepare_one_hot_omics_data(
        genotype_array=test_array,
        na_augment_perc=1.0,
        na_augment_prob=1.0,
        test_mode=False,
    )

    assert prepared_array_train != test_array
    assert (test_array_copy == test_array).all()

    assert (prepared_array_train[:, -1, :] == 1).all()


def test_prepare_genotype_array_test_mode():
    test_array = torch.zeros((1, 4, 100), dtype=torch.uint8).detach().numpy()
    test_array_copy = deepcopy(test_array)

    prepared_array_test = prepare_omics.prepare_one_hot_omics_data(
        genotype_array=test_array,
        na_augment_perc=1.0,
        na_augment_prob=1.0,
        test_mode=True,
    )
    assert prepared_array_test != test_array
    assert (test_array_copy == test_array).all()

    assert prepared_array_test.sum().item() == 0


def test_prepare_sequence_data():
    test_input = np.array([i for i in range(100)])
    test_input_copy = deepcopy(test_input)

    input_config_mock = MagicMock()
    input_config_mock.computed_max_length = 64
    input_config_mock.encode_func.return_value = [0]
    prepared_tensor = prepare_sequence.prepare_sequence_data(
        sequence_input_object=input_config_mock,
        cur_file_content_tokenized=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor != test_input

    assert len(prepared_tensor) == 64
    assert len(test_input) == 100


def test_prepare_bytes_data():
    test_input = np.array([i for i in range(100)])
    test_input_copy = deepcopy(test_input)

    input_config_mock = MagicMock()
    input_config_mock.input_config.input_type_info.max_length = 64
    input_config_mock.vocab.get.return_value = 0
    prepared_tensor = prepare_bytes.prepare_bytes_data(
        bytes_input_object=input_config_mock,
        bytes_data=test_input,
        test_mode=False,
    )

    assert (test_input == test_input_copy).all()
    assert prepared_tensor != test_input

    assert len(prepared_tensor) == 64
    assert len(test_input) == 100


def test_prepare_image_data():
    normalization_stats = ImageNormalizationStats(channel_means=[0], channel_stds=[0.1])
    base_transforms, all_transforms = get_image_transforms(
        target_size=(32, 32),
        normalization_stats=normalization_stats,
        auto_augment=False,
    )

    input_config_mock = MagicMock()
    input_config_mock.base_transforms = base_transforms
    input_config_mock.all_transforms = all_transforms

    arr = np.random.rand(64, 64)
    image_data = Image.fromarray(np.uint8(arr * 255))
    arr_pil = np.array(image_data)

    prepared_tensor = prepare_image.prepare_image_data(
        image_input_object=input_config_mock, image_data=image_data, test_mode=False
    )

    assert (arr != image_data).any()
    assert arr_pil.shape == (64, 64)
    assert arr.shape == (64, 64)

    assert prepared_tensor.shape == (1, 32, 32)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
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
                        "output_info": {"output_name": "test_output"},
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
def test_get_file_sample_id_iterator(
    create_test_config: Configs,
    create_test_data: "TestDataConfig",
    parse_test_cl_args: Dict[str, Any],
):
    test_experiment_config = create_test_config
    test_data_config = create_test_data

    input_configs = test_experiment_config.input_configs
    assert len(input_configs) == 1

    input_data_source = input_configs[0].input_info.input_source

    iterator = local_ops.get_file_sample_id_iterator(
        data_source=input_data_source, ids_to_keep=None
    )
    all_ids = [i for i in iterator]

    assert len(all_ids) == test_data_config.n_per_class * len(
        test_data_config.target_classes
    )

    iterator_empty = local_ops.get_file_sample_id_iterator(
        data_source=input_data_source, ids_to_keep=["does_not_exists"]
    )
    all_ids = [i for i in iterator_empty]
    assert len(all_ids) == 0


def test_process_tensor_to_length():
    test_tensor = torch.arange(0, 100)

    test_tensor_simple_trunc = common.process_tensor_to_length(
        tensor=test_tensor, max_length=50, sampling_strategy_if_longer="from_start"
    )
    assert len(test_tensor_simple_trunc) == 50
    assert (test_tensor_simple_trunc == test_tensor[:50]).all()

    test_tensor_unif_trunc = common.process_tensor_to_length(
        tensor=test_tensor, max_length=50, sampling_strategy_if_longer="uniform"
    )
    assert len(test_tensor_unif_trunc) == 50
    assert len(test_tensor_unif_trunc) == len(set(test_tensor_unif_trunc))

    test_tensor_padded = common.process_tensor_to_length(
        tensor=test_tensor, max_length=128, sampling_strategy_if_longer="uniform"
    )
    assert len(test_tensor_padded) == 128


def test_sample_sequence_uniform():
    test_tensor = torch.arange(0, 100)

    sampled_tensor = common._sample_sequence_uniform(
        tensor=test_tensor, tensor_length=len(test_tensor), max_length=50
    )
    assert len(sampled_tensor) == 50
    assert len(sampled_tensor) == len(set(sampled_tensor))


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary", "modalities": ["omics", "sequence"]},
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
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-04},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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
def test_impute_missing_modalities(
    create_test_config: Configs,
    create_test_data: "TestDataConfig",
    parse_test_cl_args: Dict[str, Any],
):
    test_experiment_config = create_test_config
    test_data_config = create_test_data

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_experiment_config.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_experiment_config.global_config.valid_size
    )

    input_objects = train.set_up_inputs_for_training(
        inputs_configs=test_experiment_config.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )
    impute_dtypes = imputation._get_default_impute_dtypes(inputs_objects=input_objects)
    impute_fill_values = imputation._get_default_impute_fill_values(
        inputs_objects=input_objects
    )

    test_inputs_all_avail = {k: torch.empty(10) for k in input_objects.keys()}

    no_fill = imputation.impute_missing_modalities(
        inputs_values=test_inputs_all_avail,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )
    assert no_fill == test_inputs_all_avail

    test_inputs_missing_tabular = {
        k: v for k, v in test_inputs_all_avail.items() if "tabular" not in k
    }
    filled_tabular = imputation.impute_missing_modalities(
        inputs_values=test_inputs_missing_tabular,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    transformers = input_objects["test_tabular"].labels.label_transformers
    origin_extra_label_encoder = transformers["OriginExtraCol"]
    na_transformer_index = origin_extra_label_encoder.transform(["NA"]).item()
    filled_na_value = filled_tabular["test_tabular"]["OriginExtraCol"]
    assert na_transformer_index == filled_na_value
    assert filled_tabular["test_tabular"]["ExtraTarget"] == 0.0

    test_inputs_missing_omics = {
        k: v for k, v in test_inputs_all_avail.items() if k != "test_genotype"
    }
    with_imputed_omics = imputation.impute_missing_modalities(
        inputs_values=test_inputs_missing_omics,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )
    assert len(with_imputed_omics) == 3
    assert with_imputed_omics["test_genotype"].numel() == test_data_config.n_snps * 4


def test_impute_single_missing_modality():
    imputed_test_tensor = imputation.impute_single_missing_modality(
        shape=(10, 10), fill_value=0, dtype=torch.float
    )
    assert imputed_test_tensor.numel() == 100
    assert len(imputed_test_tensor.shape) == 2
    assert imputed_test_tensor.shape[0] == 10
    assert imputed_test_tensor.shape[1] == 10

    assert (imputed_test_tensor == 0.0).all()


@pytest.mark.parametrize(
    "subset_indices",
    [
        None,
        range(10),
        range(0, 50, 2),
        range(50, 100),
        range(0, 100, 2),
    ],
)
def test_load_array_from_disk(subset_indices: Union[None, Sequence[int]]):
    test_arr = np.zeros((4, 100))
    test_arr[-1, :50] = 1
    test_arr[0, 50:] = 1

    with patch(
        "eir.data_load.data_preparation_modules.prepare_omics.np.load",
        return_value=test_arr,
        autospec=True,
    ):
        loaded = prepare_omics.omics_load_wrapper(
            input_source="fake",
            data_pointer=Path("fake"),
            subset_indices=subset_indices,
        )

    expected = test_arr
    if subset_indices is not None:
        expected = test_arr[:, subset_indices]

    assert (loaded == expected).all()
