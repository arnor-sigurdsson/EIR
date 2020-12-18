import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from snp_pred import train
from snp_pred.data_load import datasets
from snp_pred.data_load.datasets import al_datasets


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_set_up_datasets(
    create_test_cl_args,
    create_test_data,
    parse_test_cl_args,
    create_test_data_dimensions,
):
    c = create_test_data
    n_classes = len(c.target_classes)
    data_dimensions = create_test_data_dimensions

    cl_args = create_test_cl_args

    target_labels, tabular_input_labels = train.get_target_and_tabular_input_labels(
        cl_args=cl_args, custom_label_parsing_operations=None
    )
    train_dataset, valid_dataset = datasets.set_up_datasets_from_cl_args(
        cl_args=cl_args,
        data_dimensions=data_dimensions,
        target_labels=target_labels,
        tabular_inputs_labels=tabular_input_labels,
    )

    assert len(train_dataset) + len(valid_dataset) == c.n_per_class * n_classes

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
def test_construct_dataset_init_params_from_cl_args(
    args_config, create_test_data, create_test_labels, create_test_data_dimensions
):

    target_labels, tabular_input_labels = create_test_labels
    data_dimensions = create_test_data_dimensions

    args_config.target_con_columns = ["Height"]
    args_config.extra_con_columns = ["BMI"]

    constructed_args = datasets.construct_default_dataset_kwargs_from_cl_args(
        cl_args=args_config,
        target_labels_dict=target_labels.train_labels,
        data_dimensions=data_dimensions,
        tabular_labels_dict=tabular_input_labels.train_labels,
        na_augment=True,
    )

    assert len(constructed_args) == 5

    assert "omics_cl_args" in constructed_args["data_sources"]
    assert constructed_args["data_sources"]["omics_cl_args"] == args_config.data_source
    assert "tabular_cl_args" not in constructed_args["data_sources"]

    expected_target_cols = {"con": ["Height"], "cat": ["Origin"]}
    assert constructed_args["target_columns"] == expected_target_cols


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
@pytest.mark.parametrize(
    "create_test_cl_args",
    [{"custom_cl_args": {"na_augment_perc": 0.05}}],
    indirect=True,
)
def test_datasets(
    dataset_type: str,
    create_test_data: pytest.fixture,
    create_test_cl_args: pytest.fixture,
    parse_test_cl_args,
    create_test_data_dimensions,
):
    """
    We set `na_augment_perc` here to 0.0 as a safety guard against it having be set
    in the defined args config. This is because the `check_dataset` currently assumes
    no SNPs have been dropped out.
    """
    c = create_test_data
    cl_args = create_test_cl_args
    classes_tested = sorted(list(c.target_classes.keys()))
    data_dimensions = create_test_data_dimensions

    if dataset_type == "disk":
        cl_args.memory_dataset = False

    train_no_samples = int(
        len(classes_tested) * c.n_per_class * (1 - cl_args.valid_size)
    )
    valid_no_sample = int(len(classes_tested) * c.n_per_class * cl_args.valid_size)

    target_labels, tabular_input_labels = train.get_target_and_tabular_input_labels(
        cl_args=cl_args, custom_label_parsing_operations=None
    )
    train_dataset, valid_dataset = datasets.set_up_datasets_from_cl_args(
        cl_args=cl_args,
        data_dimensions=data_dimensions,
        target_labels=target_labels,
        tabular_inputs_labels=tabular_input_labels,
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
    test_genotype = test_inputs["omics_cl_args"]

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
