import csv
from argparse import Namespace
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from human_origins_supervised.data_load import datasets
from human_origins_supervised.data_load.datasets import al_datasets


def check_dataset(
    dataset: al_datasets,
    exp_no_sample: int,
    classes_tested: List[str],
    cl_args: Namespace,
    target_column="Origin",
) -> None:

    assert len(dataset) == exp_no_sample

    transformed_values_in_dataset = set(
        i.labels[cl_args.target_column] for i in dataset.samples
    )
    expected_transformed_values = set(range(len(classes_tested)))
    assert transformed_values_in_dataset == expected_transformed_values

    tt_it = dataset.target_transformers[target_column].inverse_transform
    assert (tt_it(range(len(classes_tested))) == classes_tested).all()

    test_sample, test_label, test_id = dataset[0]

    assert test_label[target_column] in expected_transformed_values
    assert test_id == dataset.samples[0].sample_id


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
def test_set_up_datasets(create_test_cl_args, create_test_data, parse_test_cl_args):
    _, test_data_params = create_test_data

    cl_args = create_test_cl_args
    n_classes = 3 if test_data_params["class_type"] == "multi" else 2

    n_per_class = parse_test_cl_args["n_per_class"]
    num_snps = parse_test_cl_args["n_snps"]

    # try setting up some labels and arrays that should not be included
    for i in range(10):
        random_arr = np.random.rand(4, num_snps)
        np.save(Path(cl_args.data_folder, f"SampleIgnoreFile_{i}.npy"), random_arr)

    with open(cl_args.label_file, "a") as csv_file:
        bad_label_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL
        )

        for i in range(10):
            bad_label_writer.writerow([f"SampleIgnoreLabel_{i}", "BadLabel"])

    # patch since we don't create run folders while testing
    joblib_patch_target = "human_origins_supervised.data_load.datasets.joblib"
    with patch(joblib_patch_target, autospec=True) as m:
        train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

        assert m.dump.call_count == 1

    assert len(train_dataset) + len(valid_dataset) == n_per_class * n_classes

    valid_ids = [i.sample_id for i in valid_dataset.samples]
    train_ids = [i.sample_id for i in train_dataset.samples]

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}],
    indirect=True,
)
@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
def test_datasets(
    dataset_type: str,
    create_test_data: pytest.fixture,
    create_test_cl_args: pytest.fixture,
    parse_test_cl_args,
):
    _, test_data_params = create_test_data
    cl_args = create_test_cl_args

    classes_tested = get_classes_tested(test_data_fixture_params=test_data_params)
    n_per_class = parse_test_cl_args["n_per_class"]

    if dataset_type == "disk":
        cl_args.memory_dataset = False

    train_no_samples = int(len(classes_tested) * n_per_class * (1 - cl_args.valid_size))
    valid_no_sample = int(len(classes_tested) * n_per_class * cl_args.valid_size)

    # patch since we don't create run folders while testing
    joblib_patch_target = "human_origins_supervised.data_load.datasets.joblib"
    with patch(joblib_patch_target, autospec=True) as m:
        train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

        assert m.dump.call_count == 1

    check_dataset(train_dataset, train_no_samples, classes_tested, cl_args)
    check_dataset(valid_dataset, valid_no_sample, classes_tested, cl_args)

    cl_args.target_width = 1200
    with patch(joblib_patch_target, autospec=True) as m:
        train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

        assert m.dump.call_count == 1

    test_sample_pad, test_label_pad, test_id_pad = train_dataset[0]
    assert test_sample_pad.shape[-1] == 1200


def get_classes_tested(test_data_fixture_params):

    classes_tested = ["Asia", "Europe"]
    if test_data_fixture_params["class_type"] in ("multi", "regression"):
        classes_tested += ["Africa"]
    classes_tested.sort()

    return classes_tested
