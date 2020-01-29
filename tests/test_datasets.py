import csv
from argparse import Namespace
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

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
    c = create_test_data
    n_classes = len(c.target_classes)

    cl_args = create_test_cl_args

    # try setting up some labels and arrays that should not be included
    for i in range(10):
        random_arr = np.random.rand(4, c.n_snps)
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

    assert len(train_dataset) + len(valid_dataset) == c.n_per_class * n_classes

    valid_ids = [i.sample_id for i in valid_dataset.samples]
    train_ids = [i.sample_id for i in train_dataset.samples]

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


def test_construct_dataset_init_params_from_cl_args(args_config):
    pass


@pytest.mark.parametrize(
    "test_input,expected",
    [  # test case 1
        (
            (["con_1", "con_2"], ["cat_1", "cat_2"]),
            {"con": ["con_1", "con_2"], "cat": ["cat_1", "cat_2"]},
        ),
        # test case 2
        ((["con_1", "con_2"], []), {"con": ["con_1", "con_2"], "cat": []}),
    ],
)
def test_merge_target_columns_pass(test_input, expected):
    test_output = datasets.merge_target_columns(*test_input)
    assert test_output == expected


def test_merge_target_columns_fail():
    with pytest.raises(ValueError):
        datasets.merge_target_columns([], [])


def test_save_target_transformer():

    test_transformer = StandardScaler()
    test_transformer.fit([[1, 2, 3, 4, 5]])

    # patch since we don't create run folders while testing
    joblib_patch_target = "human_origins_supervised.data_load.datasets.joblib"
    with patch(joblib_patch_target, autospec=True) as m:
        datasets.save_target_transformer(
            run_folder=Path("/tmp/"),
            transformer_name="harry_du_bois",
            target_transformer_object=test_transformer,
        )

        assert m.dump.call_count == 1

        _, m_kwargs = m.dump.call_args
        # check that we have correct name, with target_transformer tagged on
        assert m_kwargs["filename"].name == ("harry_du_bois_target_transformer.save")


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "binary"}, {"class_type": "multi"}, {"class_type": "regression"}],
    indirect=True,
)
def test_set_up_all_target_transformers_prev(create_test_datasets, create_test_data):
    train_dataset, _ = create_test_datasets


def test_set_up_all_target_transformers():
    test_labels_dict = {
        "1": {"Origin": "Asia", "Height": 150},
        "2": {"Origin": "Africa", "Height": 190},
        "3": {"Origin": "Europe", "Height": 170},
    }

    test_target_columns_dict = {"con": ["Height"], "cat": ["Origin"]}

    all_target_transformers = datasets.set_up_all_target_transformers(
        labels_dict=test_labels_dict, target_columns=test_target_columns_dict
    )

    # TODO: Finish.
    assert all_target_transformers


def test_fit_transformer_on_target_column():
    pass


def test_transform_sample_labels():
    pass


def test_transform_label_value():
    pass


def set_up_num_classes():
    pass


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
    c = create_test_data
    cl_args = create_test_cl_args
    classes_tested = sorted(list(c.target_classes.keys()))

    if dataset_type == "disk":
        cl_args.memory_dataset = False

    train_no_samples = int(
        len(classes_tested) * c.n_per_class * (1 - cl_args.valid_size)
    )
    valid_no_sample = int(len(classes_tested) * c.n_per_class * cl_args.valid_size)

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
