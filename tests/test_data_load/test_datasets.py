import csv
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from human_origins_supervised.data_load import datasets
from human_origins_supervised.data_load.datasets import al_datasets


def get_joblib_patch_target():
    return "human_origins_supervised.data_load.datasets.joblib"


@patch(get_joblib_patch_target(), autospec=True)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
def test_set_up_datasets(
    patched_joblib: MagicMock, create_test_cl_args, create_test_data, parse_test_cl_args
):
    c = create_test_data
    n_classes = len(c.target_classes)

    cl_args = create_test_cl_args

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)
    assert patched_joblib.dump.call_count == 1

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


def test_construct_dataset_init_params_from_cl_args(args_config):
    constructed_args = datasets.construct_dataset_init_params_from_cl_args(args_config)

    assert len(constructed_args) == 3
    assert constructed_args["data_folder"] == args_config.data_folder
    assert constructed_args["target_width"] == args_config.target_width

    expected_target_cols = {"con": [], "cat": ["Origin"]}
    assert constructed_args["target_columns"] == expected_target_cols


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


@patch(get_joblib_patch_target(), autospec=True)
def test_save_target_transformer(patched_joblib):

    test_transformer = StandardScaler()
    test_transformer.fit([[1, 2, 3, 4, 5]])

    datasets.save_target_transformer(
        run_folder=Path("/tmp/"),
        transformer_name="harry_du_bois",
        target_transformer_object=test_transformer,
    )

    assert patched_joblib.dump.call_count == 1

    _, m_kwargs = patched_joblib.dump.call_args
    # check that we have correct name, with target_transformers tagged on
    assert m_kwargs["filename"].name == "harry_du_bois_target_transformer.save"


@pytest.fixture()
def get_transformer_test_data():
    test_labels_dict = {
        "1": {"Origin": "Asia", "Height": 150},
        "2": {"Origin": "Africa", "Height": 190},
        "3": {"Origin": "Europe", "Height": 170},
    }

    test_target_columns_dict = {"con": ["Height"], "cat": ["Origin"]}

    return test_labels_dict, test_target_columns_dict


def test_set_up_all_target_transformers(get_transformer_test_data):
    test_labels_dict, test_target_columns_dict = get_transformer_test_data

    all_target_transformers = datasets.set_up_all_target_transformers(
        labels_dict=test_labels_dict, target_columns=test_target_columns_dict
    )

    height_transformer = all_target_transformers["Height"]
    assert isinstance(height_transformer, StandardScaler)

    origin_transformer = all_target_transformers["Origin"]
    assert isinstance(origin_transformer, LabelEncoder)


def test_fit_scaler_transformer_on_target_column(get_transformer_test_data):
    test_labels_dict, test_target_columns_dict = get_transformer_test_data

    height_transformer = datasets._fit_transformer_on_target_column(
        labels_dict=test_labels_dict, target_column="Height", column_type="con"
    )

    assert height_transformer.n_samples_seen_ == 3
    assert height_transformer.mean_ == 170
    assert height_transformer.transform([[170]]) == 0


def test_fit_labelencoder_transformer_on_target_column(get_transformer_test_data):
    test_labels_dict, test_target_columns_dict = get_transformer_test_data

    origin_transformer = datasets._fit_transformer_on_target_column(
        labels_dict=test_labels_dict, target_column="Origin", column_type="cat"
    )

    assert origin_transformer.transform(["Africa"]).item() == 0
    assert origin_transformer.transform(["Europe"]).item() == 2


def test_streamline_values_for_transformer():
    test_values = np.array([1, 2, 3, 4, 5])

    scaler_transformer = StandardScaler()
    streamlined_values_scaler = datasets._streamline_values_for_transformers(
        transformer=scaler_transformer, values=test_values
    )
    assert streamlined_values_scaler.shape == (5, 1)

    encoder_transformer = LabelEncoder()
    streamlined_values_encoder = datasets._streamline_values_for_transformers(
        transformer=encoder_transformer, values=test_values
    )
    assert streamlined_values_encoder.shape == (5,)


@pytest.mark.parametrize(
    "test_input_key,expected",
    [
        ("1", {"Origin_as_int": 1, "Scaled_height_int": -1}),  # asia
        ("2", {"Origin_as_int": 0, "Scaled_height_int": 1}),  # africa
        ("3", {"Origin_as_int": 2, "Scaled_height_int": 0}),  # europe
    ],
)
def test_transform_all_labels_in_sample(
    test_input_key, expected, get_transformer_test_data
):
    test_labels_dict, test_target_columns_dict = get_transformer_test_data

    target_transformers = datasets.set_up_all_target_transformers(
        labels_dict=test_labels_dict, target_columns=test_target_columns_dict
    )

    test_input_dict = test_labels_dict[test_input_key]
    transformed_sample_labels = datasets._transform_all_labels_in_sample(
        target_transformers=target_transformers, sample_label_dict=test_input_dict
    )

    assert transformed_sample_labels["Origin"] == expected["Origin_as_int"]
    assert int(transformed_sample_labels["Height"]) == expected["Scaled_height_int"]


@pytest.mark.parametrize(
    "test_input,expected",
    [({"Transformer": StandardScaler()}, 0.0), ({"Transformer": LabelEncoder()}, 2)],
)
def test_transform_single_label_value(test_input, expected):
    test_transformer = test_input["Transformer"]
    test_data = np.array([0, 1, 2, 3, 4])

    test_data_streamlined = datasets._streamline_values_for_transformers(
        test_transformer, test_data
    )

    test_transformer.fit(test_data_streamlined)

    transformed_value = datasets._transform_single_label_value(
        transformer=test_transformer, label_value=2
    )

    assert transformed_value == expected


def test_set_up_num_classes(get_transformer_test_data):
    test_labels_dict, test_target_columns_dict = get_transformer_test_data

    target_transformers = datasets.set_up_all_target_transformers(
        labels_dict=test_labels_dict, target_columns=test_target_columns_dict
    )

    num_classes = datasets._set_up_num_classes(target_transformers=target_transformers)

    assert num_classes["Height"] == 1
    assert num_classes["Origin"] == 3


@patch(get_joblib_patch_target(), autospec=True)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
def test_datasets(
    patched_joblib: MagicMock,
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

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    assert patched_joblib.dump.call_count == 1

    check_dataset(
        dataset=train_dataset,
        exp_no_sample=train_no_samples,
        classes_tested=classes_tested,
    )
    check_dataset(
        dataset=valid_dataset,
        exp_no_sample=valid_no_sample,
        classes_tested=classes_tested,
    )


@patch(get_joblib_patch_target(), autospec=True)
@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "binary"}, {"task_type": "multi"}], indirect=True
)
@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
def test_dataset_padding(
    patched_joblib: MagicMock,
    dataset_type: str,
    create_test_data: pytest.fixture,
    create_test_cl_args: pytest.fixture,
    parse_test_cl_args,
):
    cl_args = create_test_cl_args

    if dataset_type == "disk":
        cl_args.memory_dataset = False

    cl_args.target_width = 1200

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)
    assert patched_joblib.dump.call_count == 1

    test_sample_pad, test_label_pad, test_id_pad = train_dataset[0]
    assert test_sample_pad.shape[-1] == 1200


def check_dataset(
    dataset: al_datasets,
    exp_no_sample: int,
    classes_tested: List[str],
    target_column="Origin",
) -> None:

    assert len(dataset) == exp_no_sample

    transformed_values_in_dataset = set(
        i.labels[target_column] for i in dataset.samples
    )
    expected_transformed_values = set(range(len(classes_tested)))
    assert transformed_values_in_dataset == expected_transformed_values

    tt_it = dataset.target_transformers[target_column].inverse_transform

    assert (tt_it(range(len(classes_tested))) == classes_tested).all()

    test_sample, test_label, test_id = dataset[0]

    assert test_label[target_column] in expected_transformed_values
    assert test_id == dataset.samples[0].sample_id
