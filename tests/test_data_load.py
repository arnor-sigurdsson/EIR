import csv
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from human_origins_supervised.data_load import datasets


@pytest.fixture
def create_label_testfile(request, tmp_path):
    test_fpath = tmp_path / "test_labels.csv"
    in_indirect_param = hasattr(request, "param")

    labels = ["Tall", "Medium", "Short"]
    if in_indirect_param and request.param == "reg":
        labels = [150, 170, 190]

    with open(test_fpath, "w") as csv_file:
        label_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL
        )
        label_writer.writerow(["ID", "Origin"])

        for label in labels:
            for index in range(100):
                label_writer.writerow([f"{index}_{label}", label])

    if in_indirect_param:
        return test_fpath, request.param
    return test_fpath


def test_get_meta_from_label_file(create_label_testfile):
    df_label = datasets.get_meta_from_label_file(create_label_testfile, "Origin")

    assert df_label.index.name == "ID"
    assert df_label.shape[0] == 300
    assert [i for i in df_label.Origin.value_counts()] == [100] * 3

    ids_to_keep = list(df_label.index[df_label.Origin.isin(("Short", "Medium"))])
    df_label_filtered = datasets.get_meta_from_label_file(
        create_label_testfile, "Origin", ids_to_keep=ids_to_keep
    )

    assert df_label_filtered.shape[0] == 200
    assert [i for i in df_label_filtered.Origin.value_counts()] == [100] * 2
    assert set(df_label_filtered.Origin.unique()) == {"Short", "Medium"}


def test_parse_label_df(create_label_testfile):
    df_label = datasets.get_meta_from_label_file(create_label_testfile, "Origin")

    def test_column_op_1(df, column_name, replace_dict):
        df = df.replace({column_name: replace_dict})
        return df

    def test_column_op_2(df, column_name, multiplier):
        df[column_name] = df[column_name] * multiplier
        return df

    replace_dict_args = {"replace_dict": {"Short": "Small"}}
    multiplier_dict_arg = {"multiplier": 2}
    test_column_ops = {
        "Origin": [
            (test_column_op_1, replace_dict_args),
            (test_column_op_2, multiplier_dict_arg),
        ]
    }

    df_label_parsed = datasets.parse_label_df(df_label, test_column_ops)

    assert df_label_parsed.shape[0] == 300
    assert [i for i in df_label_parsed.Origin.value_counts()] == [100] * 3
    assert set(df_label_parsed.Origin.unique()) == {
        "SmallSmall",
        "MediumMedium",
        "TallTall",
    }


@pytest.mark.parametrize("create_label_testfile", ("cls", "reg"), indirect=True)
def test_set_up_database_labels(args_config, create_label_testfile):
    label_file, model_task = create_label_testfile
    args_config.label_file = label_file
    args_config.model_task = model_task

    all_ids = []
    with open(label_file, "r") as infile:
        csv_reader = csv.reader(infile, delimiter=",", quotechar='"')
        next(csv_reader, None)
        for line in csv_reader:
            all_ids.append(line[0])

    train_ids = [i for i in all_ids if int(i.split("_")[0]) % 2 == 0]
    valid_ids = [i for i in all_ids if i not in train_ids]

    with patch(
        "human_origins_supervised.data_load.datasets.joblib", autospec=True
    ) as m:
        train_labels_dict, valid_labels_dict = datasets.set_up_dataset_labels(
            args_config, all_ids, train_ids, valid_ids
        )

        if model_task == "reg":
            assert m.dump.call_count == 1

    train_dict_ids = [
        i for i in train_labels_dict.keys() if int(i.split("_")[0]) % 2 == 0
    ]
    valid_dict_ids = [
        i for i in valid_labels_dict.keys() if int(i.split("_")[0]) % 2 != 0
    ]
    assert len(train_labels_dict) == len(valid_labels_dict)
    assert train_dict_ids == train_ids
    assert valid_dict_ids == valid_ids

    if model_task == "cls":
        for dict_ in train_labels_dict, valid_labels_dict:
            assert set(i["Origin"] for i in dict_.values()) == {
                "Short",
                "Medium",
                "Tall",
            }

    else:
        for dict_ in train_labels_dict, valid_labels_dict:
            values = (i["Origin"] for i in dict_.values())
            for num in values:
                assert 2 > num > -2


def test_set_up_datasets(create_test_cl_args, create_test_data):
    _, test_data_params = create_test_data
    cl_args = create_test_cl_args
    n_classes = 3 if test_data_params["class_type"] == "multi" else 2

    # try setting up some labels and arrays that should not be included
    for i in range(10):
        random_arr = np.random.rand(4, 1000)
        np.save(Path(cl_args.data_folder, f"SampleIgnoreFile_{i}.npy"), random_arr)

    with open(cl_args.label_file, "a") as csv_file:
        bad_label_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL
        )

        for i in range(10):
            bad_label_writer.writerow([f"SampleIgnoreLabel_{i}", "BadLabel"])

    # Note that currently split happens before filtering, so we won't get exact numbers
    # back as some get filtered from `all_ids`,
    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args, valid_size=0.2)

    assert len(train_dataset) + len(valid_dataset) == 100 * n_classes
    assert set(valid_dataset.ids).isdisjoint(set(train_dataset.ids))

    assert len([i for i in train_dataset.ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in valid_dataset.ids if i.startswith("SampleIgnore")]) == 0


@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
def test_datasets(
    dataset_type: str,
    create_test_data: pytest.fixture,
    create_test_cl_args: pytest.fixture,
):
    test_path, test_data_params = create_test_data
    cl_args = create_test_cl_args

    if dataset_type == "disk":
        cl_args.memory_dataset = False

    classes_tested = ["Asia", "Europe"]
    if test_data_params["class_type"] == "multi":
        classes_tested += ["Africa"]
    classes_tested.sort()

    train_no_samples = int(len(classes_tested) * 100 * 0.9)
    valid_no_sample = int(len(classes_tested) * 100 * 0.1)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    for dataset, exp_no_sample in zip(
        (train_dataset, valid_dataset), (train_no_samples, valid_no_sample)
    ):
        assert len(dataset) == exp_no_sample
        assert set(i.label[cl_args.label_column] for i in dataset.samples) == set(
            classes_tested
        )
        assert set(dataset.labels_unique) == set(classes_tested)

        le_it = dataset.label_encoder.inverse_transform
        assert (le_it(range(len(classes_tested))) == classes_tested).all()

        test_sample, test_label, test_id = dataset[0]

        le_t = dataset.label_encoder.transform
        test_label_string = dataset.samples[0].label[cl_args.label_column]
        assert test_label == le_t([test_label_string])
        assert test_id == dataset.samples[0].sample_id

    cl_args.target_width = 1200
    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)
    test_sample_pad, test_label_pad, test_id_pad = train_dataset[0]
    assert test_sample_pad.shape[-1] == 1200
