import csv
from pathlib import Path

import numpy as np
import pytest

from human_origins_supervised.data_load import datasets


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "packbits"},
        {"class_type": "multi", "data_type": "packbits"},
    ],
    indirect=True,
)
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

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    assert len(train_dataset) + len(valid_dataset) == 200 * n_classes

    valid_ids = [i.sample_id for i in valid_dataset.samples]
    train_ids = [i.sample_id for i in train_dataset.samples]

    assert set(valid_ids).isdisjoint(set(train_ids))
    assert len([i for i in valid_ids if i.startswith("SampleIgnore")]) == 0
    assert len([i for i in train_ids if i.startswith("SampleIgnore")]) == 0


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"class_type": "binary", "data_type": "packbits"},
        {"class_type": "multi", "data_type": "packbits"},
    ],
    indirect=True,
)
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

    train_no_samples = int(len(classes_tested) * 200 * 0.9)
    valid_no_sample = int(len(classes_tested) * 200 * 0.1)

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
