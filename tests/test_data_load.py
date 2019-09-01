from typing import Union, Type

import pytest

from human_origins_supervised.models import data_load


@pytest.mark.parametrize(
    "dataset_class", [data_load.DiskArrayDataset, data_load.MemoryArrayDataset]
)
def test_memory_dataset(
    dataset_class: Union[
        Type[data_load.MemoryArrayDataset], Type[data_load.DiskArrayDataset]
    ],
    create_test_data: pytest.fixture,
    create_test_cl_args: pytest.fixture,
):
    test_path, test_data_params = create_test_data
    cl_args = create_test_cl_args

    classes_tested = ["Asia", "Europe"]
    if test_data_params["class_type"] == "multi":
        classes_tested += ["Africa"]
    classes_tested.sort()

    target_no_samples = len(classes_tested) * 100

    full_dataset = dataset_class(
        data_folder=test_path / "test_arrays",
        model_task=cl_args.model_task,
        target_width=cl_args.target_width,
        label_fpath=cl_args.label_file,
        label_column=cl_args.label_column,
        data_type=test_data_params["data_type"],
    )

    assert len(full_dataset) == target_no_samples
    assert len(full_dataset.arrays) == target_no_samples
    assert len(full_dataset.ids) == target_no_samples

    assert len(full_dataset.labels) == target_no_samples
    assert set(full_dataset.labels) == set(classes_tested)
    assert set(full_dataset.labels_unique) == set(classes_tested)

    le_it = full_dataset.label_encoder.inverse_transform
    assert (le_it(range(len(classes_tested))) == classes_tested).all()

    test_sample, test_label, test_id = full_dataset[0]
    assert test_label == full_dataset.labels_numerical[0]
    assert test_id == full_dataset.ids[0]

    if isinstance(full_dataset, data_load.MemoryArrayDataset):
        assert (test_sample == full_dataset.arrays[0]).all()

    full_dataset_with_padding = dataset_class(
        data_folder=test_path / "test_arrays",
        model_task=cl_args.model_task,
        target_width=cl_args.target_width + 200,
        label_fpath=cl_args.label_file,
        label_column=cl_args.label_column,
        data_type=test_data_params["data_type"],
    )

    test_sample_pad, test_label_pad, test_id_pad = full_dataset_with_padding[0]
    assert test_sample_pad.shape[-1] == 1200
