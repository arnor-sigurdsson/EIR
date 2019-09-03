import pytest

from human_origins_supervised.data_load import datasets


@pytest.mark.parametrize("dataset_type", ["memory", "disk"])
def test_memory_dataset(
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
