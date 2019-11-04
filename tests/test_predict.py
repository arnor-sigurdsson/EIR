from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch

from conftest import cleanup
from human_origins_supervised import predict
from human_origins_supervised.models.models import Model


def test_load_model(args_config, tmp_path):
    """
    We need `create_test_data` here because the create_test_model fixture depends on it
    down the line, and we need to pass in params for the subrequest in the
    `create_test_data` fixture definition.
    """

    cl_args = args_config
    model = Model(cl_args, 1, None, cl_args.contn_columns)

    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    loaded_model = predict.load_model(model_path, model.num_classes, cl_args)
    # make sure we're in eval model
    assert not loaded_model.training

    loaded_model.train()
    for key in list(model.__dict__.keys()):
        # torch modules don't behave well with __eq__, better to use check the param
        # values as is done below
        if key not in ["_modules"]:
            assert model.__dict__[key] == loaded_model.__dict__[key], key

    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert p1.data.ne(p2.data).sum() == 0


def test_modify_train_cl_args_for_testing():
    cl_args_from_train = Namespace(data_folder="train/data/folder", lr=1e-3)
    cl_args_from_predict = Namespace(data_folder="test/data/folder")

    mixed_args = predict.modify_train_cl_args_for_testing(
        cl_args_from_train, cl_args_from_predict
    )
    assert mixed_args.data_folder == "test/data/folder"
    assert mixed_args.lr == 1e-3


@pytest.mark.parametrize(
    "create_test_data",
    [{"class_type": "regression", "data_type": "uint8"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [{"custom_cl_args": {"contn_columns": ["OriginExtraCol"]}}],
    indirect=True,
)
def test_load_labels_for_testing(
    create_test_data, create_test_dataset, create_test_cl_args, keep_outputs
):
    """
    Note here we are treating the generated test data (i.e. by tests, not test-set-data)
    as the testing-set.
    """
    cl_args = create_test_cl_args
    # so we test the scaling of the test set as well

    run_path = Path(f"runs/{cl_args.run_name}/")

    test_labels_dict = predict.load_labels_for_testing(cl_args)
    df_test = pd.DataFrame.from_dict(test_labels_dict, orient="index")

    # make sure test data extra column was scaled correctly
    assert df_test["OriginExtraCol"].between(-2, 2).all()

    # make sure that target column is unchanged (within expected bounds)
    assert df_test[cl_args.target_column].max() < 160
    assert df_test[cl_args.target_column].min() > 90

    if not keep_outputs:
        cleanup(run_path)


@pytest.mark.parametrize(
    "create_test_data", [{"class_type": "multi", "data_type": "uint8"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"memory_dataset": True}},
        {"custom_cl_args": {"memory_dataset": False}},
    ],
    indirect=True,
)
def test_set_up_test_dataset(
    create_test_data: pytest.fixture,
    create_test_dataset: pytest.fixture,
    create_test_cl_args: pytest.fixture,
    keep_outputs,
):
    test_path, test_data_params = create_test_data
    cl_args = create_test_cl_args

    run_path = Path(f"runs/{cl_args.run_name}/")

    classes_tested = ["Asia", "Europe"]
    if test_data_params["class_type"] == "multi":
        classes_tested += ["Africa"]
    classes_tested.sort()

    test_labels_dict = predict.load_labels_for_testing(cl_args)
    test_dataset = predict.set_up_test_dataset(cl_args, test_labels_dict)

    # TODO: Merge this with functionality in test_datasets into a common function
    assert len(test_dataset) == 2000 * len(classes_tested)
    assert set(i.labels[cl_args.target_column] for i in test_dataset.samples) == set(
        classes_tested
    )
    assert set(test_dataset.labels_unique) == set(classes_tested)

    tt_it = test_dataset.target_transformer.inverse_transform
    assert (tt_it(range(len(classes_tested))) == classes_tested).all()

    test_sample, test_label, test_id = test_dataset[0]

    tt_t = test_dataset.target_transformer.transform
    test_label_string = test_dataset.samples[0].labels[cl_args.target_column]
    assert test_label == tt_t([test_label_string])
    assert test_id == test_dataset.samples[0].sample_id

    if not keep_outputs:
        cleanup(run_path)


def test_predict():
    pass
