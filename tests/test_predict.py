from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.nn import CrossEntropyLoss

from conftest import cleanup
from human_origins_supervised import predict
from human_origins_supervised import train
from human_origins_supervised.models.models import CNNModel
from test_data_load.test_datasets import check_dataset


def test_load_model(args_config, tmp_path):
    """
    We need `create_test_data` here because the create_test_model fixture depends on it
    down the line, and we need to pass in params for the subrequest in the
    `create_test_data` fixture definition.
    """

    cl_args = args_config
    model: torch.nn.Module = CNNModel(cl_args, 1, None, cl_args.contn_columns).to(
        device=cl_args.device
    )

    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    loaded_model = predict._load_model(
        model_path, model.num_classes, cl_args, cl_args.device
    )
    # make sure we're in eval model
    assert not loaded_model.training

    loaded_model.train()
    for key in list(model.__dict__.keys()):
        # torch modules don't behave well with __eq__, better to use check the param
        # values as is done below
        if key not in ["_modules"]:
            assert model.__dict__[key] == loaded_model.__dict__[key], key

    for param_model, param_loaded in zip(model.parameters(), loaded_model.parameters()):
        assert param_model.data.ne(param_loaded.data).sum() == 0


def test_modify_train_cl_args_for_testing():
    cl_args_from_train = Namespace(data_folder="train/data/folder", lr=1e-3)
    cl_args_from_predict = Namespace(data_folder="test/data/folder")

    mixed_args = predict._modify_train_cl_args_for_testing(
        cl_args_from_train, cl_args_from_predict
    )
    assert mixed_args.data_folder == "test/data/folder"
    assert mixed_args.lr == 1e-3


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "regression"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [{"custom_cl_args": {"contn_columns": ["OriginExtraCol"]}}],
    indirect=True,
)
def test_load_labels_for_testing(
    create_test_data, create_test_datasets, create_test_cl_args, keep_outputs
):
    """
    Note here we are treating the generated test data (i.e. by tests, not test-set-data)
    as the testing-set.
    """
    cl_args = create_test_cl_args
    # so we test the scaling of the test set as well

    run_path = Path(f"runs/{cl_args.run_name}/")

    test_labels_dict = predict._load_labels_for_testing(cl_args, run_path)
    df_test = pd.DataFrame.from_dict(test_labels_dict, orient="index")

    # make sure test data extra column was scaled correctly
    assert df_test["OriginExtraCol"].between(-2, 2).all()

    # make sure that target column is unchanged (within expected bounds)
    assert df_test[cl_args.target_column].max() < 160
    assert df_test[cl_args.target_column].min() > 90

    if not keep_outputs:
        cleanup(run_path)


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"memory_dataset": True}},
        {"custom_cl_args": {"memory_dataset": False}},
    ],
    indirect=True,
)
def test_set_up_test_dataset(
    request,
    create_test_data: pytest.fixture,
    create_test_datasets,
    create_test_cl_args: pytest.fixture,
    keep_outputs,
):
    test_path, test_data_params = create_test_data
    cl_args = create_test_cl_args

    n_per_class = request.config.getoption("--num_samples_per_class")

    run_path = Path(f"runs/{cl_args.run_name}/")

    classes_tested = ["Asia", "Europe"]
    if test_data_params["task_type"] == "multi":
        classes_tested += ["Africa"]
    classes_tested.sort()

    test_labels_dict = predict._load_labels_for_testing(cl_args, run_path)
    test_dataset = predict._set_up_test_dataset(cl_args, test_labels_dict, run_path)

    exp_no_samples = n_per_class * len(classes_tested)
    check_dataset(test_dataset, exp_no_samples, classes_tested, cl_args)

    if not keep_outputs:
        cleanup(run_path)


def grab_latest_model_path(saved_models_folder: Path):
    saved_models = [i for i in saved_models_folder.iterdir()]
    saved_models.sort(key=lambda x: int(x.stem.split("_")[-1]))

    return saved_models[-1]


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi", "split_to_test": True}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "run_name": "test_run_predict",
                "n_epochs": 2,
                "checkpoint_interval": 50,
                # to save time since we're not testing the modelling
                "benchmark": False,
                "get_acts": False,
            }
        }
    ],
    indirect=True,
)
def test_predict_new(keep_outputs, _prep_modelling_test_configs):
    config, test_config = _prep_modelling_test_configs
    test_path = config.cl_args.data_folder

    train.train_ignite(config)

    model_path = grab_latest_model_path(test_config.run_path / "saved_models")
    predict_cl_args = Namespace(
        model_path=model_path,
        batch_size=64,
        evaluate=True,
        data_folder=test_path / "test_arrays_test_set",
        output_folder=test_path,
        device="cpu",
    )

    predict.predict(predict_cl_args)

    df_test = pd.read_csv(test_path / "predictions.csv", index_col="ID")
    classes_sorted = sorted(config.train_dataset.target_transformer.classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert (classes_sorted == df_test.columns).all()

    for cls in classes_sorted:
        class_indices = [i for i in df_test.index if i.endswith(cls)]
        df_cur_class = df_test.loc[class_indices]
        num_correct = (df_cur_class.idxmax(axis=1) == cls).sum()

        # check that most were correct
        assert num_correct / df_cur_class.shape[0] > 0.9

    if not keep_outputs:
        cleanup(test_config.run_path)


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi", "split_to_test": True}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "run_name": "test_run_predict",
                "n_epochs": 2,
                "checkpoint_interval": 50,
                # to save time since we're not testing the modelling
                "benchmark": False,
                "get_acts": False,
            }
        }
    ],
    indirect=True,
)
def test_predict(
    create_test_data,
    create_test_cl_args,
    create_test_dloaders,
    create_test_model,
    create_test_optimizer,
    create_test_datasets,
    keep_outputs,
):
    test_path, test_data_params = create_test_data

    cl_args = create_test_cl_args
    train_dloader, valid_dloader, train_dataset, valid_dataset = create_test_dloaders
    model = create_test_model
    optimizer = create_test_optimizer
    criterion = CrossEntropyLoss()

    train_dataset, valid_dataset = create_test_datasets
    target_transformer = train_dataset.target_transformer

    run_path = Path(f"runs/{cl_args.run_name}/")

    config = train.Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        train_dataset.labels_dict,
        target_transformer,
        cl_args.data_width,
    )

    train.train_ignite(config)

    model_path = grab_latest_model_path(run_path / "saved_models")
    predict_cl_args = Namespace(
        model_path=model_path,
        batch_size=64,
        evaluate=True,
        data_folder=test_path / "test_arrays_test_set",
        output_folder=test_path,
        device="cpu",
    )

    predict.predict(predict_cl_args)

    df_test = pd.read_csv(test_path / "predictions.csv", index_col="ID")
    classes_sorted = sorted(train_dataset.target_transformer.classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert (classes_sorted == df_test.columns).all()

    for cls in classes_sorted:
        class_indices = [i for i in df_test.index if i.endswith(cls)]
        df_cur_class = df_test.loc[class_indices]
        num_correct = (df_cur_class.idxmax(axis=1) == cls).sum()

        # check that most were correct
        assert num_correct / df_cur_class.shape[0] > 0.9

    if not keep_outputs:
        cleanup(run_path)
