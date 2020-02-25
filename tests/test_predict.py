from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

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
    num_classes = {"Origin": 3}
    model: torch.nn.Module = CNNModel(
        cl_args=cl_args,
        num_classes=num_classes,
        embeddings_dict=None,
        extra_continuous_inputs_columns=cl_args.contn_columns,
    ).to(device=cl_args.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = predict._load_model(
        model_path=model_path,
        num_classes=model.num_classes,
        train_cl_args=cl_args,
        device=cl_args.device,
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


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "contn_columns": ["ExtraTarget"],
                "target_con_columns": ["Height"],
                "target_cat_columns": ["Origin"],
            }
        }
    ],
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

    test_labels_dict = predict._load_labels_for_testing(test_train_cl_args_mix=cl_args)
    df_test = pd.DataFrame.from_dict(test_labels_dict, orient="index")

    # make sure that target column is unchanged (within expected bounds)
    con_target_column = cl_args.target_con_columns[0]
    assert df_test[con_target_column].max() < 220
    assert df_test[con_target_column].min() > 130

    cat_target_column = cl_args.target_cat_columns[0]
    assert set(df_test[cat_target_column]) == {"Asia", "Africa", "Europe"}

    # make sure that ExtraTarget column is as expected
    extra_as_int = df_test["ExtraTarget"].astype(int)
    height_as_int = df_test["Height"].astype(int)
    assert (extra_as_int == (height_as_int - 50)).all()

    if not keep_outputs:
        cleanup(run_path)


@patch("human_origins_supervised.predict._load_transformers", autospec=True)
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
    mocked_load_transformers, create_test_data, create_test_cl_args
):
    test_data_config = create_test_data
    c = test_data_config
    cl_args = create_test_cl_args
    classes_tested = sorted(list(c.target_classes.keys()))

    test_labels_dict = predict._load_labels_for_testing(test_train_cl_args_mix=cl_args)

    target_column = create_test_cl_args.target_cat_columns[0]
    mock_encoder = LabelEncoder().fit(["Asia", "Europe", "Africa"])
    mocked_load_transformers.return_value = {target_column: mock_encoder}

    test_dataset = predict._set_up_test_dataset(
        test_train_cl_args_mix=cl_args, test_labels_dict=test_labels_dict
    )

    exp_no_samples = c.n_per_class * len(classes_tested)

    check_dataset(
        dataset=test_dataset,
        exp_no_sample=exp_no_samples,
        classes_tested=classes_tested,
        target_column=target_column,
    )


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
                "get_acts": False,
            }
        }
    ],
    indirect=True,
)
def test_predict(keep_outputs, prep_modelling_test_configs):
    config, test_config = prep_modelling_test_configs
    test_path = Path(config.cl_args.data_folder).parent

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

    predict.predict(predict_cl_args=predict_cl_args)

    df_test = pd.read_csv(test_path / "predictions.csv", index_col="ID")

    target_column = config.cl_args.target_cat_columns[0]
    target_classes = sorted(config.target_transformers[target_column].classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert (target_classes == df_test.columns).all()

    for cls in target_classes:
        class_indices = [i for i in df_test.index if i.endswith(cls)]
        df_cur_class = df_test.loc[class_indices]
        num_correct = (df_cur_class.idxmax(axis=1) == cls).sum()

        # check that most were correct
        assert num_correct / df_cur_class.shape[0] > 0.9

    if not keep_outputs:
        cleanup(test_config.run_path)
