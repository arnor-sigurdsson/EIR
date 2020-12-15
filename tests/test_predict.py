import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from snp_pred import predict
from snp_pred import train
from snp_pred.data_load.label_setup import TabularFileInfo
from snp_pred.models.models_cnn import CNNModel
from tests.conftest import cleanup
from tests.test_data_load.test_datasets import check_dataset


def test_load_model(args_config, tmp_path):
    """
    We need `create_test_data` here because the create_test_model fixture depends on it
    down the line, and we need to pass in params for the subrequest in the
    `create_test_data` fixture definition.
    """

    cl_args = args_config

    num_outputs_per_target = {"Origin": 3}
    model = CNNModel(
        cl_args=cl_args,
        num_outputs_per_target=num_outputs_per_target,
    )
    model = model.to(device=cl_args.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = predict._load_model(
        model_path=model_path,
        model_class=CNNModel,
        model_init_kwargs={
            "cl_args": cl_args,
            "num_outputs_per_target": num_outputs_per_target,
        },
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
    cl_args_from_train = Namespace(data_source="train/data/folder", lr=1e-3)
    cl_args_from_predict = Namespace(data_source="test/data/folder")

    mixed_args = predict._modify_train_cl_args_for_testing(
        cl_args_from_train, cl_args_from_predict
    )
    assert mixed_args.data_source == "test/data/folder"
    assert mixed_args.lr == 1e-3


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {
            "custom_cl_args": {
                "extra_con_columns": ["ExtraTarget"],
                "target_con_columns": ["Height"],
                "target_cat_columns": ["Origin"],
            }
        }
    ],
    indirect=True,
)
def test_load_labels_for_predict(
    create_test_data, create_test_datasets, create_test_cl_args, keep_outputs
):
    """
    Note here we are treating the generated test data (i.e. by tests, not test-set-data)
    as the testing-set.
    """
    cl_args = create_test_cl_args

    run_path = Path(f"runs/{cl_args.run_name}/")

    test_ids = predict.gather_ids_from_data_source(
        data_source=Path(cl_args.data_source)
    )
    tabular_info = set_up_all_label_data(cl_args=cl_args)

    df_test = predict._load_labels_for_predict(
        tabular_info=tabular_info, ids_to_keep=test_ids
    )

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


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_cl_args",
    [
        {"custom_cl_args": {"memory_dataset": True}},
        {"custom_cl_args": {"memory_dataset": False}},
    ],
    indirect=True,
)
def test_set_up_test_dataset(create_test_data, create_test_cl_args):
    test_data_config = create_test_data
    c = test_data_config
    cl_args = create_test_cl_args
    classes_tested = sorted(list(c.target_classes.keys()))

    test_ids = predict.gather_ids_from_data_source(
        data_source=Path(cl_args.data_source)
    )
    tabular_info = set_up_all_label_data(cl_args=cl_args)

    df_test = predict._load_labels_for_predict(
        tabular_info=tabular_info, ids_to_keep=test_ids
    )

    target_column = create_test_cl_args.target_cat_columns[0]
    mock_encoder = LabelEncoder().fit(["Asia", "Europe", "Africa"])
    transformers = {target_column: mock_encoder}

    df_test_dict = predict.parse_labels_for_testing(
        tabular_info=tabular_info,
        df_labels_test=df_test,
        label_transformers=transformers,
    )

    test_dataset = predict._set_up_test_dataset(
        test_train_cl_args_mix=cl_args,
        test_labels_dict=df_test_dict,
        tabular_inputs_labels_dict=None,
    )

    exp_no_samples = c.n_per_class * len(classes_tested)

    check_dataset(
        dataset=test_dataset,
        exp_no_sample=exp_no_samples,
        classes_tested=classes_tested,
        target_transformers=transformers,
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
                "sample_interval": 50,
                # to save time since we're not testing the modelling
                "get_acts": False,
            }
        }
    ],
    indirect=True,
)
def test_predict(keep_outputs, prep_modelling_test_configs):
    config, test_config = prep_modelling_test_configs
    test_path = Path(config.cl_args.data_source).parent

    train.train(config)

    model_path = grab_latest_model_path(test_config.run_path / "saved_models")
    predict_cl_args = Namespace(
        model_path=model_path,
        batch_size=64,
        evaluate=True,
        data_source=test_path / "test_arrays_test_set",
        output_folder=test_path,
        device="cpu",
        num_workers=0,
    )

    predict_config = predict.get_default_predict_config(
        run_folder=test_config.run_path, predict_cl_args=predict_cl_args
    )

    predict.predict(predict_cl_args=predict_cl_args, predict_config=predict_config)

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
        assert num_correct / df_cur_class.shape[0] > 0.80

    if not keep_outputs:
        cleanup(test_config.run_path)


def set_up_all_label_data(cl_args: argparse.Namespace) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=cl_args.label_file,
        con_columns=cl_args.target_con_columns + cl_args.extra_con_columns,
        cat_columns=cl_args.target_cat_columns + cl_args.extra_cat_columns,
        parsing_chunk_size=cl_args.label_parsing_chunk_size,
    )

    return table_info
