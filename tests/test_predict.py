import argparse
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from eir import predict
from eir import train
from eir.data_load.label_setup import TabularFileInfo
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.omics_models import get_omics_model_init_kwargs
from tests.conftest import cleanup
from tests.test_data_load.test_datasets import check_dataset


def test_load_model(args_config, tmp_path):
    """
    We need `create_test_data` here because the create_test_model fixture depends on it
    down the line, and we need to pass in params for the subrequest in the
    `create_test_data` fixture definition.
    """

    cl_args = args_config

    data_dimension = train.DataDimensions(channels=1, height=4, width=1000)
    cnn_init_kwargs = get_omics_model_init_kwargs(
        model_type="cnn", cl_args=cl_args, data_dimensions=data_dimension
    )
    model = CNNModel(**cnn_init_kwargs)
    model = model.to(device=cl_args.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = predict._load_model(
        model_path=model_path,
        model_class=CNNModel,
        model_init_kwargs=cnn_init_kwargs,
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


@pytest.mark.parametrize(
    "train_cl_args, predict_cl_args, expected_train_after, expected_predict_after",
    [
        (
            Namespace(key1=1, key2=2),
            Namespace(key2="should_be_present_in_train", key3="unique_to_test"),
            Namespace(key1=1, key2="should_be_present_in_train"),
            Namespace(key3="unique_to_test"),
        )
    ],
)
def test_converge_train_and_predict_cl_args(
    train_cl_args: Namespace,
    predict_cl_args: Namespace,
    expected_train_after: Namespace,
    expected_predict_after: Namespace,
) -> None:
    train_converged, predict_converged = predict._converge_train_and_predict_cl_args(
        train_cl_args=train_cl_args, predict_cl_args=predict_cl_args
    )

    assert train_converged == expected_train_after
    assert predict_converged == expected_predict_after


@pytest.mark.parametrize(
    "train_cl_args, predict_cl_args, expected_train_after_overload",
    [
        (
            Namespace(key1=1, key2=2),
            Namespace(key2="should_be_present_in_train"),
            Namespace(key1=1, key2="should_be_present_in_train"),
        )
    ],
)
def test_overload_train_cl_args_for_predict(
    train_cl_args: Namespace, predict_cl_args: Namespace, expected_train_after_overload
) -> None:
    train_after_overload = predict._overload_train_cl_args_for_predict(
        train_cl_args=train_cl_args, predict_cl_args=predict_cl_args
    )
    assert train_after_overload == expected_train_after_overload


@pytest.mark.parametrize(
    "namespace, keys_to_remove, expected_namespace_after_filter",
    [(Namespace(key1=1, key2=2), ["key1"], Namespace(key2=2))],
)
def test_remove_keys_from_namespace(
    namespace: Namespace,
    keys_to_remove: Iterable[str],
    expected_namespace_after_filter: Namespace,
) -> None:

    test_output = predict._remove_keys_from_namespace(
        namespace=namespace, keys_to_remove=keys_to_remove
    )
    assert test_output == expected_namespace_after_filter


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

    test_ids = predict.gather_ids_from_tabular_file(file_path=Path(cl_args.label_file))
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
def test_set_up_test_dataset(
    create_test_data, create_test_cl_args, create_test_data_dimensions
):
    test_data_config = create_test_data
    c = test_data_config
    cl_args = create_test_cl_args
    data_dimensions = create_test_data_dimensions
    classes_tested = sorted(list(c.target_classes.keys()))

    test_ids = predict.gather_ids_from_tabular_file(file_path=Path(cl_args.label_file))
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

    test_dataset = predict._set_up_default_test_dataset(
        data_dimensions=data_dimensions,
        cl_args=cl_args,
        target_labels_dict=df_test_dict,
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
    test_path = Path(config.cl_args.omics_sources[0]).parent

    train.train(config)

    model_path = grab_latest_model_path(test_config.run_path / "saved_models")
    predict_cl_args = Namespace(
        model_path=model_path,
        batch_size=64,
        evaluate=True,
        label_file=config.cl_args.label_file,
        omics_sources=[test_path / "test_arrays_test_set"],
        omics_names=["test"],
        output_folder=test_path,
        device="cpu",
        dataloader_workers=0,
        get_acts=True,
        act_classes=None,
        max_acts_per_class=None,
    )

    train_config = predict._load_serialized_train_config(
        run_folder=test_config.run_path
    )

    predict_config = predict.get_default_predict_config(
        loaded_train_config=train_config, predict_cl_args=predict_cl_args
    )

    predict.predict(predict_cl_args=predict_cl_args, predict_config=predict_config)

    predict._compute_predict_activations(
        train_config=train_config,
        predict_config=predict_config,
        predict_cl_args=predict_cl_args,
    )

    origin_predictions_path = test_path / "Origin" / "predictions.csv"
    df_test = pd.read_csv(origin_predictions_path, index_col="ID")

    target_column = config.cl_args.target_cat_columns[0]
    target_classes = sorted(config.target_transformers[target_column].classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert set(target_classes).issubset(set(df_test.columns))

    for cls in target_classes:
        class_indices = [i for i in df_test.index if i.endswith(cls)]
        df_cur_class = df_test.loc[class_indices]
        num_correct = (df_cur_class.idxmax(axis=1) == cls).sum()

        # check that most were correct
        assert num_correct / df_cur_class.shape[0] > 0.80

    assert (test_path / "Origin/activations").exists()
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
