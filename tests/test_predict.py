from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Union, Sequence, Mapping, Tuple

import pandas as pd
import pytest
import torch
import yaml
from aislib.misc_utils import ensure_path_exists
from sklearn.preprocessing import LabelEncoder

import eir.models.omics.omics_models
import eir.setup.config
import eir.setup.input_setup
import eir.train
from eir import predict
from eir import train
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.omics_models import get_omics_model_init_kwargs
from eir.setup import config
from eir.setup import schemas
from eir.setup.config import object_to_primitives
from tests.conftest import cleanup, TestDataConfig, ModelTestConfig
from tests.test_data_load.test_datasets import check_dataset

al_config_instances = Union[
    schemas.GlobalConfig,
    schemas.InputConfig,
    schemas.PredictorConfig,
    schemas.TargetConfig,
]


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_load_model(create_test_config: config.Configs, tmp_path: Path):
    """
    We need `create_test_data` here because the create_test_model fixture depends on it
    down the line, and we need to pass in params for the subrequest in the
    `create_test_data` fixture definition.
    """

    test_configs = create_test_config
    gc = test_configs.global_config

    data_dimension = eir.setup.input_setup.DataDimensions(
        channels=1, height=4, width=1000
    )

    assert len(test_configs.input_configs) == 1
    cnn_model_config = test_configs.input_configs[0].model_config
    cnn_init_kwargs = get_omics_model_init_kwargs(
        model_type="cnn",
        model_config=cnn_model_config,
        data_dimensions=data_dimension,
    )
    model = CNNModel(**cnn_init_kwargs)
    model = model.to(device=gc.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = predict._load_model(
        model_path=model_path,
        model_class=CNNModel,
        model_init_kwargs=cnn_init_kwargs,
        device=gc.device,
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


def test_get_named_pred_dict_iterators(tmp_path: Path) -> None:

    keys = {"global_configs", "input_configs", "predictor_configs", "target_configs"}
    paths = {}

    for k in keys:
        test_yaml_data = {f"key_{k}": f"value_{k}"}
        cur_outpath = (tmp_path / k).with_suffix(".yaml")
        ensure_path_exists(path=cur_outpath)

        with open(cur_outpath, "w") as out_yaml:
            yaml.dump(data=test_yaml_data, stream=out_yaml)

        paths.setdefault(k, []).append(cur_outpath)

    test_predict_cl_args = Namespace(**paths)

    named_iterators = predict.get_named_pred_dict_iterators(
        predict_cl_args=test_predict_cl_args
    )

    for key, key_iter in named_iterators.items():
        for dict_ in key_iter:
            assert dict_ == {f"key_{key}": f"value_{key}"}


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_train_predict_matched_config_generator(create_test_config, tmp_path: Path):
    test_configs = create_test_config

    test_predict_cl_args = _setup_test_namespace_for_matched_config_test(
        test_configs=test_configs, predict_cl_args_save_path=tmp_path
    )

    named_test_iterators = predict.get_named_pred_dict_iterators(
        predict_cl_args=test_predict_cl_args
    )

    matched_iterator = predict.get_train_predict_matched_config_generator(
        train_configs=test_configs, named_dict_iterators=named_test_iterators
    )

    # TODO: Note that these conditions currently come from
    #       _overload_test_yaml_object_for_predict. Later we can configure this
    #       further.
    for name, train_config_dict, predict_config_dict_to_inject in matched_iterator:
        if name == "input_configs":
            assert train_config_dict != predict_config_dict_to_inject
            assert (
                predict_config_dict_to_inject["input_info"]["input_source"]
                == "predict_input_source_overloaded"
            )
        else:
            assert train_config_dict == predict_config_dict_to_inject


def _setup_test_namespace_for_matched_config_test(
    test_configs: config.Configs,
    predict_cl_args_save_path: Path,
    do_inject_test_values: bool = True,
) -> Namespace:
    keys = ("global_configs", "input_configs", "predictor_configs", "target_configs")
    name_to_attr_map = {
        "global_configs": "global_config",
        "predictor_configs": "predictor_config",
    }
    paths = {}
    for k in keys:
        attr_name = name_to_attr_map.get(k, k)
        test_yaml_obj = getattr(test_configs, attr_name)

        obj_as_primitives = _overload_test_yaml_object_for_predict(
            test_yaml_obj=test_yaml_obj,
            cur_key=k,
            do_inject_test_values=do_inject_test_values,
        )

        if isinstance(obj_as_primitives, Sequence):
            name_object_iterator = enumerate(obj_as_primitives)
        elif isinstance(obj_as_primitives, Mapping):
            name_object_iterator = enumerate([obj_as_primitives])
        else:
            raise ValueError()

        for idx, obj_primitive_to_dump in name_object_iterator:
            cur_outpath = (predict_cl_args_save_path / f"{k}_{idx}").with_suffix(
                ".yaml"
            )
            ensure_path_exists(path=cur_outpath)
            with open(cur_outpath, "w") as out_yaml:
                yaml.dump(data=obj_primitive_to_dump, stream=out_yaml)

            paths.setdefault(k, []).append(cur_outpath)

    test_predict_cl_args = Namespace(**paths)

    return test_predict_cl_args


def _overload_test_yaml_object_for_predict(
    test_yaml_obj: al_config_instances, cur_key: str, do_inject_test_values: bool = True
):
    test_yaml_obj_copy = deepcopy(test_yaml_obj)
    obj_as_primitives = object_to_primitives(obj=test_yaml_obj_copy)
    if cur_key == "input_configs":
        for idx, input_dict in enumerate(obj_as_primitives):
            if do_inject_test_values:
                input_dict = eir.setup.config.recursive_dict_replace(
                    dict_=input_dict,
                    dict_to_inject={
                        "input_info": {
                            "input_source": "predict_input_source_overloaded"
                        }
                    },
                )
            obj_as_primitives[idx] = input_dict

    return obj_as_primitives


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_overload_train_configs_for_predict(
    create_test_config: config.Configs, tmp_path: Path
) -> None:

    test_configs = create_test_config

    test_predict_cl_args = _setup_test_namespace_for_matched_config_test(
        test_configs=test_configs, predict_cl_args_save_path=tmp_path
    )

    named_test_iterators = predict.get_named_pred_dict_iterators(
        predict_cl_args=test_predict_cl_args
    )

    matched_iterator = predict.get_train_predict_matched_config_generator(
        train_configs=test_configs, named_dict_iterators=named_test_iterators
    )

    overloaded_train_config = predict.overload_train_configs_for_predict(
        matched_dict_iterator=matched_iterator
    )

    # TODO: Note that these conditions currently come from
    #       _overload_test_yaml_object_for_predict. Later we can configure this
    #       further.
    for input_config in overloaded_train_config.input_configs:
        assert input_config.input_info.input_source == "predict_input_source_overloaded"


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "run_name": "extra_inputs",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_cat_columns": [],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        },
    ],
    indirect=True,
)
def test_load_labels_for_predict(
    create_test_data: TestDataConfig,
    create_test_config: config.Configs,
    keep_outputs: bool,
):
    """
    Note here we are treating the generated test data (i.e. by tests, not test-set-data)
    as the testing-set.
    """
    test_configs = create_test_config

    run_path = Path(f"runs/{test_configs.global_config.run_name}/")

    test_ids = predict.gather_all_ids_from_target_configs(
        target_configs=test_configs.target_configs
    )

    tabular_infos = train.get_tabular_target_file_infos(
        target_configs=test_configs.target_configs
    )
    assert len(tabular_infos) == 1
    target_tabular_info = tabular_infos[0]

    df_test = predict._load_labels_for_predict(
        tabular_info=target_tabular_info, ids_to_keep=test_ids
    )

    # make sure that target columns are unchanged (within expected bounds)
    assert len(target_tabular_info.con_columns) == 1
    con_target_column = target_tabular_info.con_columns[0]
    assert df_test[con_target_column].max() < 220
    assert df_test[con_target_column].min() > 130

    assert len(target_tabular_info.cat_columns) == 1
    cat_target_column = target_tabular_info.cat_columns[0]
    assert set(df_test[cat_target_column]) == {"Asia", "Africa", "Europe"}

    if not keep_outputs:
        cleanup(run_path)


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {"memory_dataset": True},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {"memory_dataset": False},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_set_up_test_dataset(
    create_test_data: TestDataConfig,
    create_test_config: config.Configs,
):
    test_data_config = create_test_data
    test_configs = create_test_config

    test_ids = predict.gather_all_ids_from_target_configs(
        target_configs=test_configs.target_configs
    )

    tabular_infos = train.get_tabular_target_file_infos(
        target_configs=test_configs.target_configs
    )
    assert len(tabular_infos) == 1
    target_tabular_info = tabular_infos[0]

    df_test = predict._load_labels_for_predict(
        tabular_info=target_tabular_info, ids_to_keep=test_ids
    )

    assert len(target_tabular_info.cat_columns) == 1
    target_column = target_tabular_info.cat_columns[0]
    mock_encoder = LabelEncoder().fit(["Asia", "Europe", "Africa"])
    transformers = {target_column: mock_encoder}

    test_target_labels = predict.parse_labels_for_predict(
        con_targets=target_tabular_info.con_columns,
        cat_targets=target_tabular_info.cat_columns,
        df_labels_test=df_test,
        label_transformers=transformers,
    )

    test_inputs = predict.set_up_inputs_for_testing(
        inputs_configs=test_configs.input_configs,
        ids=test_ids,
        hooks=None,
        run_name=test_configs.global_config.run_name,
    )

    test_dataset = predict._set_up_default_test_dataset(
        configs=test_configs,
        target_labels_dict=test_target_labels,
        inputs_as_dict=test_inputs,
    )

    classes_tested = sorted(list(test_data_config.target_classes.keys()))
    exp_no_samples = test_data_config.n_per_class * len(classes_tested)

    check_dataset(
        dataset=test_dataset,
        exp_no_sample=exp_no_samples,
        classes_tested=classes_tested,
        target_transformers=transformers,
        target_column=target_column,
    )


def grab_best_model_path(saved_models_folder: Path):
    saved_models = [i for i in saved_models_folder.iterdir()]
    saved_models.sort(key=lambda x: float(x.stem.split("=")[-1]))

    return saved_models[-1]


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi", "split_to_test": True}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "run_name": "test_run_predict",
                    "n_epochs": 4,
                    "checkpoint_interval": 50,
                    "sample_interval": 50,
                    "get_acts": False,
                    "batch_size": 64,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "genome-local-net"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "model_type": "tabular",
                            "extra_cat_columns": [],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_predict(
    keep_outputs: bool,
    prep_modelling_test_configs: Tuple[train.Experiment, ModelTestConfig],
    tmp_path: Path,
):
    experiment, model_test_config = prep_modelling_test_configs
    train_configs_for_testing = experiment.configs

    train.train(experiment=experiment)

    test_predict_cl_args_files_only = _setup_test_namespace_for_matched_config_test(
        test_configs=train_configs_for_testing,
        predict_cl_args_save_path=tmp_path,
        do_inject_test_values=False,
    )

    extra_test_predict_kwargs = {
        "model_path": grab_best_model_path(model_test_config.run_path / "saved_models"),
        "evaluate": True,
        "output_folder": tmp_path,
    }
    all_predict_kwargs = {
        **test_predict_cl_args_files_only.__dict__,
        **extra_test_predict_kwargs,
    }
    predict_cl_args = Namespace(**all_predict_kwargs)

    train_configs_for_testing = predict._load_serialized_train_experiment(
        run_folder=model_test_config.run_path
    )

    predict_config = predict.get_default_predict_config(
        loaded_train_experiment=train_configs_for_testing,
        predict_cl_args=predict_cl_args,
    )

    predict.predict(predict_cl_args=predict_cl_args, predict_config=predict_config)

    predict._compute_predict_activations(
        loaded_train_experiment=train_configs_for_testing,
        predict_config=predict_config,
    )

    origin_predictions_path = tmp_path / "Origin" / "predictions.csv"
    df_test = pd.read_csv(origin_predictions_path, index_col="ID")

    tabular_infos = train.get_tabular_target_file_infos(
        target_configs=train_configs_for_testing.configs.target_configs
    )
    assert len(tabular_infos) == 1
    target_tabular_info = tabular_infos[0]

    assert len(target_tabular_info.cat_columns) == 1
    target_column = target_tabular_info.cat_columns[0]

    target_classes = sorted(experiment.target_transformers[target_column].classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert set(target_classes).issubset(set(df_test.columns))

    preds = df_test.drop("True Label", axis=1).values.argmax(axis=1)
    true_labels = df_test["True Label"]

    preds_accuracy = (preds == true_labels).sum() / len(true_labels)
    assert preds_accuracy > 0.95

    assert (tmp_path / "Origin/activations").exists()
    if not keep_outputs:
        cleanup(model_test_config.run_path)
