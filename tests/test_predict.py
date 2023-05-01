from argparse import Namespace
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import pytest
import torch

import eir.experiment_io.experiment_io
import eir.models.model_setup
import eir.models.omics.omics_models
import eir.predict_modules.predict_attributions
import eir.predict_modules.predict_config
import eir.predict_modules.predict_data
import eir.predict_modules.predict_input_setup
import eir.predict_modules.predict_target_setup
import eir.setup.config
import eir.setup.input_setup
import eir.setup.input_setup_modules.common
import eir.train
from eir import predict
from eir import train
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.omics_models import get_omics_model_init_kwargs
from eir.setup import config
from tests.conftest import get_system_info
from tests.setup_tests.fixtures_create_experiment import ModelTestConfig
from tests.test_predict_modules.test_predict_config import (
    setup_test_namespace_for_matched_config_test,
)


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
                        "model_config": {"model_type": "cnn"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {"n_iter_before_swa": 20},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "cnn"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
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

    data_dimension = eir.setup.input_setup_modules.common.DataDimensions(
        channels=1, height=4, width=1000
    )

    assert len(test_configs.input_configs) == 1
    cnn_model_config = test_configs.input_configs[0].model_config
    cnn_init_kwargs = get_omics_model_init_kwargs(
        model_type="cnn",
        model_config=cnn_model_config.model_init_config,
        data_dimensions=data_dimension,
    )
    model = CNNModel(**cnn_init_kwargs)
    model = model.to(device=gc.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = eir.models.model_setup.load_model(
        model_path=model_path,
        model_class=CNNModel,
        model_init_kwargs=cnn_init_kwargs,
        device=gc.device,
        test_mode=True,
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


def grab_best_model_path(saved_models_folder: Path):
    saved_models = [i for i in saved_models_folder.iterdir()]
    saved_models.sort(key=lambda x: float(x.stem.split("=")[-1]))

    return saved_models[-1]


def _get_predict_test_data_parametrization() -> List[Dict[str, Any]]:
    """
    We skip the deeplake tests in the GHA Linux host, as for some reason it raises
    a SIGKILL (-9).
    """

    params = [
        {
            "task_type": "multi",
            "split_to_test": True,
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "manual_test_data_creator": lambda: "test_predict",
            "source": "local",
            "extras": {"array_dims": 1},
        }
    ]

    in_gha, platform = get_system_info()

    if in_gha and platform == "Linux":
        return params

    deeplake_params = [
        {
            "task_type": "multi",
            "split_to_test": True,
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_predict",
            "source": "deeplake",
            "extras": {"array_dims": 1},
        }
    ]

    params += deeplake_params

    return params


@pytest.mark.parametrize(
    argnames="act_background_source", argvalues=["train", "predict"]
)
@pytest.mark.parametrize(
    argnames="create_test_data",
    argvalues=_get_predict_test_data_parametrization(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "n_iter_before_swa": 200,
                    "output_folder": "test_run_predict",
                    "n_epochs": 8,
                    "checkpoint_interval": 200,
                    "sample_interval": 200,
                    "lr": 0.001,
                    "attribution_background_samples": 128,
                    "compute_attributions": False,
                    "batch_size": 64,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"subset_snps_file": "auto"},
                        "model_config": {"model_type": "genome-local-net"},
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_sequence_albert"},
                        "model_config": {
                            "window_size": 16,
                            "position": "embed",
                            "model_type": "albert",
                            "model_init_config": {
                                "num_hidden_layers": 2,
                                "num_attention_heads": 4,
                                "embedding_size": 12,
                                "hidden_size": 16,
                                "intermediate_size": 32,
                            },
                        },
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": [],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                    {
                        "input_info": {"input_name": "test_array"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_predict(
    act_background_source: str,
    prep_modelling_test_configs: Tuple[train.Experiment, ModelTestConfig],
    tmp_path: Path,
):
    experiment, model_test_config = prep_modelling_test_configs
    train_configs_for_testing = experiment.configs

    train.train(experiment=experiment)

    test_predict_cl_args_files_only = setup_test_namespace_for_matched_config_test(
        test_configs=train_configs_for_testing,
        predict_cl_args_save_path=tmp_path,
        do_inject_test_values=False,
        monkeypatch_train_to_test_paths=True,
    )

    extra_test_predict_kwargs = {
        "model_path": grab_best_model_path(model_test_config.run_path / "saved_models"),
        "evaluate": True,
        "output_folder": tmp_path,
        "act_background_source": act_background_source,
    }
    all_predict_kwargs = {
        **test_predict_cl_args_files_only.__dict__,
        **extra_test_predict_kwargs,
    }
    predict_cl_args = Namespace(**all_predict_kwargs)

    train_configs_for_testing = (
        eir.experiment_io.experiment_io.load_serialized_train_experiment(
            run_folder=model_test_config.run_path
        )
    )

    predict_config = predict.get_default_predict_config(
        loaded_train_experiment=train_configs_for_testing,
        predict_cl_args=predict_cl_args,
    )

    predict.predict(predict_cl_args=predict_cl_args, predict_config=predict_config)

    eir.predict_modules.predict_attributions.compute_predict_attributions(
        loaded_train_experiment=train_configs_for_testing,
        predict_config=predict_config,
    )

    origin_predictions_path = tmp_path / "test_output" / "Origin" / "predictions.csv"
    df_test = pd.read_csv(origin_predictions_path, index_col="ID")

    tabular_infos = train.get_tabular_target_file_infos(
        output_configs=train_configs_for_testing.configs.output_configs
    )
    assert len(tabular_infos) == 1
    target_tabular_info = tabular_infos["test_output"]

    assert len(target_tabular_info.cat_columns) == 1
    target_column = target_tabular_info.cat_columns[0]

    output = experiment.outputs["test_output"]
    target_classes = sorted(output.target_transformers[target_column].classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert set(target_classes).issubset(set(df_test.columns))

    label_columns = [i for i in df_test.columns if "True Label" in i]
    preds = df_test.drop(label_columns, axis=1).values.argmax(axis=1)
    true_labels = df_test["True Label"]

    preds_accuracy = (preds == true_labels).sum() / len(true_labels)
    assert preds_accuracy > 0.7

    assert (tmp_path / "test_output/Origin/attributions").exists()
