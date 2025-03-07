from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from eir import predict, train
from eir.experiment_io.experiment_io import load_serialized_train_experiment
from eir.models.input.array.models_cnn import CNNModel
from eir.models.input.omics.omics_models import get_omics_model_init_kwargs
from eir.models.model_setup_modules.model_io import load_model
from eir.predict_modules.predict_attributions import compute_predict_attributions
from eir.setup import config
from eir.setup.input_setup_modules.common import DataDimensions
from eir.target_setup.target_label_setup import get_tabular_target_file_infos
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
                        "output_info": {"output_name": "test_output_tabular"},
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
                "global_configs": {
                    "model": {"n_iter_before_swa": 20},
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "cnn"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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

    data_dimension = DataDimensions(channels=1, height=4, width=1000)

    assert len(test_configs.input_configs) == 1
    cnn_model_config = test_configs.input_configs[0].model_config
    cnn_init_kwargs = get_omics_model_init_kwargs(
        model_type="cnn",
        model_config=cnn_model_config.model_init_config,
        data_dimensions=data_dimension,
    )
    model = CNNModel(**cnn_init_kwargs)
    model = model.to(device=gc.be.device)

    model_path = tmp_path / "model.pt"
    torch.save(obj=model.state_dict(), f=model_path)

    loaded_model = load_model(
        model_path=model_path,
        model_class=CNNModel,
        model_init_kwargs=cnn_init_kwargs,
        device=gc.be.device,
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

    for param_model, param_loaded in zip(
        model.parameters(), loaded_model.parameters(), strict=False
    ):
        assert param_model.data.ne(param_loaded.data).sum() == 0


def grab_best_model_path(saved_models_folder: Path):
    saved_models = list(saved_models_folder.iterdir())
    saved_models.sort(key=lambda x: float(x.stem.split("=")[-1]))

    return saved_models[-1]


def _get_predict_test_data_parametrization() -> list[dict[str, Any]]:
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
                "array",
            ),
            "manual_test_data_creator": lambda: "test_predict",
            "source": "deeplake",
            "extras": {"array_dims": 1},
        }
    ]

    params += deeplake_params

    return params


@pytest.mark.parametrize(
    argnames="attribution_background_source", argvalues=["train", "predict"]
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
                    "basic_experiment": {
                        "output_folder": "test_run_predict",
                        "n_epochs": 8,
                        "batch_size": 64,
                        "dataloader_workers": 2,
                    },
                    "model": {
                        "n_iter_before_swa": 200,
                    },
                    "optimization": {
                        "lr": 0.001,
                    },
                    "evaluation_checkpoint": {
                        "checkpoint_interval": 200,
                        "sample_interval": 200,
                    },
                    "attribution_analysis": {
                        "compute_attributions": False,
                        "attribution_background_samples": 128,
                    },
                    "latent_sampling": {
                        "layers_to_sample": [
                            "fusion_modules.computed.fusion_modules.fusion.0.0.fc_1",
                        ]
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {
                            "subset_snps_file": "auto",
                            "modality_dropout_rate": 0.05,
                        },
                        "model_config": {"model_type": "genome-local-net"},
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "input_type_info": {
                            "modality_dropout_rate": 0.05,
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence_albert"},
                        "input_type_info": {
                            "modality_dropout_rate": 0.05,
                        },
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
                        "input_type_info": {
                            "modality_dropout_rate": 0.05,
                        },
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                        "input_type_info": {
                            "modality_dropout_rate": 0.05,
                        },
                        "model_config": {
                            "model_init_config": {
                                "layers": [2],
                                "kernel_width": 2,
                                "kernel_height": 2,
                                "down_stride_width": 2,
                                "down_stride_height": 2,
                            },
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                            "modality_dropout_rate": 0.05,
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "modality_dropout_rate": 0.05,
                        },
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                                "kernel_height": 1,
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
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                    {
                        "output_info": {"output_name": "test_output_sequence"},
                    },
                    {
                        "output_info": {
                            "output_name": "test_output_array",
                        },
                        "model_config": {
                            "model_type": "lcl",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 3,
                            },
                        },
                    },
                    {
                        "output_info": {
                            "output_name": "test_output_image",
                        },
                        "output_type_info": {
                            "loss": "mse",
                            "size": [16, 16],
                        },
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "channel_exp_base": 4,
                                "allow_pooling": False,
                            },
                        },
                    },
                    {
                        "output_info": {"output_name": "test_output_survival"},
                        "output_type_info": {
                            "event_column": "BinaryOrigin",
                            "time_column": "Time",
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_predict(
    attribution_background_source: str,
    prep_modelling_test_configs: tuple[train.Experiment, ModelTestConfig],
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
        "attribution_background_source": attribution_background_source,
    }
    all_predict_kwargs = {
        **test_predict_cl_args_files_only.__dict__,
        **extra_test_predict_kwargs,
    }
    predict_cl_args = Namespace(**all_predict_kwargs)

    device = predict.maybe_parse_device_from_predict_args(
        predict_cl_args=predict_cl_args
    )

    train_configs_for_testing = load_serialized_train_experiment(
        run_folder=model_test_config.run_path,
        device=device,
    )

    predict_config = predict.get_default_predict_experiment(
        loaded_train_experiment=train_configs_for_testing,
        predict_cl_args=predict_cl_args,
        inferred_run_folder=model_test_config.run_path,
    )

    predict.predict(
        predict_cl_args=predict_cl_args,
        predict_experiment=predict_config,
        run_folder=model_test_config.run_path,
    )

    compute_predict_attributions(
        run_folder=model_test_config.run_path,
        loaded_train_experiment=train_configs_for_testing,
        predict_config=predict_config,
    )

    # We set this up here to have access to the tabular output paths (from train)
    train_configs_for_testing_src_train = load_serialized_train_experiment(
        run_folder=model_test_config.run_path,
        device=device,
        source_folder="configs",
    )

    _check_tabular_predict_results(
        tmp_path_=tmp_path,
        base_experiment=experiment,
        train_configs_for_testing=train_configs_for_testing_src_train,
    )

    _check_sequence_predict_results(tmp_path=tmp_path, expected_n_samples=10)

    _check_array_predict_results(tmp_path=tmp_path, expected_n_samples=10)


def _check_sequence_predict_results(tmp_path: Path, expected_n_samples: int) -> None:
    sequence_predictions_path = (
        tmp_path / "results/test_output_sequence/test_output_sequence/samples/0/auto"
    )
    assert sequence_predictions_path.exists()

    found_files = [i for i in sequence_predictions_path.iterdir() if i.suffix == ".txt"]
    assert len(found_files) == expected_n_samples


def _check_array_predict_results(tmp_path: Path, expected_n_samples: int) -> None:
    array_predictions_path = (
        tmp_path / "results/test_output_array/test_output_array/samples/0/auto"
    )
    assert array_predictions_path.exists()

    found_files = [i for i in array_predictions_path.iterdir() if i.suffix == ".npy"]
    assert len(found_files) == expected_n_samples


def _check_tabular_predict_results(
    tmp_path_: Path,
    base_experiment: train.Experiment,
    train_configs_for_testing: predict.LoadedTrainExperiment,
) -> None:
    origin_predictions_path = (
        tmp_path_ / "test_output_tabular" / "Origin" / "predictions.csv"
    )
    df_test = pd.read_csv(origin_predictions_path, index_col="ID")

    tabular_infos = get_tabular_target_file_infos(
        output_configs=train_configs_for_testing.configs.output_configs
    )
    assert len(tabular_infos) == 2  # both survival and tabular outputs
    target_tabular_info = tabular_infos["test_output_tabular"]

    assert len(target_tabular_info.cat_columns) == 1
    target_column = target_tabular_info.cat_columns[0]

    output = base_experiment.outputs["test_output_tabular"]
    target_classes = sorted(output.target_transformers[target_column].classes_)

    # check that columns in predictions.csv are in correct sorted order
    assert set(target_classes).issubset(set(df_test.columns))

    label_columns = [i for i in df_test.columns if "True Label" in i]
    predictions = df_test.drop(label_columns, axis=1).values.argmax(axis=1)
    true_labels = df_test["True Label"]

    prediction_accuracy = (predictions == true_labels).sum() / len(true_labels)
    assert prediction_accuracy > 0.7

    assert (tmp_path_ / "test_output_tabular/Origin/attributions").exists()
