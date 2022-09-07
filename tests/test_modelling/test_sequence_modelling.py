from copy import copy
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pandas as pd
import pytest

from eir import train
from eir.models.model_setup import (
    _get_hf_sequence_feature_extractor_objects,
    _get_manual_out_features_for_external_feature_extractor,
)
from eir.setup.setup_utils import get_all_hf_model_names
from eir.train_utils.utils import seed_everything
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.setup_modelling_test_data.setup_sequence_test_data import (
    get_continent_keyword_map,
)
from tests.test_modelling.test_modelling_utils import (
    check_performance_result_wrapper,
)

seed_everything(seed=0)


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("sequence",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Positional Encoding and Max Pooling
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "encode", "pool": "max"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
        # Case 2: Classification - Positional Embedding, Windowed, Auto dff,
        # Avg Pooling and mixing
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "mixing_alpha": 0.1,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {
                            "window_size": 16,
                            "position": "embed",
                            "pool": "avg",
                            "model_init_config": {"dim_feedforward": "auto"},
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
        # Case 3: Regression
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "output_folder": "test_regression",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": [],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            },
        },
        # Case 4: Multi Task
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "output_folder": "test_multi_task",
                    "gradient_noise": 0.001,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "embed"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    }
                ],
            },
        },
        # Case 5: Multi Task with Mixing
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "output_folder": "test_multi_task_with_mixing",
                    "mixing_alpha": 0.5,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    }
                ],
            },
        },
        # Case 6: External model: Albert
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "output_folder": "test_albert",
                    "mixing_alpha": 0.0,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {
                            "position": "embed",
                            "window_size": 16,
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
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_sequence_modelling(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    thresholds = get_sequence_test_args(
        mixing=experiment.configs.global_config.mixing_alpha
    )

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        cat_targets = output_config.output_type_info.target_cat_columns
        con_targets = output_config.output_type_info.target_con_columns

        for target_name in cat_targets:

            activation_paths = test_config.activations_paths[output_name][target_name]
            target_transformer = experiment.outputs[output_name].target_transformers[
                target_name
            ]

            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                thresholds=thresholds,
            )

            for input_name in experiment.inputs.keys():
                cur_activation_root = activation_paths[input_name]
                _check_sequence_activations_wrapper(
                    activation_root_folder=cur_activation_root,
                    target_classes=target_transformer.classes_,
                    strict=True,
                )

        for _ in con_targets:
            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                thresholds=thresholds,
            )


def get_sequence_test_args(mixing: float) -> Tuple[float, float]:

    thresholds = (0.8, 0.7)
    if mixing:
        thresholds = (0.0, 0.7)

    return thresholds


def _check_sequence_activations_wrapper(
    activation_root_folder: Path,
    target_classes: Sequence[str],
    strict: bool = True,
):
    """
    We have the strict flag as in some cases it will by default predict one class,
    not being specifically activated by input tokens for that class, while only
    being activated by the other N-1 class tokens to move the prediction towards
    those.
    """

    seq_csv_gen = _get_sequence_activations_csv_generator(
        activation_root_folder=activation_root_folder,
        target_classes=target_classes,
    )
    cat_class_keyword_map = get_continent_keyword_map()

    targets_acts_success = []
    multi_class = False if len(target_classes) == 2 else True

    for target_class, csv_file in seq_csv_gen:
        df_seq_acts = pd.read_csv(filepath_or_buffer=csv_file)
        expected_tokens = cat_class_keyword_map[target_class]
        success = _check_sequence_activations(
            df_activations=df_seq_acts,
            top_n_activations=40,
            expected_top_tokens_pool=expected_tokens,
            must_match_n=len(expected_tokens) - 4,
            fail_fast=strict,
        )
        targets_acts_success.append(success)

    if multi_class:

        must_n_successes = len(target_classes) - 1
        if strict:
            must_n_successes = len(target_classes)

        assert sum(targets_acts_success) >= must_n_successes
    else:
        assert any(targets_acts_success)


def _get_sequence_activations_csv_generator(
    activation_root_folder: Path, target_classes: Iterable[str]
):
    for target_class in target_classes:
        cur_path = (
            activation_root_folder
            / target_class
            / f"token_influence_{target_class}.csv"
        )
        yield target_class, cur_path


def _check_sequence_activations(
    df_activations: pd.DataFrame,
    top_n_activations: int,
    expected_top_tokens_pool: Iterable[str],
    must_match_n: int,
    fail_fast: bool = True,
) -> bool:
    df_activations = df_activations.sort_values(by="Shap_Value", ascending=False)
    df_top_n_rows = df_activations.head(top_n_activations)
    top_tokens = df_top_n_rows["Token"]

    matching = [i for i in top_tokens if i in expected_top_tokens_pool]

    success = len(matching) >= must_match_n
    if fail_fast:
        assert success, (matching, top_tokens)
    return success


@pytest.mark.parametrize(
    "model_name",
    get_all_hf_model_names(),
)
def test_external_nlp_feature_extractor_forward(model_name: str):

    model_config = _get_common_model_config_overload()
    model_config_parsed = _parse_model_specific_config_values(
        model_config=model_config, model_name=model_name
    )

    feature_extractor_objects = _get_hf_sequence_feature_extractor_objects(
        model_name=model_name,
        model_config=model_config_parsed,
        feature_extractor_max_length=64,
        num_chunks=1,
        pool=None,
    )
    _get_manual_out_features_for_external_feature_extractor(
        input_length=64,
        embedding_dim=feature_extractor_objects.embedding_dim,
        num_chunks=1,
        feature_extractor=feature_extractor_objects.feature_extractor,
        pool=None,
    )


def _get_common_model_config_overload() -> dict:
    n_heads = 4
    n_layers = 2
    embedding_dim = 16

    config = {
        "num_hidden_layers": n_layers,
        "attention_types": [[["global"], 1]],
        "encoder_layers": n_layers,
        "num_encoder_layers": n_layers,
        "decoder_layers": n_layers,
        "num_decoder_layers": n_layers,
        "block_sizes": [2],
        "attention_head_size": 4,
        "embedding_size": embedding_dim,
        "d_embed": embedding_dim,
        "hidden_size": 16,
        "num_attention_heads": n_heads,
        "encoder_attention_heads": n_heads,
        "decoder_attention_heads": n_heads,
        "num_heads": n_heads,
        "intermediate_size": 32,
        "d_model": 16,
        "pad_token_id": 0,
        "attention_window": 16,
        "axial_pos_embds_dim": (4, 12),
        "axial_pos_shape": (8, 8),
    }

    assert config["hidden_size"] % n_heads == 0

    return config


def _parse_model_specific_config_values(model_config: dict, model_name: str) -> dict:
    mc = copy(model_config)

    if model_name == "prophetnet":
        mc.pop("num_hidden_layers")
    elif model_name == "xlm-prophetnet":
        mc.pop("num_hidden_layers")

    return mc
