from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from eir import train
from eir.train_utils.utils import seed_everything
from tests.conftest import should_skip_in_gha, should_skip_in_gha_macos
from tests.setup_tests.setup_modelling_test_data.setup_sequence_test_data import (
    get_continent_keyword_map,
)
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

seed_everything(seed=0)


def _get_sequence_test_specific_fusion_configs() -> dict:
    sequence_fusion_configs = {
        "model_config": {
            "fc_task_dim": 256,
            "fc_do": 0.05,
            "rb_do": 0.05,
            "stochastic_depth_p": 0.0,
            "layers": [2],
        }
    }

    return sequence_fusion_configs


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "source": "deeplake",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Positional Encoding and Max Pooling
        # Note we add more capacity to fusion models as it helps make attribution
        # analysis more stable
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_classification",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "encode", "pool": "max"},
                    }
                ],
                "fusion_configs": _get_sequence_test_specific_fusion_configs(),
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
        # Case 2: Classification - Positional Embedding, Windowed, Auto dff,
        # Avg Pooling and mixing, bpe tokenizer
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_classification",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    },
                    "training_control": {
                        "mixing_alpha": 0.1,
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "input_type_info": {"tokenizer": "bpe"},
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
                        "output_info": {"output_name": "test_output_tabular"},
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
                    "basic_experiment": {
                        "n_epochs": 12,
                        "memory_dataset": True,
                        "output_folder": "test_regression",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": [],
                            "target_con_columns": ["Height"],
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

    _sequence_test_check_wrapper(experiment=experiment, test_config=test_config)


@pytest.mark.skipif(condition=should_skip_in_gha(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "source": "local",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 4: Multi Task
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "n_epochs": 12,
                        "memory_dataset": True,
                        "output_folder": "test_multi_task",
                    },
                    "optimization": {
                        "gradient_noise": 0.001,
                    },
                    "attribution_analysis": {
                        "attribution_background_samples": 8,
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "embed", "pool": "avg"},
                    }
                ],
                "fusion_configs": _get_sequence_test_specific_fusion_configs(),
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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
                    "basic_experiment": {
                        "n_epochs": 12,
                        "memory_dataset": True,
                        "output_folder": "test_multi_task_with_mixing",
                    },
                    "training_control": {
                        "mixing_alpha": 0.5,
                    },
                    "attribution_analysis": {
                        "attribution_background_samples": 8,
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "fusion_configs": _get_sequence_test_specific_fusion_configs(),
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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
                    "basic_experiment": {
                        "n_epochs": 12,
                        "memory_dataset": True,
                        "output_folder": "test_albert",
                    },
                    "training_control": {
                        "mixing_alpha": 0.0,
                    },
                    "attribution_analysis": {
                        "attribution_background_samples": 8,
                    },
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
                "fusion_configs": _get_sequence_test_specific_fusion_configs(),
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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
def test_mt_sequence_modelling(prep_modelling_test_configs):
    """
    NOTE: Currently we skip these in GHA as the runners sometimes get a SIGKILL -9
    randomly, not due to this test in particular (running separately works),
    but if the overall task takes too long / too many resources. Might be temporary
    GHA issue, but for now we skip.
    """
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    _sequence_test_check_wrapper(experiment=experiment, test_config=test_config)


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi",
            "modalities": ("sequence",),
            "extras": {"sequence_csv_source": True},
            "split_to_test": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Positional Encoding and Max Pooling
        # Note we add more capacity to fusion models as it helps make attribution
        # analysis more stable
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_classification",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "encode", "pool": "max"},
                    }
                ],
                "fusion_configs": _get_sequence_test_specific_fusion_configs(),
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_sequence_modelling_csv(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    _sequence_test_check_wrapper(experiment=experiment, test_config=test_config)


def _sequence_test_check_wrapper(experiment: train.Experiment, test_config):
    output_configs = experiment.configs.output_configs

    thresholds = get_sequence_test_args(mixing=experiment.configs.gc.tc.mixing_alpha)

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        cat_targets = output_config.output_type_info.target_cat_columns
        con_targets = output_config.output_type_info.target_con_columns

        for target_name in cat_targets:
            attribution_paths = test_config.attributions_paths[output_name][target_name]
            target_transformer = experiment.outputs[output_name].target_transformers[
                target_name
            ]

            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                max_thresholds=thresholds,
            )

            for input_name in experiment.inputs:
                cur_attribution_root = attribution_paths[input_name]
                _check_sequence_attributions_wrapper(
                    attribution_root_folder=cur_attribution_root,
                    target_classes=target_transformer.classes_,
                    strict=False,
                )

        for _ in con_targets:
            check_performance_result_wrapper(
                outputs=experiment.outputs,
                run_path=test_config.run_path,
                max_thresholds=thresholds,
            )


def get_sequence_test_args(mixing: float) -> tuple[float, float]:
    thresholds = (0.8, 0.7)
    if mixing:
        thresholds = (0.0, 0.7)

    return thresholds


def _check_sequence_attributions_wrapper(
    attribution_root_folder: Path,
    target_classes: Sequence[str],
    strict: bool = True,
):
    """
    We have the strict flag as in some cases it will by default predict one class,
    not being specifically activated by input tokens for that class, while only
    being activated by the other N-1 class tokens to move the prediction towards
    those.
    """

    seq_csv_gen = _get_sequence_attributions_csv_generator(
        attribution_root_folder=attribution_root_folder,
        target_classes=target_classes,
    )
    cat_class_keyword_map = get_continent_keyword_map()

    targets_acts_success = []
    multi_class = len(target_classes) != 2

    for target_class, csv_file in seq_csv_gen:
        df_seq_acts = pd.read_csv(filepath_or_buffer=csv_file)
        expected_tokens = cat_class_keyword_map[target_class]
        success = _check_sequence_attributions(
            df_attributions=df_seq_acts,
            top_n_attributions=30,
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


def _get_sequence_attributions_csv_generator(
    attribution_root_folder: Path, target_classes: Iterable[str]
):
    for target_class in target_classes:
        cur_path = (
            attribution_root_folder
            / target_class
            / f"token_influence_{target_class}.csv"
        )
        yield target_class, cur_path


def _check_sequence_attributions(
    df_attributions: pd.DataFrame,
    top_n_attributions: int,
    expected_top_tokens_pool: Iterable[str],
    must_match_n: int,
    fail_fast: bool = True,
) -> bool:
    df_attributions = df_attributions.sort_values(by="Attribution", ascending=False)
    df_top_n_rows = df_attributions.head(top_n_attributions)
    top_tokens = df_top_n_rows["Input"]

    matching = [i for i in top_tokens if i in expected_top_tokens_pool]

    success = len(matching) >= must_match_n
    if fail_fast:
        assert success, (matching, top_tokens)
    return success
