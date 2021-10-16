from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import pandas as pd
import pytest

from eir import train
from eir.setup.config import get_all_targets
from eir.train_utils.utils import seed_everything
from tests.test_modelling.setup_modelling_test_data.setup_sequence_test_data import (
    get_continent_keyword_map,
)
from tests.test_modelling.test_modelling_utils import check_test_performance_results

seed_everything(seed=0)


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
        # Case 1: Classification - Positional Encoding
        {
            "injections": {
                "global_configs": {
                    "run_name": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "input_type_info": {"position": "encode"},
                    }
                ],
            },
        },
        # Case 2: Classification - Positional Embedding and Windowed
        {
            "injections": {
                "global_configs": {
                    "run_name": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "input_type_info": {"window_size": 16, "position": "embed"},
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
                    "run_name": "test_regression",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "target_configs": {
                    "target_cat_columns": [],
                    "target_con_columns": ["Height"],
                },
            },
        },
        # Case 4: Multi Task
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "run_name": "test_multi_task",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
        # Case 5: Multi Task with Mixing
        {
            "injections": {
                "global_configs": {
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "run_name": "test_multi_task_with_mixing",
                    "mixing_alpha": 0.5,
                    "mixing_type": "mixup",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                    }
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height", "ExtraTarget"],
                },
            },
        },
    ],
    indirect=True,
)
def test_sequence_modelling(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)

    thresholds = get_sequence_test_args(
        mixing=experiment.configs.global_config.mixing_type
    )
    for cat_target_column in targets.cat_targets:
        target_transformer = experiment.target_transformers[cat_target_column]
        target_classes = target_transformer.classes_

        activation_paths = test_config.activations_paths[cat_target_column]

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=cat_target_column,
            metric="mcc",
            thresholds=thresholds,
        )

        for input_name in experiment.inputs.keys():
            cur_activation_root = activation_paths[input_name]
            _get_sequence_activations_csv_generator(
                activation_root_folder=cur_activation_root,
                target_classes=target_classes,
            )

    for con_target_column in targets.con_targets:
        activation_paths = test_config.activations_paths[con_target_column]

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=con_target_column,
            metric="r2",
            thresholds=thresholds,
        )

        for input_name in experiment.inputs.keys():
            cur_activation_root = activation_paths[input_name]
            _get_sequence_activations_csv_generator(
                activation_root_folder=cur_activation_root,
                target_classes=[con_target_column],
            )


def get_sequence_test_args(mixing: Union[None, str]) -> Tuple[float, float]:

    thresholds = (0.8, 0.7)
    if mixing is not None:
        thresholds = (0.0, 0.7)

    return thresholds


def _check_sequence_activations_wrapper(
    activation_root_folder: Path,
    target_classes: Sequence[str],
):

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
            top_n_activations=20,
            expected_top_tokens_pool=expected_tokens,
            must_match_n=len(expected_tokens) - 3,
            fail_fast=multi_class,
        )
        targets_acts_success.append(success)

    if multi_class:
        assert all(targets_acts_success)
    else:
        assert any(targets_acts_success)


def _get_sequence_activations_csv_generator(
    activation_root_folder: Path, target_classes: Iterable[str]
):
    for target_class in target_classes:
        cur_path = (
            activation_root_folder
            / target_class
            / f"feature_importance_{target_class}.csv"
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
        assert success, matching
    return success
