from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from eir import train
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    pass


def get_array_data_to_test():
    cases = []

    for task in ["multi"]:
        for dim in [1, 2, 3]:
            cur_case = {
                "task_type": task,
                "modalities": ("array",),
                "extras": {"array_dims": dim},
            }
            cases.append(cur_case)

    return cases


def _get_classification_output_configs() -> Sequence[dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output_tabular"},
            "output_type_info": {
                "target_cat_columns": ["Origin"],
                "target_con_columns": [],
            },
        }
    ]

    return output_configs


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data",
    get_array_data_to_test(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: CNN
        {
            "injections": {
                "global_configs": {
                    "training_control": {
                        "weighted_sampling_columns": ["test_output_tabular__Origin"],
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "normalization": "element",
                        },
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                                "kernel_height": 1,
                                "attention_inclusion_cutoff": 256,
                            },
                        },
                    }
                ],
                "output_configs": _get_classification_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_array_classification(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        check_performance_result_wrapper(
            outputs=experiment.outputs,
            run_path=test_config.run_path,
            max_thresholds=(0.8, 0.8),
        )

        output_name = output_config.output_info.output_name
        attribution_paths = test_config.attributions_paths[output_name]

        check_array_activations(
            attribution_path=attribution_paths,
            target_name="Origin",
            at_least_n_matches=4,
            all_attribution_target_classes_must_pass=False,
        )


def check_array_activations(
    attribution_path: dict[str, dict[str, Path]],
    target_name: str,
    at_least_n_matches: int,
    all_attribution_target_classes_must_pass=True,
):
    expected_1d_top_indices = list(range(0, 100, 10))

    successes = []

    for target_folder_name, dict_with_path_to_input in attribution_path.items():
        if target_folder_name != target_name:
            continue

        folder_with_arrays = dict_with_path_to_input["test_array"]

        for target_class_file in folder_with_arrays.iterdir():
            cur_attributions = np.load(file=target_class_file)

            n_dims = len(cur_attributions.shape)
            cur_attributions_summed = np.sum(
                cur_attributions, axis=tuple(range(n_dims - 1))
            )

            assert cur_attributions_summed.shape == (100,)

            top_indices = sorted(np.argsort(cur_attributions_summed)[-10:])

            intersection = set(top_indices).intersection(set(expected_1d_top_indices))
            if len(intersection) >= at_least_n_matches:
                successes.append(True)

    if all_attribution_target_classes_must_pass:
        assert all(successes)
    else:
        must_match_n = len(successes) - 1
        must_match_n = max(must_match_n, 1)
        assert sum(successes) >= must_match_n


def _get_regression_output_configs() -> Sequence[dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output_tabular"},
            "output_type_info": {
                "target_cat_columns": [],
                "target_con_columns": ["Height"],
            },
        }
    ]

    return output_configs


@pytest.mark.skipif(
    condition=should_skip_in_gha_macos(), reason="In GHA and platform is Darwin."
)
@pytest.mark.parametrize(
    "create_test_data",
    get_array_data_to_test(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: CNN
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "memory_dataset": True,
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "normalization": "channel",
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
                    }
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
        # Case 2: LCL
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "normalization": None,
                        },
                        "model_config": {
                            "model_type": "lcl",
                        },
                    },
                    {
                        "input_info": {"input_name": "test_array_lcl_patch"},
                        "input_type_info": {
                            "normalization": None,
                        },
                        "model_config": {
                            "model_type": "lcl",
                            "model_init_config": {
                                "patch_size": [1, 1, 10],
                                "kernel_width": "patch",
                            },
                        },
                    },
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
        # Case 3: Transformer
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_array"},
                        "input_type_info": {
                            "normalization": None,
                        },
                        "model_config": {
                            "model_type": "transformer",
                            "model_init_config": {
                                "patch_size": [1, 1, 10],
                                "embedding_dim": 32,
                            },
                        },
                    },
                ],
                "output_configs": _get_regression_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_array_regression(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        check_performance_result_wrapper(
            outputs=experiment.outputs,
            run_path=test_config.run_path,
            max_thresholds=(0.8, 0.8),
        )

        output_name = output_config.output_info.output_name
        attribution_paths = test_config.attributions_paths[output_name]

        check_array_activations(
            attribution_path=attribution_paths,
            target_name="Height",
            at_least_n_matches=4,
        )
