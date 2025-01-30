from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

from eir import train
from eir.train_utils.utils import seed_everything
from eir.utils.logging import get_logger
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import (
        ModelTestConfig,
        al_modelling_test_configs,
    )

seed_everything(seed=0)

logger = get_logger(name=__name__)


def _get_output_array_data_parameters() -> Sequence[dict]:
    base = {
        "task_type": "multi",
        "modalities": ("array",),
        "extras": {"array_dims": np.nan},
        "split_to_test": True,
        "source": "FILL",
    }

    parameters = []

    for dims in [1]:
        cur_base = deepcopy(base)
        for source in ["local"]:
            cur_base["source"] = source
            cur_base["extras"]["array_dims"] = dims
            parameters.append(cur_base)

    return parameters


def _get_array_out_parametrization(loss: str) -> dict[str, Any]:
    assert loss in ["mse", "diffusion"]

    # Note we set output name here same as input below for diffusion compatibility
    output_type_info = {
        "loss": loss,
    }
    if loss == "diffusion":
        output_type_info["diffusion_time_steps"] = 50

    output_configs = [
        {
            "output_info": {
                "output_name": "test_array",
            },
            "output_type_info": output_type_info,
            "model_config": {
                "model_type": "cnn",
                "model_init_config": {
                    "channel_exp_base": 3,
                    "allow_pooling": False,
                },
            },
        },
    ]

    if loss == "mse":
        output_configs.append(
            {
                "output_info": {
                    "output_name": "test_output_array_lcl",
                },
                "output_type_info": {
                    "loss": loss,
                },
                "model_config": {
                    "model_type": "lcl",
                    "model_init_config": {
                        "kernel_width": 8,
                        "channel_exp_base": 3,
                        "attention_inclusion_cutoff": 128,
                    },
                },
            },
        )

    epochs = 15 if loss == "mse" else 20
    configs = {
        "global_configs": {
            "basic_experiment": {
                "output_folder": "test_array_generation",
                "n_epochs": epochs,
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
                        "layers": [1],
                        "kernel_width": 4,
                        "kernel_height": 4,
                        "channel_exp_base": 4,
                        "down_stride_width": 1,
                        "down_stride_height": 1,
                        "attention_inclusion_cutoff": 256,
                        "allow_first_conv_size_reduction": False,
                        "down_sample_every_n_blocks": 2,
                    },
                },
            },
            {
                "input_info": {"input_name": "copy_test_array"},
                "input_type_info": {
                    "normalization": "channel",
                },
                "model_config": {
                    "model_type": "cnn",
                    "model_init_config": {
                        "kernel_width": 4,
                        "kernel_height": 4,
                        "channel_exp_base": 4,
                        "down_stride_width": 1,
                        "down_stride_height": 1,
                        "attention_inclusion_cutoff": 0,
                        "allow_first_conv_size_reduction": False,
                        "down_sample_every_n_blocks": 2,
                    },
                },
            },
        ],
        "output_configs": output_configs,
    }

    return configs


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    _get_output_array_data_parameters(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": _get_array_out_parametrization(loss="mse"),
        },
        {
            "injections": _get_array_out_parametrization(loss="diffusion"),
        },
    ],
    indirect=True,
)
def test_array_output_modelling(
    prep_modelling_test_configs: "al_modelling_test_configs",
) -> None:
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.5, 0.5),
        min_thresholds=(1.5, 1.5),
    )

    _array_output_test_check_wrapper(experiment=experiment, test_config=test_config)


def _array_output_test_check_wrapper(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    mse_threshold: float = 0.35,
    cosine_similarity_threshold: float = 0.6,
    success_rate_threshold: float = 0.8,
) -> None:
    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        is_diffusion = output_config.output_type_info.loss == "diffusion"

        if output_type != "array":
            continue

        latest_sample = test_config.last_sample_folders[output_name][output_name]
        auto_folder = latest_sample / "auto"

        total_checks = 0
        passed_checks = 0
        did_check = False

        for f in auto_folder.iterdir():
            if f.suffix != ".npy":
                continue

            try:
                generated_array = np.load(str(f))

                index = f.name.split("_")[0]
                matching_input_folder = auto_folder / f"{index}_inputs"
                matching_input_array_file = matching_input_folder / "test_array.npy"
                matching_input_array = np.load(str(matching_input_array_file))

                mse = mean_squared_error(
                    y_true=matching_input_array.ravel(),
                    y_pred=generated_array.ravel(),
                )

                mse_check_passed = mse < mse_threshold
                cosine_check_passed = True

                if not (matching_input_array.sum() == 0 or is_diffusion):
                    matching_input_array = matching_input_array.copy()
                    # due to deeplake arrays not storing 0s but as very small numbers
                    matching_input_array[matching_input_array < 1e-8] = 0.0

                    cosine_similarity = 1 - cosine(
                        u=matching_input_array.ravel().astype(np.float32),
                        v=generated_array.ravel(),
                    )
                    cosine_check_passed = (
                        cosine_similarity > cosine_similarity_threshold
                    )

                if mse_check_passed and cosine_check_passed:
                    passed_checks += 1

                total_checks += 1
                did_check = True

            except Exception as e:
                logger.error(f"Error checking array: {f} - {e}")
                total_checks += 1
                continue

        assert did_check, "No arrays were checked"

        if total_checks > 0:
            success_rate = passed_checks / total_checks
            assert success_rate >= success_rate_threshold, (
                f"Only {success_rate:.1%} of arrays passed all checks, "
                f"which is below the required"
                f" threshold of {success_rate_threshold:.1%}. "
                f"{passed_checks} passed out of {total_checks} total checks."
            )
