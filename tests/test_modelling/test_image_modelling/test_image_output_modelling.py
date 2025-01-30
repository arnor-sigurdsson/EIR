from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

from eir import train
from eir.train_utils.utils import seed_everything
from tests.conftest import should_skip_in_gha_macos
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import (
        ModelTestConfig,
        al_modelling_test_configs,
    )

seed_everything(seed=0)


def _get_output_image_data_parameters() -> Sequence[dict]:
    base = {
        "task_type": "multi",
        "modalities": ("image",),
        "extras": {},
        "split_to_test": True,
        "source": "FILL",
    }

    parameters = []

    cur_base = deepcopy(base)
    for source in ["local", "deeplake"]:
        cur_base["source"] = source
        parameters.append(cur_base)

    return parameters


def _get_image_out_parametrization(loss: str) -> dict[str, Any]:
    assert loss in ["mse", "diffusion"]

    # Note we set output name here same as input below for diffusion compatibility
    output_type_info = {
        "loss": loss,
        "size": [16, 16],
    }
    if loss == "diffusion":
        output_type_info["diffusion_time_steps"] = 50

    output_configs = [
        {
            "output_info": {
                "output_name": "test_image",
            },
            "output_type_info": output_type_info,
            "model_config": {
                "model_type": "cnn",
                "model_init_config": {
                    "channel_exp_base": 4,
                    "allow_pooling": False,
                },
            },
        }
    ]

    if loss == "diffusion":
        output_configs[0]["tensor_broker_config"] = {
            "message_configs": [
                {
                    "name": "last_cnn_layer_cat_conv",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "final_layer.0",
                    "use_from_cache": ["first_cnn_layer_copy"],
                    "cache_fusion_type": "cat+conv",
                    "projection_type": "linear",
                },
                {
                    "name": "last_cnn_layer_lcl_sum",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "final_layer.0",
                    "use_from_cache": ["first_downsample_layer"],
                    "cache_fusion_type": "sum",
                    "projection_type": "lcl_residual",
                },
                {
                    "name": "last_cnn_layer_ca",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "final_layer.0",
                    "use_from_cache": ["first_cnn_layer"],
                    "cache_fusion_type": "cross-attention",
                    "projection_type": "sequence",
                },
                {
                    "name": "last_cnn_layer_grouped_linear",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "final_layer.0",
                    "use_from_cache": ["first_downsample_layer"],
                    "cache_fusion_type": "sum",
                    "projection_type": "grouped_linear",
                },
                {
                    "name": "last_cnn_layer_interpolate",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "final_layer.0",
                    "use_from_cache": ["first_downsample_layer"],
                    "cache_fusion_type": "sum",
                    "projection_type": "interpolate",
                },
                {
                    "name": "first_upscale_cnn_layer_lcl_and_mlp_residual",
                    "layer_path": "output_modules.test_image.feature_extractor."
                    "blocks.block_0",
                    "use_from_cache": ["first_cnn_layer_copy"],
                    "cache_fusion_type": "sum",
                    "projection_type": "lcl+mlp_residual",
                },
            ],
        }

    input_configs = [
        {
            "input_info": {"input_name": "test_image"},
            "input_type_info": {
                "auto_augment": False,
                "size": [16, 16],
            },
            "model_config": {
                "model_type": "cnn",
                "model_init_config": {
                    "layers": [1],
                    "kernel_width": 3,
                    "kernel_height": 3,
                    "channel_exp_base": 4,
                    "down_stride_width": 1,
                    "down_stride_height": 1,
                    "attention_inclusion_cutoff": 256,
                    "allow_first_conv_size_reduction": False,
                    "down_sample_every_n_blocks": 1,
                },
            },
        }
    ]

    if loss == "diffusion":
        input_configs[0]["tensor_broker_config"] = {
            "message_configs": [
                {
                    "name": "first_cnn_layer",
                    "layer_path": "input_modules.test_image.feature_extractor"
                    ".conv.0.conv_1",
                    "cache_tensor": True,
                },
                {
                    "name": "first_downsample_layer",
                    "layer_path": "input_modules.test_image.feature_extractor.conv.3",
                    "cache_tensor": True,
                },
            ]
        }

    # Add copy_config only for diffusion
    if loss == "diffusion":
        copy_config = {
            "input_info": {"input_name": "copy_test_image"},
            "input_type_info": {
                "auto_augment": False,
                "size": [16, 16],
            },
            "model_config": {
                "model_type": "cnn",
                "model_init_config": {
                    "layers": [1],
                    "kernel_width": 3,
                    "kernel_height": 3,
                    "channel_exp_base": 4,
                    "down_stride_width": 1,
                    "down_stride_height": 1,
                    "attention_inclusion_cutoff": 256,
                    "allow_first_conv_size_reduction": False,
                    "down_sample_every_n_blocks": 1,
                },
            },
            "tensor_broker_config": {
                "message_configs": [
                    {
                        "name": "first_cnn_layer_copy",
                        "layer_path": "input_modules.copy_test_image.feature_extractor"
                        ".conv.0.conv_1",
                        "cache_tensor": True,
                    },
                ]
            },
        }
        input_configs.append(copy_config)

    epochs = 15 if loss == "mse" else 15
    configs = {
        "global_configs": {
            "basic_experiment": {
                "output_folder": "test_image_generation",
                "n_epochs": epochs,
                "memory_dataset": True,
            }
        },
        "input_configs": input_configs,
        "fusion_configs": {
            "model_type": "pass-through",
            "model_config": {},
        },
        "output_configs": output_configs,
    }

    return configs


@pytest.mark.skipif(condition=should_skip_in_gha_macos(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    _get_output_image_data_parameters(),
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": _get_image_out_parametrization(loss="mse"),
        },
        {
            "injections": _get_image_out_parametrization(loss="diffusion"),
        },
    ],
    indirect=True,
)
def test_image_output_modelling(
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

    _image_output_test_check_wrapper(experiment=experiment, test_config=test_config)


def _image_output_test_check_wrapper(
    experiment: train.Experiment,
    test_config: "ModelTestConfig",
    mse_threshold: float = 0.35,
    cosine_similarity_threshold: float = 0.6,
) -> None:
    output_configs = experiment.configs.output_configs

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        is_diffusion = output_config.output_type_info.loss == "diffusion"

        if output_type != "image":
            continue

        latest_sample = test_config.last_sample_folders[output_name][output_name]
        auto_folder = latest_sample / "auto"

        did_check = False
        for f in auto_folder.iterdir():
            if f.suffix != ".png":
                continue

            generated_image = Image.open(f)
            generated_array = np.array(generated_image) / 255.0

            index = f.name.split("_")[0]
            matching_input_folder = auto_folder / f"{index}_inputs"
            matching_input_array_file = matching_input_folder / "test_image.png"
            matching_input_image = Image.open(matching_input_array_file)
            matching_input_array = np.array(matching_input_image) / 255.0

            mse = mean_squared_error(
                y_true=matching_input_array.ravel(),
                y_pred=generated_array.ravel(),
            )
            assert mse < mse_threshold

            did_check = True

            # due to deeplake arrays not storing 0s but as very small numbers
            matching_input_array[matching_input_array < 1e-8] = 0.0

            # Skip if all 0s or diffusion
            if matching_input_array.sum() == 0 or is_diffusion:
                continue

            cosine_similarity = 1 - cosine(
                u=matching_input_array.ravel().astype(np.float32),
                v=generated_array.ravel(),
            )
            assert cosine_similarity > cosine_similarity_threshold

        assert did_check
