from typing import Tuple

import pytest

from eir import train
from eir.train_utils.utils import seed_everything
from tests.test_modelling.test_modelling_utils import (
    check_performance_result_wrapper,
)

seed_everything(seed=0)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("image",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Basic Residual Net Configured from scratch
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_image_classification",
                    "n_epochs": 6,
                    "memory_dataset": True,
                    "compute_attributions": True,
                    "attribution_background_samples": 256,
                    "mixing_alpha": 1.0,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_image"},
                        "input_type_info": {
                            "size": [16],
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
        # Case 2: Classification - Established ResNet18 architecture
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_image_classification",
                    "n_epochs": 6,
                    "memory_dataset": True,
                    "compute_attributions": True,
                    "attribution_background_samples": 256,
                    "mixing_alpha": 0.0,
                    "lr": 1e-03,
                    "wd": 0.0,
                },
                "input_configs": [
                    {
                        "input_info": {
                            "input_name": "test_image",
                        },
                        "input_type_info": {
                            "mixing_subtype": "cutmix",
                            "size": [16, 16],
                        },
                        "model_config": {"model_type": "resnet18"},
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
    ],
    indirect=True,
)
def test_image_modelling(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    thresholds = get_image_test_args(
        mixing=experiment.configs.global_config.mixing_alpha
    )

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=thresholds,
    )


def get_image_test_args(mixing: float) -> Tuple[float, float]:
    thresholds = (0.7, 0.6)
    if mixing:
        thresholds = (0.0, 0.6)

    return thresholds
