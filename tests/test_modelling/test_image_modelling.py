from typing import Tuple, Union

import pytest

from eir import train
from eir.setup.config import get_all_targets
from eir.train_utils.utils import seed_everything
from tests.test_modelling.test_modelling_utils import check_test_performance_results

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
                    "run_name": "test_image_classification",
                    "n_epochs": 6,
                    "memory_dataset": True,
                    "get_acts": False,
                    "mixing_type": "mixup",
                    "mixing_alpha": 1.0,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_image"},
                    }
                ],
            },
        },
        # Case 2: Classification - Established ResNet18 architecture
        {
            "injections": {
                "global_configs": {
                    "run_name": "test_image_classification",
                    "n_epochs": 6,
                    "memory_dataset": True,
                    "get_acts": False,
                    "mixing_type": "mixup",
                    "mixing_alpha": 1.0,
                },
                "input_configs": [
                    {
                        "input_info": {
                            "input_name": "test_image",
                        },
                        "input_type_info": {"model_type": "resnet18"},
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

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)

    thresholds = get_image_test_args(
        mixing=experiment.configs.global_config.mixing_type
    )
    for cat_target_column in targets.cat_targets:

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=cat_target_column,
            metric="mcc",
            thresholds=thresholds,
        )

    for con_target_column in targets.con_targets:

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=con_target_column,
            metric="r2",
            thresholds=thresholds,
        )


def get_image_test_args(mixing: Union[None, str]) -> Tuple[float, float]:

    thresholds = (0.7, 0.6)
    if mixing is not None:
        thresholds = (0.0, 0.6)

    return thresholds
