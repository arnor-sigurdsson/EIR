from typing import Tuple

import pytest

from eir import train
from eir.setup.config import get_all_targets
from eir.train_utils.utils import seed_everything
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
        # Case 1: Classification - Basic Transformer
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification_vanilla_transformer_bytes",
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "mixing_alpha": 1.0,
                    "act_background_samples": 8,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_bytes"},
                        "model_config": {
                            "position": "embed",
                            "window_size": 64,
                            "model_type": "sequence-default",
                            "model_init_config": {
                                "num_heads": 2,
                                "num_layers": 2,
                                "dropout": 0.10,
                            },
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

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)

    thresholds = get_bytes_test_args(
        mixing=experiment.configs.global_config.mixing_alpha
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


def get_bytes_test_args(mixing: float) -> Tuple[float, float]:

    thresholds = (0.7, 0.6)
    if mixing:
        thresholds = (0.0, 0.6)

    return thresholds
