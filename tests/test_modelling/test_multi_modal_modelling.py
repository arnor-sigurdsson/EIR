from typing import Tuple, TYPE_CHECKING

import pytest

from eir import train
from eir.setup.config import get_all_targets
from tests.test_modelling.test_modelling_utils import check_test_performance_results

if TYPE_CHECKING:
    from tests.conftest import ModelTestConfig


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 6,
                    "act_background_samples": 8,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "extra_cat_columns": ["OriginExtraCol"],
                            "extra_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "predictor_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            },
        }
    ],
    indirect=True,
)
def test_multi_modal_multi_task(
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    targets = get_all_targets(targets_configs=experiment.configs.target_configs)
    for cat_column in targets.cat_targets:

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=cat_column,
            metric="mcc",
            thresholds=(0.9, 0.9),
        )

    for con_column in targets.con_targets:

        check_test_performance_results(
            run_path=test_config.run_path,
            target_column=con_column,
            metric="r2",
            thresholds=(0.9, 0.9),
        )
