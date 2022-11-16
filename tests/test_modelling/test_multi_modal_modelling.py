from typing import Tuple, TYPE_CHECKING

import pytest

from eir import train
from tests.test_modelling.test_modelling_utils import (
    check_performance_result_wrapper,
)

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
            "random_samples_dropped_from_modalities": True,
            "source": "local",
        },
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "source": "deeplake",
        },
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
                    "gradient_clipping": 1.0,
                    "lr": 0.001,
                    "gradient_noise": 0.01,
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
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
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
                        "output_info": {"output_name": "test_output_copy"},
                        "output_type_info": {
                            "target_cat_columns": [],
                            "target_con_columns": ["Height"],
                        },
                    },
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
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

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        thresholds=(0.85, 0.85),
    )
