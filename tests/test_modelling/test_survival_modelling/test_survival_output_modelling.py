from typing import TYPE_CHECKING, Dict, Sequence

import pytest

from eir import train
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    pass


def _get_survival_output_configs() -> Sequence[Dict]:
    output_configs = [
        {
            "output_info": {"output_name": "test_output_survival"},
            "output_type_info": {
                "event_column": "BinaryOrigin",
                "time_column": "Time",
            },
        }
    ]

    return output_configs


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: GLN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {"l1": 1e-04},
                        },
                    }
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 256,
                        "layers": [2],
                    }
                },
                "output_configs": _get_survival_output_configs(),
            },
        },
    ],
    indirect=True,
)
def test_survival_modelling(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.7, 0.7),
        min_thresholds=(2.0, 2.0),
    )
