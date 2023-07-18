from typing import Any, Dict

import pytest

from eir.data_load.data_source_modules import local_ops
from eir.setup.config import Configs
from tests.setup_tests.fixtures_create_data import TestDataConfig


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
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_get_file_sample_id_iterator(
    create_test_config: Configs,
    create_test_data: "TestDataConfig",
    parse_test_cl_args: Dict[str, Any],
):
    test_experiment_config = create_test_config
    test_data_config = create_test_data

    input_configs = test_experiment_config.input_configs
    assert len(input_configs) == 1

    input_data_source = input_configs[0].input_info.input_source

    iterator = local_ops.get_file_sample_id_iterator(
        data_source=input_data_source, ids_to_keep=None
    )
    all_ids = [i for i in iterator]

    assert len(all_ids) == test_data_config.n_per_class * len(
        test_data_config.target_classes
    )

    iterator_empty = local_ops.get_file_sample_id_iterator(
        data_source=input_data_source, ids_to_keep=["does_not_exists"]
    )
    all_ids = [i for i in iterator_empty]
    assert len(all_ids) == 0
