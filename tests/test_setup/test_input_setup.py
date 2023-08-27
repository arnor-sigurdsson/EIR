from typing import TYPE_CHECKING

import pytest

from eir.setup import input_setup

if TYPE_CHECKING:
    from eir.setup.config import Configs


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary", "modalities": ["omics", "sequence"]},
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
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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
def test_get_input_name_config_iterator(create_test_config: "Configs"):
    test_configs = create_test_config

    named_input_configs_iterator = input_setup.get_input_name_config_iterator(
        input_configs=test_configs.input_configs
    )
    for name, config in named_input_configs_iterator:
        name_from_config = config.input_info.input_name
        assert name == name_from_config
