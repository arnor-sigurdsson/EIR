from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from eir.data_load import label_setup
from eir.setup.output_setup import get_output_name_config_iterator
from eir.setup.output_setup_modules.tabular_output_setup import (
    set_up_num_outputs_per_target,
)

if TYPE_CHECKING:
    from eir.setup.config import Configs


def test_set_up_num_classes(get_transformer_test_data):
    df_test, test_target_columns_dict = get_transformer_test_data

    test_transformers = label_setup._get_fit_label_transformers(
        df_labels_train=df_test,
        df_labels_full=df_test,
        label_columns=test_target_columns_dict,
        impute_missing=False,
    )

    num_classes = set_up_num_outputs_per_target(
        target_transformers=test_transformers, cat_loss="CrossEntropyLoss"
    )

    assert num_classes["Height"] == 1
    assert num_classes["Origin"] == 3


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
                            "model_init_config": {"l1": 1e-04},
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

    named_output_configs = get_output_name_config_iterator(
        output_configs=test_configs.output_configs
    )
    for name, config in named_output_configs:
        name_from_config = config.output_info.output_name
        assert name == name_from_config

    expected_to_fail = deepcopy(test_configs)
    expected_to_fail.output_configs[0].output_info.output_name = "test_genotype.failme"

    with pytest.raises(ValueError):
        named_output_configs = get_output_name_config_iterator(
            output_configs=expected_to_fail.output_configs
        )
        for _ in named_output_configs:
            pass
