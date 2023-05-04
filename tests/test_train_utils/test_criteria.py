from typing import Callable
from unittest.mock import MagicMock
from copy import copy

import pytest
import torch
from torch import nn

from eir.setup.config import Configs
from eir.setup.output_setup import set_up_outputs_for_training
from eir.target_setup.target_label_setup import MergedTargetLabels
from eir.train_utils.criteria import (
    get_criteria,
    _calc_con_loss,
    build_loss_dict,
    _parse_loss_name,
    _get_label_smoothing,
    get_loss_callable,
)


def _get_criteria_parametrization():
    base = {
        "injections": {
            "input_configs": [
                {
                    "input_info": {"input_name": "test_genotype"},
                    "model_config": {"model_type": "cnn"},
                },
            ],
            "output_configs": [
                {
                    "output_info": {"output_name": "test_output"},
                    "output_type_info": {
                        "target_cat_columns": ["Origin"],
                        "target_con_columns": ["Height"],
                    },
                }
            ],
        }
    }

    parametrization = []

    for con_case in [
        "MSELoss",
        "L1Loss",
        "SmoothL1Loss",
        "PoissonNLLLoss",
        "GaussianNLLLoss",
    ]:
        cur_case = copy(base)
        cur_output_configs = _get_output_configs(
            cat_loss_name="CrossEntropyLoss", con_loss_name=con_case
        )
        cur_case["injections"]["output_configs"] = cur_output_configs
        parametrization.append(cur_case)

    for cat_case in ["CrossEntropyLoss"]:
        cur_case = copy(base)
        cur_output_configs = _get_output_configs(
            cat_loss_name=cat_case, con_loss_name="MSELoss"
        )
        cur_case["injections"]["output_configs"] = cur_output_configs
        parametrization.append(cur_case)

    return parametrization


def _get_output_configs(cat_loss_name: str, con_loss_name: str) -> list[dict]:
    base = [
        {
            "output_info": {"output_name": "test_output"},
            "output_type_info": {
                "target_cat_columns": ["Origin"],
                "target_con_columns": ["Height"],
                "cat_loss_name": cat_loss_name,
                "con_loss_name": con_loss_name,
            },
        }
    ]

    return base


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    _get_criteria_parametrization(),
    indirect=True,
)
def test_get_criteria(
    create_test_config: Configs,
    create_test_labels: MergedTargetLabels,
):
    target_labels = create_test_labels

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    test_criteria = get_criteria(outputs_as_dict=outputs_as_dict)
    test_input = torch.randn(10, 1)
    test_target = torch.randn(10, 1)

    for output_name, output_object in outputs_as_dict.items():
        for column_name in output_object.target_columns["con"]:
            cur_callable = test_criteria[output_name][column_name]
            assert cur_callable.func is _calc_con_loss
            cur_callable(input=test_input, target=test_target)

        for column_name in output_object.target_columns["cat"]:
            cur_callable = test_criteria[output_name][column_name]
            assert isinstance(cur_callable, nn.CrossEntropyLoss)
            cur_callable(input=test_input, target=test_target)


def test_build_loss_dict():
    loss_dict = build_loss_dict()
    assert isinstance(loss_dict, dict)
    assert set(loss_dict.keys()) == {"cat", "con"}
    assert isinstance(loss_dict["cat"], list)
    assert isinstance(loss_dict["con"], list)


def test_parse_loss_name():
    output_config = MagicMock()
    output_config.output_type_info.cat_loss_name = "CrossEntropyLoss"
    output_config.output_type_info.con_loss_name = "MSELoss"
    column_type = "cat"
    result = _parse_loss_name(output_config=output_config, column_type=column_type)
    assert result == "CrossEntropyLoss"

    column_type = "con"
    result = _parse_loss_name(output_config=output_config, column_type=column_type)
    assert result == "MSELoss"

    with pytest.raises(ValueError):
        _parse_loss_name(output_config=output_config, column_type="unknown")


def test_get_label_smoothing():
    output_config = MagicMock()
    output_config.output_type_info.cat_label_smoothing = 0.1
    assert _get_label_smoothing(output_config=output_config, column_type="cat") == 0.1
    assert _get_label_smoothing(output_config=output_config, column_type="con") == 0.0

    with pytest.raises(ValueError):
        _get_label_smoothing(output_config=output_config, column_type="unknown")


def test_calc_con_loss():
    input_tensor = torch.randn(10, 1)
    target = torch.randn(10, 1)
    loss_func = nn.MSELoss()
    loss = _calc_con_loss(input=input_tensor, target=target, loss_func=loss_func)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_get_loss_callable():
    criteria = MagicMock()
    loss_callable = get_loss_callable(criteria=criteria)
    assert isinstance(loss_callable, Callable)
    assert loss_callable.keywords["criteria"] == criteria
