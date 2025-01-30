from collections.abc import Callable
from copy import copy
from functools import partial
from unittest.mock import MagicMock, create_autospec

import pytest
import torch
from torch import nn

from eir.setup.config import Configs
from eir.setup.input_setup import set_up_inputs_for_training
from eir.setup.output_setup import set_up_outputs_for_training
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.setup.schemas import OutputConfig
from eir.target_setup.target_label_setup import MergedTargetLabels
from eir.train_utils.criteria import (
    _calc_con_loss,
    _get_label_smoothing,
    _parse_loss_name,
    build_loss_dict,
    get_criteria,
    get_loss_callable,
    get_supervised_criterion,
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
                    "output_info": {"output_name": "test_output_tabular"},
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
            "output_info": {"output_name": "test_output_tabular"},
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

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=list(target_labels.train_labels["ID"]),
        valid_ids=list(target_labels.valid_labels["ID"]),
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=target_labels.label_transformers,
    )

    test_criteria = get_criteria(outputs_as_dict=outputs_as_dict)
    test_input = {"Height": torch.randn(10, 1), "Origin": torch.randn(10, 3)}
    test_target = {"Height": torch.randn(10, 1), "Origin": torch.randint(0, 3, (10,))}
    test_criteria["test_output_tabular"](test_input, test_target)

    for output_name in outputs_as_dict:
        cur_criteria = test_criteria[output_name]
        cur_criteria(test_input, test_target)


def test_build_loss_dict():
    loss_dict = build_loss_dict()
    assert isinstance(loss_dict, dict)
    assert set(loss_dict.keys()) == {"cat", "con"}
    assert isinstance(loss_dict["cat"], list)
    assert isinstance(loss_dict["con"], list)


def test_parse_loss_name():
    output_config = create_autospec(spec=OutputConfig)
    output_config.output_type_info = create_autospec(spec=TabularOutputTypeConfig)
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
    output_config = create_autospec(spec=OutputConfig)
    output_config.output_type_info = create_autospec(spec=TabularOutputTypeConfig)
    output_config.output_type_info.cat_label_smoothing = 0.1
    assert _get_label_smoothing(output_config=output_config, column_type="cat") == 0.1
    assert _get_label_smoothing(output_config=output_config, column_type="con") == 0.0

    with pytest.raises(ValueError):
        _get_label_smoothing(output_config=output_config, column_type="unknown")


@pytest.mark.parametrize(
    "column_type, loss_name, cat_label_smoothing, expected_type",
    [
        ("con", "MSELoss", 0.0, partial),
        ("con", "L1Loss", 0.0, partial),
        ("con", "SmoothL1Loss", 0.0, partial),
        ("con", "PoissonNLLLoss", 0.0, partial),
        ("con", "HuberLoss", 0.0, partial),
        ("cat", "CrossEntropyLoss", 0.0, nn.CrossEntropyLoss),
    ],
)
def test_get_supervised_criterion(
    column_type,
    loss_name,
    cat_label_smoothing,
    expected_type,
):
    result = get_supervised_criterion(
        column_type_=column_type,
        loss_name=loss_name,
        cat_label_smoothing_=cat_label_smoothing,
    )
    assert isinstance(result, expected_type)

    with pytest.raises(AssertionError):
        get_supervised_criterion(
            column_type_=column_type,
            loss_name="NonExistentLoss",
            cat_label_smoothing_=cat_label_smoothing,
        )


@pytest.mark.parametrize(
    "loss_func",
    [
        nn.MSELoss(),
        nn.L1Loss(),
        nn.SmoothL1Loss(),
        nn.PoissonNLLLoss(),
        nn.HuberLoss(),
    ],
)
def test_calc_con_loss(loss_func):
    input_tensor = torch.randn(10, 1)
    target = torch.randn(10, 1)

    loss = _calc_con_loss(input=input_tensor, target=target, loss_func=loss_func)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])


def test_get_loss_callable():
    criteria = MagicMock()
    loss_callable = get_loss_callable(criteria=criteria)
    assert isinstance(loss_callable, Callable)
    assert loss_callable.keywords["criteria"] == criteria
