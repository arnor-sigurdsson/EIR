from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import pytest
import torch

from eir.serve_modules.serve_input_setup import (
    ServeBatch,
    _impute_missing_tabular_values,
    _setup_tabular_input_for_serve,
    general_pre_process,
    general_pre_process_raw_inputs,
    general_pre_process_raw_inputs_wrapper,
    get_input_setup_function_for_serve,
    parse_request_input_data_wrapper,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo, InputConfig
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.schemas import InputDataConfig, TabularInputDataConfig


@pytest.fixture
def mock_serve_experiment():
    mock_exp = Mock()
    mock_exp.inputs = {
        "tabular_input": Mock(spec=ComputedServeTabularInputInfo),
        "sequence_input": Mock(spec=ComputedSequenceInputInfo),
    }
    mock_exp.hooks = Mock()
    mock_exp.hooks.step_func_hooks.base_prepare_batch = []
    mock_exp.fabric = Mock()
    mock_exp.fabric.device = "cpu"
    return mock_exp


def test_general_pre_process(mock_serve_experiment):
    data = [
        {
            "tabular_input": {"feature1": 1.0, "feature2": "category1"},
            "sequence_input": "ATGC",
        },
        {
            "tabular_input": {"feature1": 2.0, "feature2": "category2"},
            "sequence_input": "GCTA",
        },
    ]

    with (
        patch(
            "eir.serve_modules.serve_input_setup.parse_request_input_data_wrapper"
        ) as mock_parse,
        patch(
            "eir.serve_modules.serve_input_setup.general_pre_process_raw_inputs_wrapper"
        ) as mock_prepare,
        patch(
            "eir.serve_modules.serve_input_setup.call_hooks_stage_iterable"
        ) as mock_collate,
    ):
        mock_parse.return_value = data
        mock_prepare.return_value = [
            {
                "tabular_input": torch.tensor([1.0, 0]),
                "sequence_input": torch.tensor([1, 2, 3, 4]),
            },
            {
                "tabular_input": torch.tensor([2.0, 1]),
                "sequence_input": torch.tensor([4, 3, 2, 1]),
            },
        ]
        mock_collate.return_value = {
            "batch": ServeBatch(
                pre_hook_inputs={},
                inputs={
                    "tabular_input": torch.tensor([[1.0, 0], [2.0, 1]]),
                    "sequence_input": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
                },
                inputs_split={},
                target_labels={},
                ids=["Serve_0", "Serve_1"],
            )
        }

        result = general_pre_process(data, mock_serve_experiment)

        assert isinstance(result, ServeBatch)
        assert "tabular_input" in result.inputs
        assert "sequence_input" in result.inputs
        assert len(result.ids) == 2
        assert result.ids[0].startswith("Serve_")


def test_parse_request_input_data_wrapper():
    data = [{"feature1": 1.0, "feature2": "category1"}]
    input_objects = {"tabular_input": Mock(spec=ComputedServeTabularInputInfo)}

    with (
        patch("eir.serve_modules.serve_input_setup._load_request_data") as mock_load,
        patch(
            "eir.serve_modules.serve_input_setup.prepare_request_input_data_wrapper"
        ) as mock_prepare,
    ):
        mock_load.return_value = data
        mock_prepare.return_value = [
            {"tabular_input": {"feature1": 1.0, "feature2": 0}}
        ]

        result = parse_request_input_data_wrapper(data, input_objects)

        assert len(result) == 1
        assert "tabular_input" in result[0]
        assert result[0]["tabular_input"]["feature1"] == 1.0


def test_get_input_setup_function_for_serve():
    setup_func = get_input_setup_function_for_serve("tabular")
    assert setup_func == _setup_tabular_input_for_serve

    with pytest.raises(KeyError):
        get_input_setup_function_for_serve("unknown_type")


@patch("eir.serve_modules.serve_input_setup.load_transformers")
def test_setup_tabular_input_for_serve(mock_load_transformers):
    mock_load_transformers.return_value = {
        "input_name": {
            "feature1": Mock(),
            "feature2": Mock(),
        }
    }

    input_config = create_autospec(spec=InputConfig, instance=True)

    input_info = create_autospec(spec=InputDataConfig, instance=True)
    input_config.input_info = input_info
    input_config.input_info.input_name = "input_name"

    input_type_info = create_autospec(spec=TabularInputDataConfig, instance=True)
    input_config.input_type_info = input_type_info
    input_config.input_type_info.input_cat_columns = ["feature2"]
    input_config.input_type_info.input_con_columns = ["feature1"]

    result = _setup_tabular_input_for_serve(
        input_config=input_config, output_folder=Path("/output")
    )

    assert isinstance(result, ComputedServeTabularInputInfo)
    assert result.input_config == input_config


def test_general_pre_process_raw_inputs_wrapper(mock_serve_experiment):
    raw_inputs = [
        {
            "tabular_input": {"feature1": 1.0, "feature2": "category1"},
            "sequence_input": "ATGC",
        },
        {
            "tabular_input": {"feature1": 2.0, "feature2": "category2"},
            "sequence_input": "GCTA",
        },
    ]

    with patch(
        "eir.serve_modules.serve_input_setup.general_pre_process_raw_inputs"
    ) as mock_process:
        mock_process.side_effect = [
            {
                "tabular_input": torch.tensor([1.0, 0]),
                "sequence_input": torch.tensor([1, 2, 3, 4]),
            },
            {
                "tabular_input": torch.tensor([2.0, 1]),
                "sequence_input": torch.tensor([4, 3, 2, 1]),
            },
        ]

        result = general_pre_process_raw_inputs_wrapper(
            raw_inputs, mock_serve_experiment
        )

        assert len(result) == 2
        assert torch.all(result[0]["tabular_input"].eq(torch.tensor([1.0, 0])))
        assert torch.all(result[1]["sequence_input"].eq(torch.tensor([4, 3, 2, 1])))


def test_general_pre_process_raw_inputs(mock_serve_experiment):
    raw_input = {
        "tabular_input": {"feature1": 1.0, "feature2": "category1"},
        "sequence_input": "ATGC",
    }

    mock_serve_experiment.inputs["tabular_input"].encode_func = lambda x: {
        "feature1": 1.0,
        "feature2": 0,
    }
    mock_serve_experiment.inputs["sequence_input"].encode_func = lambda x: [1, 2, 3, 4]

    with (
        patch(
            "eir.serve_modules.serve_input_setup.prepare_inputs_memory"
        ) as mock_prepare,
        patch(
            "eir.serve_modules.serve_input_setup.impute_missing_modalities_wrapper"
        ) as mock_impute,
    ):
        mock_prepare.return_value = {
            "tabular_input": torch.tensor([1.0, 0]),
            "sequence_input": torch.tensor([1, 2, 3, 4]),
        }
        mock_impute.return_value = {
            "tabular_input": torch.tensor([1.0, 0]),
            "sequence_input": torch.tensor([1, 2, 3, 4]),
        }

        result = general_pre_process_raw_inputs(raw_input, mock_serve_experiment)

        assert "tabular_input" in result
        assert "sequence_input" in result
        assert torch.all(result["tabular_input"].eq(torch.tensor([1.0, 0])))
        assert torch.all(result["sequence_input"].eq(torch.tensor([1, 2, 3, 4])))


def test_impute_missing_tabular_values():
    input_object = Mock(spec=ComputedServeTabularInputInfo)
    inputs_values = {"feature1": 1.0, "feature2": "category1"}

    result = _impute_missing_tabular_values(input_object, inputs_values)

    assert result == inputs_values


if __name__ == "__main__":
    pytest.main()
