from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest
import torch
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig as SeqOutputModuleConfig,
)
from eir.serve_modules.serve_post_process import (
    _ensure_streamlined_tabular_values,
    _normalize_categorical_outputs,
    _normalize_continuous_outputs,
    _post_process_array_outputs,
    _post_process_tabular_output,
    general_post_process,
)
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import ArrayOutputModuleConfig, TabularOutputModuleConfig


@pytest.fixture
def mock_tabular_output_object():
    mock = create_autospec(ComputedTabularOutputInfo, instance=True)
    mock.output_config = Mock()
    mock.output_config.model_config = Mock(spec=TabularOutputModuleConfig)
    mock.target_columns = {"con": ["continuous_col"], "cat": ["categorical_col"]}
    mock.target_transformers = {
        "continuous_col": Mock(spec=StandardScaler),
        "categorical_col": Mock(classes_=["class1", "class2"]),
    }
    return mock


@pytest.fixture
def mock_sequence_output_object():
    mock = create_autospec(ComputedSequenceOutputInfo, instance=True)
    mock.output_config = Mock()
    mock.output_config.model_config = Mock(spec=SeqOutputModuleConfig)
    mock.output_config.output_type_info = Mock(split_on=" ")
    return mock


@pytest.fixture
def mock_array_output_object():
    mock = create_autospec(ComputedArrayOutputInfo, instance=True)
    mock.output_config = Mock()
    mock.output_config.model_config = Mock(spec=ArrayOutputModuleConfig)
    return mock


def test_ensure_streamlined_tabular_values():
    inputs = {"col1": torch.tensor([1.0, 2.0]), "col2": torch.tensor([3.0, 4.0])}
    result = _ensure_streamlined_tabular_values(tabular_model_outputs=inputs)
    assert result == inputs

    with pytest.raises(AssertionError):
        _ensure_streamlined_tabular_values(
            tabular_model_outputs={"col1": torch.tensor([1.0]), "col2": [2, 3]}
        )


def test_normalize_categorical_outputs():
    inputs = torch.tensor([1.0, 2.0, 0.5])
    result = _normalize_categorical_outputs(outputs=inputs)
    expected = tuple(softmax(inputs))
    assert np.allclose(result, expected)


def test_normalize_continuous_outputs():
    mock_scaler = Mock(spec=StandardScaler)
    mock_scaler.inverse_transform.return_value = np.array([[2.0]])

    inputs = torch.tensor([1.5])
    result = _normalize_continuous_outputs(outputs=inputs, transformer=mock_scaler)

    mock_scaler.inverse_transform.assert_called_once()
    assert result == (2.0,)


def test_post_process_array_outputs():
    array = np.array([1, 2, 3, 4])
    result = _post_process_array_outputs(array=array)
    assert isinstance(result, str)
    assert len(result) > 0


def test_post_process_tabular_output(mock_tabular_output_object):
    tabular_outputs = {
        "continuous_col": torch.tensor([0.5]),
        "categorical_col": torch.tensor([0.7, 0.3]),
    }

    mock_tabular_output_object.target_transformers[
        "continuous_col"
    ].inverse_transform.return_value = np.array([[1.5]])

    result = _post_process_tabular_output(
        output_object=mock_tabular_output_object, tabular_outputs=tabular_outputs
    )

    assert "continuous_col" in result
    assert "categorical_col" in result
    assert result["continuous_col"] == {"continuous_col": 1.5}
    assert len(result["categorical_col"]) == 2
    assert sum(result["categorical_col"].values()) == pytest.approx(1.0)


def test_general_post_process(mock_tabular_output_object, mock_array_output_object):
    outputs = {
        "tabular_output": {
            "continuous_col": torch.tensor([0.5]),
            "categorical_col": torch.tensor([0.7, 0.3]),
        },
        "array_output": {
            "array_output": np.array([1, 2, 3, 4]),
        },
    }
    output_objects = {
        "tabular_output": mock_tabular_output_object,
        "array_output": mock_array_output_object,
    }
    input_objects = {}

    mock_tabular_output_object.target_transformers[
        "continuous_col"
    ].inverse_transform.return_value = np.array([[1.5]])

    with patch(
        "eir.setup.config_setup_modules.config_setup_utils.object_to_primitives",
        autospec=True,
    ) as mock_to_primitives:
        mock_to_primitives.side_effect = lambda x: x

        result = general_post_process(
            outputs=outputs,
            output_objects=output_objects,
            input_objects=input_objects,
        )

    assert "tabular_output" in result
    assert "array_output" in result
    assert isinstance(result["tabular_output"], dict)
    assert isinstance(result["array_output"], str)
    assert result["tabular_output"]["continuous_col"] == {"continuous_col": 1.5}
    assert len(result["tabular_output"]["categorical_col"]) == 2
    assert sum(result["tabular_output"]["categorical_col"].values()) == pytest.approx(
        1.0
    )
