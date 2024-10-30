import numpy as np
import torch

from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    _should_skip_modality,
)
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import (
    _streamline_tabular_data_for_transformers,
)


class MockLabelTransformer:
    def __init__(self, name):
        self.name = name

    def transform(self, value):
        return value * 2


def test_streamline_tabular_data_for_transformers():
    tabular_input = {
        "feature1": np.array([1, 2, 3]),
        "feature2": np.array([4, 5, 6]),
    }

    transformers = {
        "feature1": MockLabelTransformer("feature1"),
        "feature2": MockLabelTransformer("feature2"),
    }

    result = _streamline_tabular_data_for_transformers(
        tabular_input=tabular_input, transformers=transformers
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == set(tabular_input.keys())
    for name, value in result.items():
        assert isinstance(value, torch.Tensor)
        expected = transformers[name].transform(tabular_input[name])
        value_np = value.numpy().squeeze()
        assert np.array_equal(value_np, expected)


def test_modality_skip_when_test_mode() -> None:
    assert not _should_skip_modality(modality_dropout_rate=0.5, test_mode=True)


def test_modality_skip_when_modality_dropout_rate_zero() -> None:
    assert not _should_skip_modality(modality_dropout_rate=0.0, test_mode=False)


def test_modality_skip_when_test_mode_false_modality_dropout_rate_non_zero() -> None:
    torch.manual_seed(0)
    assert _should_skip_modality(modality_dropout_rate=1.0, test_mode=False)
