import numpy as np

from eir.train_utils.train_handlers_sequence_output import (
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
        assert isinstance(value, np.ndarray)
        expected = transformers[name].transform(tabular_input[name])
        assert np.array_equal(value, expected)
