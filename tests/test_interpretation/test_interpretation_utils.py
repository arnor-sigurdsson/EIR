import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from eir.interpretation.interpretation_utils import get_long_format_attribution_df


def test_get_tabular_attribution_df_basic():
    input_data = {
        "feature1": [0.1, 0.2, 0.3],
        "feature2": [0.4, 0.5, 0.6],
        "feature3": [0.7, 0.8, 0.9],
    }
    expected_output = pd.DataFrame(
        {
            "Input": [
                "feature1",
                "feature1",
                "feature1",
                "feature2",
                "feature2",
                "feature2",
                "feature3",
                "feature3",
                "feature3",
            ],
            "Attribution": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    result = get_long_format_attribution_df(parsed_attributions=input_data)
    assert_frame_equal(result, expected_output)


def test_get_tabular_attribution_df_uneven_lists():
    input_data = {
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.5, 0.6],
        "feature3": [0.7, 0.8, 0.9],
    }
    expected_output = pd.DataFrame(
        {
            "Input": [
                "feature1",
                "feature1",
                "feature1",
                "feature1",
                "feature2",
                "feature2",
                "feature3",
                "feature3",
                "feature3",
            ],
            "Attribution": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )

    result = get_long_format_attribution_df(parsed_attributions=input_data)
    assert_frame_equal(result, expected_output)


def test_get_tabular_attribution_df_empty_input():
    input_data: dict[str, list[float]] = {}
    expected_output = pd.DataFrame(columns=["Input", "Attribution"])

    result = get_long_format_attribution_df(parsed_attributions=input_data)
    assert_frame_equal(result, expected_output)


def test_get_tabular_attribution_df_single_feature():
    input_data = {"feature1": [0.1, 0.2, 0.3]}
    expected_output = pd.DataFrame(
        {"Input": ["feature1", "feature1", "feature1"], "Attribution": [0.1, 0.2, 0.3]}
    )

    result = get_long_format_attribution_df(parsed_attributions=input_data)
    assert_frame_equal(result, expected_output)


def test_get_tabular_attribution_df_large_input():
    input_data = {f"feature{i}": [float(i)] * 1000 for i in range(1000)}

    result = get_long_format_attribution_df(parsed_attributions=input_data)

    assert result.shape == (1000000, 2)
    assert set(result["Input"]) == set(input_data.keys())
    assert all(result["Attribution"] == result["Input"].apply(lambda x: float(x[7:])))


def test_get_tabular_attribution_df_output_types():
    input_data = {"feature1": [0.1, 0.2, 0.3], "feature2": [0.4, 0.5, 0.6]}

    result = get_long_format_attribution_df(parsed_attributions=input_data)

    assert isinstance(result, pd.DataFrame)
    assert result.dtypes["Input"] is np.dtype("O")
    assert result.dtypes["Attribution"] is np.dtype("float64")


@pytest.mark.parametrize(
    "input_data, expected_error",
    [
        (None, TypeError),
        ("not a dict", TypeError),
        ({"feature1": "not a list"}, ValueError),
        ({"feature1": [0.1, "not a float"]}, ValueError),
    ],
)
def test_get_tabular_attribution_df_invalid_input(input_data, expected_error):
    with pytest.raises(expected_error):
        get_long_format_attribution_df(parsed_attributions=input_data)
