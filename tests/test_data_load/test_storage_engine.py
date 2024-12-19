import numpy as np
import polars as pl
import torch

from eir.data_load.storage_engine import NULL_STRINGS, HybridStorage, is_null_value


def test_nan_preservation_in_numeric_data():
    df = pl.DataFrame(
        {
            "ID": ["sample1", "sample2", "sample3"],
            "float_col": [1.5, np.nan, 3.5],
            "int_col": [1, 2, 3],
            "float_col2": [np.nan, 4.5, np.nan],
            "mixed_col": [1, None, 3],
        }
    )

    storage = HybridStorage()
    storage.from_polars(df=df)

    row_0 = storage.get_row(0)
    assert np.isnan(row_0["float_col2"])
    assert row_0["float_col"] == 1.5
    assert row_0["int_col"] == 1

    row_1 = storage.get_row(1)
    assert np.isnan(row_1["float_col"])
    assert np.isnan(row_1["mixed_col"])
    assert row_1["float_col2"] == 4.5
    assert row_1["int_col"] == 2

    row_2 = storage.get_row(2)
    assert np.isnan(row_2["float_col2"])
    assert row_2["float_col"] == 3.5
    assert row_2["int_col"] == 3


def test_hybrid_storage_dtype_handling():
    df = pl.DataFrame(
        {
            "ID": ["sample1", "sample2"],
            "int32_col": pl.Series([1, 2], dtype=pl.Int32),
            "int64_col": pl.Series([3, 4], dtype=pl.Int64),
            "float32_col": pl.Series([1.5, np.nan], dtype=pl.Float32),
            "float64_col": pl.Series([np.nan, 2.5], dtype=pl.Float64),
        }
    )

    storage = HybridStorage()
    storage.from_polars(df)

    row_0 = storage.get_row(0)
    assert isinstance(row_0["int32_col"], (int, np.integer))
    assert isinstance(row_0["int64_col"], (int, np.integer))
    assert isinstance(row_0["float32_col"], (float, np.floating))
    assert np.isnan(row_0["float64_col"])

    row_1 = storage.get_row(1)
    assert isinstance(row_1["int32_col"], (int, np.integer))
    assert isinstance(row_1["int64_col"], (int, np.integer))
    assert np.isnan(row_1["float32_col"])
    assert isinstance(row_1["float64_col"], (float, np.floating))


def test_numeric_data_shape_consistency():
    df = pl.DataFrame(
        {
            "ID": ["sample1", "sample2"],
            "col1": [1.0, np.nan],
            "col2": [np.nan, 2.0],
            "col3": [3.0, 3.0],
        }
    )

    storage = HybridStorage()
    storage.from_polars(df)

    assert storage.numeric_data is not None
    expected_shape = (3, 2)  # (num_columns, num_rows)
    assert storage.numeric_data.shape == expected_shape

    nan_mask = torch.isnan(storage.numeric_data)
    assert nan_mask[1, 0].item()  # col2, first row
    assert nan_mask[0, 1].item()  # col1, second row
    assert not nan_mask[2, :].any()  # col3 has no NaNs


def test_float_null_values():
    # Python float
    assert is_null_value(float("nan"))
    assert is_null_value(np.float64("nan"))
    assert is_null_value(np.float32("nan"))
    assert not is_null_value(1.0)
    assert not is_null_value(0.0)
    assert not is_null_value(-1.0)

    # NumPy float32
    assert is_null_value(np.float32("nan"))
    assert not is_null_value(np.float32(1.0))
    assert not is_null_value(np.float32(0.0))
    assert not is_null_value(np.float32(-1.0))

    # NumPy float64
    assert is_null_value(np.float64("nan"))
    assert not is_null_value(np.float64(1.0))
    assert not is_null_value(np.float64(0.0))
    assert not is_null_value(np.float64(-1.0))


def test_string_null_values():
    # Test all NULL_STRINGS
    for null_str in NULL_STRINGS:
        assert is_null_value(null_str)

    # Test non-null strings
    assert not is_null_value("hello")
    assert not is_null_value("0")
    assert not is_null_value("False")
    assert not is_null_value(" ")  # space
    assert not is_null_value("NaN")  # Different case
    assert not is_null_value("NULL")  # Different case


def test_none_values():
    assert is_null_value(None)


def test_integer_values():
    assert not is_null_value(0)
    assert not is_null_value(-1)
    assert not is_null_value(1)
    assert not is_null_value(np.int32(0))
    assert not is_null_value(np.int64(0))


def test_other_types():
    # Lists
    assert not is_null_value([])
    assert not is_null_value([1, 2, 3])

    # Dictionaries
    assert not is_null_value({})
    assert not is_null_value({"a": 1})

    # Sets
    assert not is_null_value(set())
    assert not is_null_value({1, 2, 3})

    # Boolean
    assert not is_null_value(True)
    assert not is_null_value(False)

    # Numpy boolean
    assert not is_null_value(np.bool_(True))
    assert not is_null_value(np.bool_(False))

    # Numpy array
    assert not is_null_value(np.array([]))
    assert not is_null_value(np.array([1, 2, 3]))


def test_edge_cases():
    # infinity is not NaN
    assert not is_null_value(float("inf"))
    assert not is_null_value(float("-inf"))
    assert not is_null_value(np.float32("inf"))
    assert not is_null_value(np.float64("inf"))

    # very large/small numbers are not NaN
    assert not is_null_value(1e308)
    assert not is_null_value(-1e308)
    assert not is_null_value(1e-308)
    assert not is_null_value(-1e-308)
