import tempfile
from pathlib import Path

import polars as pl
import pytest

from eir.setup.config_validation import (
    validate_input_configs,
    validate_output_configs,
)
from eir.setup.schemas import (
    InputConfig,
    InputDataConfig,
    OutputConfig,
    OutputInfoConfig,
    TabularInputDataConfig,
    TabularOutputTypeConfig,
)


def test_tabular_input_duplicate_columns():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_file = tmp_path / "input.csv"
        df = pl.DataFrame({"ID": [1, 2, 3], "col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.write_csv(input_file)

        input_config = InputConfig(
            input_info=InputDataConfig(
                input_source=str(input_file),
                input_name="test_input",
                input_type="tabular",
            ),
            input_type_info=TabularInputDataConfig(
                input_cat_columns=["col1"],
                input_con_columns=["col2", "col1"],
            ),
            model_config=None,
        )

        with pytest.raises(ValueError, match="duplicate column names"):
            validate_input_configs(input_configs=[input_config])


def test_tabular_output_duplicate_columns():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_file = tmp_path / "output.csv"
        df = pl.DataFrame(
            {
                "ID": [1, 2, 3],
                "target1": [1, 2, 3],
                "target2": [4, 5, 6],
            }
        )
        df.write_csv(output_file)

        output_config = OutputConfig(
            output_info=OutputInfoConfig(
                output_source=str(output_file),
                output_name="test_output",
                output_type="tabular",
            ),
            output_type_info=TabularOutputTypeConfig(
                target_cat_columns=["target1"],
                target_con_columns=["target2", "target1"],
            ),
            model_config=None,
        )

        with pytest.raises(ValueError, match="duplicate column names"):
            validate_output_configs(output_configs=[output_config])


def test_tabular_input_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_file = tmp_path / "input.csv"
        df = pl.DataFrame({"ID": [1, 2, 3], "col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.write_csv(input_file)

        input_config = InputConfig(
            input_info=InputDataConfig(
                input_source=str(input_file),
                input_name="test_input",
                input_type="tabular",
            ),
            input_type_info=TabularInputDataConfig(
                input_cat_columns=["col1"],
                input_con_columns=["col2"],
            ),
            model_config=None,
        )

        validate_input_configs(input_configs=[input_config])


def test_tabular_output_no_duplicates():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        output_file = tmp_path / "output.csv"
        df = pl.DataFrame(
            {
                "ID": [1, 2, 3],
                "target1": [1, 2, 3],
                "target2": [4, 5, 6],
            }
        )
        df.write_csv(output_file)

        output_config = OutputConfig(
            output_info=OutputInfoConfig(
                output_source=str(output_file),
                output_name="test_output",
                output_type="tabular",
            ),
            output_type_info=TabularOutputTypeConfig(
                target_cat_columns=["target1"],
                target_con_columns=["target2"],
            ),
            model_config=None,
        )

        validate_output_configs(output_configs=[output_config])
