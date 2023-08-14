import reprlib
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig

if TYPE_CHECKING:
    from eir.setup.config import Configs


def validate_train_configs(configs: "Configs") -> None:
    validate_input_configs(input_configs=configs.input_configs)
    validate_output_configs(output_configs=configs.output_configs)


def validate_input_configs(input_configs: Sequence[schemas.InputConfig]) -> None:
    for input_config in input_configs:
        base_validate_input_info(input_info=input_config.input_info)
        input_source = Path(input_config.input_info.input_source)

        match input_config.input_type_info:
            case schemas.TabularInputDataConfig(
                input_cat_columns, input_con_columns, _, _
            ):
                expected_columns = list(input_cat_columns) + list(input_con_columns)
                validate_tabular_file(
                    file_to_check=input_source,
                    expected_columns=expected_columns,
                    name="Tabular input",
                )


def base_validate_input_info(input_info: schemas.InputDataConfig) -> None:
    if not Path(input_info.input_source).exists():
        raise ValueError(
            f"Input source {input_info.input_source} does not exist. "
            f"Please check the path is correct."
        )


def validate_output_configs(output_configs: Sequence[schemas.OutputConfig]) -> None:
    for output_config in output_configs:
        base_validate_output_info(output_info=output_config.output_info)
        output_source = Path(output_config.output_info.output_source)

        match output_config.output_type_info:
            case TabularOutputTypeConfig(
                target_cat_columns, target_con_columns, _, _, _
            ):
                expected_columns = list(target_cat_columns) + list(target_con_columns)
                validate_tabular_file(
                    file_to_check=output_source,
                    expected_columns=expected_columns,
                    name="Tabular output",
                )


def base_validate_output_info(output_info: schemas.OutputInfoConfig) -> None:
    if not Path(output_info.output_source).exists():
        raise ValueError(
            f"Output source {output_info.output_source} does not exist. "
            f"Please check the path is correct."
        )


def validate_tabular_file(
    file_to_check: Path,
    expected_columns: Sequence[str],
    name: str,
) -> None:
    header = pd.read_csv(file_to_check, nrows=0).columns.tolist()
    if "ID" not in header:
        raise ValueError(
            f"{name} file {file_to_check} does not contain a column named 'ID'. "
            f"Please check the input file."
        )

    if not set(expected_columns).issubset(header):
        raise ValueError(
            f"{name} file {file_to_check} does not contain all expected columns: "
            f"{reprlib.repr(expected_columns)}. "
            f"Please check the input file."
        )

    series_ids = pd.read_csv(
        filepath_or_buffer=file_to_check,
        usecols=["ID"],
        engine="pyarrow",
        dtype_backend="pyarrow",
    ).squeeze()
    if not series_ids.is_unique:
        non_unique = series_ids[series_ids.duplicated()].tolist()
        raise ValueError(
            f"{name} file {file_to_check} contains non-unique"
            f" IDs: {reprlib.repr(non_unique)}. "
            f"Please check the input file."
        )
