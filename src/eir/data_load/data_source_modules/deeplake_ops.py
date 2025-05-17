import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

warnings.filterwarnings("ignore", message=".*newer version of deeplake.*")

import deeplake
import numpy as np
import polars as pl
from tqdm import tqdm

if TYPE_CHECKING:
    from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
        InputHookOutput,
    )


@lru_cache
def is_deeplake_dataset(data_source: str) -> bool:
    return bool(deeplake.exists(data_source))


@lru_cache
def load_deeplake_dataset(data_source: str) -> deeplake.ReadOnlyDataset:
    dataset = deeplake.open_read_only(
        url=data_source,
        token=None,
    )

    columns = {col.name for col in dataset.schema.columns}
    if "ID" not in columns:
        raise ValueError(
            f"DeepLake dataset at {data_source} does not have an ID column. "
            f"Please add one to the dataset."
        )
    return dataset


def get_deeplake_input_source_iterable(
    deeplake_dataset: deeplake.ReadOnlyDataset, inner_key: str
) -> Any:
    columns = {col.name for col in deeplake_dataset.schema.columns}

    existence_col = f"{inner_key}_exists"
    for row in deeplake_dataset:
        if is_deeplake_sample_missing(
            row=row,
            existence_col=existence_col,
            columns=columns,
        ):
            continue

        value = row[inner_key]

        yield value


def add_deeplake_data_to_df(
    input_source: str,
    input_name: str,
    deeplake_input_inner_key: str,
    input_df: pl.DataFrame,
    data_loading_hook: "InputHookOutput",
    ids_to_keep: None | set[str],
) -> pl.DataFrame:
    assert deeplake_input_inner_key is not None, "Deeplake input inner key is None"

    deeplake_ds = load_deeplake_dataset(data_source=input_source)

    columns = {col.name for col in deeplake_ds.schema.columns}
    if deeplake_input_inner_key not in columns:
        raise ValueError(
            f"Input key {deeplake_input_inner_key} not found in deeplake dataset "
            f"{input_source}. Available columns are: {columns}."
        )

    total = len(ids_to_keep) if ids_to_keep is not None else len(deeplake_ds)
    existence_col = f"{deeplake_input_inner_key}_exists"

    ids = []
    column_arrays: dict[str, Any] = {}
    hook_callable = data_loading_hook.hook_callable
    hook_dtype = data_loading_hook.return_dtype
    is_list_dype = isinstance(hook_dtype, pl.List)

    with tqdm(total=total, desc=input_name) as pbar:
        for row in deeplake_ds:
            if is_deeplake_sample_missing(
                row=row,
                existence_col=existence_col,
                columns=columns,
            ):
                continue

            sample_id = row["ID"]

            if ids_to_keep is not None and sample_id not in ids_to_keep:
                continue

            sample_data_pointer = row.row_id
            sample_data = hook_callable(sample_data_pointer)

            if isinstance(sample_data, Path):
                sample_data = str(sample_data)

            if isinstance(sample_data, dict):
                for key in sample_data:
                    col_name = f"{input_name}__{key}"
                    if col_name not in column_arrays:
                        column_arrays[col_name] = []

                for key, value in sample_data.items():
                    col_name = f"{input_name}__{key}"
                    column_arrays[col_name].append(value)
                ids.append(sample_id)
            else:
                if input_name not in column_arrays:
                    column_arrays[input_name] = []

                if is_list_dype and isinstance(sample_data, np.ndarray):
                    sample_data = sample_data.tolist()

                column_arrays[input_name].append(sample_data)
                ids.append(sample_id)

            pbar.update(1)

    if not ids:
        return input_df

    df_dict = {"ID": pl.Series(name="ID", values=ids, dtype=pl.Utf8)}

    for col_name, values in column_arrays.items():
        df_dict[col_name] = pl.Series(name=col_name, values=values, dtype=hook_dtype)

    processed_df = pl.DataFrame(df_dict)

    if input_df.height == 0:
        return processed_df
    return input_df.join(processed_df, on="ID", how="full", coalesce=True)


def is_deeplake_sample_missing(
    row: deeplake.RowView,
    existence_col: str,
    columns: set[str],
) -> bool:
    if existence_col in columns:
        is_missing = not row[existence_col].item()  # type: ignore
        return is_missing

    return False
