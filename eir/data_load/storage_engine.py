import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Type

import numpy as np
import polars as pl
import torch

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


class ColumnType(Enum):
    NUMERIC = "numeric"
    STRING = "string"
    PATH = "path"
    FIXED_ARRAY = "fixed_array"
    VAR_ARRAY = "var_array"
    OBJECT = "object"


@dataclass
class ColumnInfo:
    name: str
    dtype: ColumnType
    original_dtype: str
    index: int
    array_shape: Optional[tuple[int, ...]] = None
    inner_dtype: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None


class HybridStorage:
    """
    Storage for mixed-type data using torch.Tensor for numeric data
    and numpy arrays for other data types that cannot be neatly
    represented as tensors.
    """

    def __init__(self):
        self.numeric_data: Optional[torch.Tensor] = None
        self.string_data: Optional[np.ndarray] = None
        self.path_data: Optional[np.ndarray] = None
        self.fixed_array_data: Optional[torch.Tensor] = None
        self.var_array_data: Optional[list[np.ndarray]] = None
        self.object_data: Optional[dict[str, list[object]]] = None

        self.numeric_columns: list[ColumnInfo] = []
        self.string_columns: list[ColumnInfo] = []
        self.path_columns: list[ColumnInfo] = []
        self.fixed_array_columns: list[ColumnInfo] = []
        self.var_array_columns: list[ColumnInfo] = []
        self.object_columns: list[ColumnInfo] = []
        self._num_rows: int = 0

    def from_polars(self, df: pl.DataFrame) -> None:
        self._num_rows = len(df)

        for idx, col in enumerate(df.columns):
            series = df.get_column(col)
            dtype = series.dtype
            dtype_str = str(dtype)

            is_fixed, inner_type, shape = _is_fixed_array(dtype=dtype)
            if is_fixed:
                self.fixed_array_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.FIXED_ARRAY,
                        original_dtype=dtype_str,
                        index=idx,
                        array_shape=shape,
                        inner_dtype=inner_type,
                    )
                )
                continue

            is_var, inner_type = _is_var_array(dtype=dtype)
            if is_var:
                self.var_array_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.VAR_ARRAY,
                        original_dtype=dtype_str,
                        index=idx,
                        inner_dtype=inner_type,
                    )
                )
                continue

            if _is_numeric_dtype(dtype=dtype):
                torch_dtype = (
                    torch.float32 if "float" in dtype_str.lower() else torch.int64
                )
                self.numeric_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.NUMERIC,
                        original_dtype=dtype_str,
                        index=idx,
                        torch_dtype=torch_dtype,
                    )
                )
            elif _is_path_dtype(name=col, values=series):
                self.path_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.PATH,
                        original_dtype=dtype_str,
                        index=idx,
                    )
                )
            elif _is_string_dtype(dtype=dtype):
                self.string_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.STRING,
                        original_dtype=dtype_str,
                        index=idx,
                    )
                )
            else:
                self.object_columns.append(
                    ColumnInfo(
                        name=col,
                        dtype=ColumnType.OBJECT,
                        original_dtype=dtype_str,
                        index=idx,
                    )
                )

        if self.numeric_columns:
            numeric_data = []
            for col_info in self.numeric_columns:
                series = df.get_column(name=col_info.name)
                numeric_data.append(torch.tensor(series.to_numpy()))
            self.numeric_data = torch.stack(numeric_data) if numeric_data else None

        if self.fixed_array_columns:
            fixed_array_data = []
            for col_info in self.fixed_array_columns:
                series = df.get_column(name=col_info.name)
                samples_np = series.to_numpy().astype(col_info.inner_dtype)
                arr = torch.tensor(samples_np)
                fixed_array_data.append(arr)

            data_stacked = torch.stack(fixed_array_data) if fixed_array_data else None
            self.fixed_array_data = data_stacked

        if self.var_array_columns:
            self.var_array_data = []
            for col_info in self.var_array_columns:
                series = df.get_column(name=col_info.name)
                # Store as numpy array for variable length lists
                self.var_array_data.append(series.to_numpy())

        if self.string_columns:
            string_data = []
            for col_info in self.string_columns:
                series = df.get_column(name=col_info.name)
                string_data.append(series.to_numpy())
            self.string_data = np.array(string_data, dtype=np.str_)

        if self.path_columns:
            path_data = []
            for col_info in self.path_columns:
                series = df.get_column(name=col_info.name)
                path_data.append(series.to_numpy())
            self.path_data = np.array(path_data, dtype=np.str_)

        if self.object_columns:
            if self.object_data is None:
                self.object_data = {}

            for col_info in self.object_columns:
                series = df.get_column(name=col_info.name)
                cur_object_data = series.to_list()
                self.object_data[col_info.name] = cur_object_data

    def get_ids(self) -> list[str]:
        id_col_info = next(col for col in self.string_columns if col.name == "ID")
        assert self.string_data is not None
        return self.string_data[self.string_columns.index(id_col_info)].tolist()

    def __len__(self) -> int:
        return self._num_rows

    def __repr__(self) -> str:
        lines = [f"HybridStorage(rows={self._num_rows})"]

        def format_columns(title: str, columns: list[ColumnInfo]) -> list[str]:
            if not columns:
                return []
            result = [f"\n{title} Columns:"]
            for col in columns:
                base = f"  {col.name} ({col.original_dtype})"
                if col.array_shape:
                    base += f" shape={col.array_shape}"
                if col.inner_dtype:
                    base += f" inner_type={col.inner_dtype}"
                result.append(base)
            return result

        sections = [
            ("Numeric", self.numeric_columns),
            ("Fixed Array", self.fixed_array_columns),
            ("Variable Array", self.var_array_columns),
            ("String", self.string_columns),
            ("Path", self.path_columns),
            ("Object", self.object_columns),
        ]

        for title, cols in sections:
            lines.extend(format_columns(title, cols))

        return "\n".join(lines)

    def __str__(self) -> str:
        counts = {
            "numeric": len(self.numeric_columns),
            "fixed_array": len(self.fixed_array_columns),
            "var_array": len(self.var_array_columns),
            "string": len(self.string_columns),
            "path": len(self.path_columns),
            "object": len(self.object_columns),
        }

        type_counts = [f"{k}={v}" for k, v in counts.items() if v > 0]

        return f"HybridStorage(rows={self._num_rows}, " + ", ".join(type_counts) + ")"

    def check_data(self) -> None:
        if self._num_rows == 0:
            raise ValueError(
                "Expected to have at least one sample, but got empty storage. "
                "Possibly there is a mismatch between input IDs and target IDs."
            )

        id_col_info = next(
            (col for col in self.string_columns if col.name == "ID"), None
        )
        if not id_col_info:
            raise ValueError("Could not find required 'ID' column in string columns")

        assert self.string_data is not None
        id_data = self.string_data[self.string_columns.index(id_col_info)]
        null_id_mask = np.char.str_len(id_data) == 0
        if null_id_mask.any():
            null_indices = np.where(null_id_mask)[0]
            raise ValueError(
                f"Expected all observations to have a sample ID associated "
                f"with them, but got rows with null IDs at indices: "
                f"{null_indices.tolist()}"
            )

        for idx in range(self._num_rows):
            row = self.get_row(idx)
            non_id_values = [v for k, v in row.items() if k != "ID"]

            if all(is_null_value(v) for v in non_id_values):
                sample_id = row["ID"]
                raise ValueError(
                    f"Expected all observations to have at least one input value "
                    f"associated with them, but got empty inputs for ID: {sample_id}"
                )

    def validate_storage(self) -> None:
        self.check_data()

        memory_bytes = (
            (
                self.numeric_data.element_size() * self.numeric_data.nelement()
                if self.numeric_data is not None
                else 0
            )
            + (self.string_data.nbytes if self.string_data is not None else 0)
            + (self.path_data.nbytes if self.path_data is not None else 0)
            + (
                self.fixed_array_data.element_size() * self.fixed_array_data.nelement()
                if self.fixed_array_data is not None
                else 0
            )
        )

        if self.var_array_data is not None:
            memory_bytes += sum(arr.nbytes for arr in self.var_array_data)
        if self.object_data is not None:
            memory_bytes += sum(sys.getsizeof(obj) for obj in self.object_data)

        memory_mb = memory_bytes / (1024**2)
        memory_unit = "MB" if memory_mb < 1024 else "GB"
        memory_value = memory_mb if memory_mb < 1024 else memory_mb / 1024

        logger.info(f"Storage is using {memory_value:.4f}{memory_unit} of memory. ")

    def get_row(self, idx: int) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if self.numeric_data is not None:
            # the isnan check there is to guard against the case where
            # if we have nan, casting directly to long will corrupt / convert it to 0
            # in torch
            for i, col_info in enumerate(self.numeric_columns):
                value_tensor = self.numeric_data[i, idx]
                if torch.isnan(value_tensor):
                    value = np.nan
                else:
                    value = value_tensor.to(dtype=col_info.torch_dtype).item()

                result[col_info.name] = value

        if self.fixed_array_data is not None:
            for i, col_info in enumerate(self.fixed_array_columns):
                result[col_info.name] = self.fixed_array_data[i, idx].numpy()

        if self.var_array_data is not None:
            for i, col_info in enumerate(self.var_array_columns):
                result[col_info.name] = self.var_array_data[i][idx]

        if self.string_data is not None:
            for i, col_info in enumerate(self.string_columns):
                result[col_info.name] = self.string_data[i, idx].item()

        if self.path_data is not None:
            for i, col_info in enumerate(self.path_columns):
                result[col_info.name] = self.path_data[i, idx].item()

        if self.object_data is not None:
            for i, col_info in enumerate(self.object_columns):
                result[col_info.name] = self.object_data[col_info.name][idx]

        return result


def _recursive_find_primitive_dtype(dtype: pl.DataType) -> Type[pl.DataType]:
    if hasattr(dtype, "inner"):
        return _recursive_find_primitive_dtype(dtype=dtype.inner)
    return type(dtype)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_fixed_array(
    dtype: pl.DataType,
) -> tuple[bool, Optional[str], Optional[tuple]]:
    if not dtype == pl.Array:
        return False, None, None

    assert isinstance(dtype, pl.Array)

    inner_dtype = dtype.inner
    assert isinstance(inner_dtype, pl.DataType)

    inner_type = _recursive_find_primitive_dtype(dtype=inner_dtype)
    inner_dtype_str = polars_dtype_to_str_dtype(polars_dtype=inner_type)
    shape = dtype.shape

    return True, inner_dtype_str, shape


def _is_var_array(dtype: pl.DataType) -> tuple[bool, Optional[str]]:
    if not dtype == pl.List:
        return False, None

    assert isinstance(dtype, pl.List)
    inner_type = dtype.inner
    return True, str(inner_type)


def _is_path_dtype(name: str, values: pl.Series) -> bool:
    if not isinstance(values[0], str):
        return False
    return any(
        p in str(values[0]).lower()
        for p in ["/var/", "/tmp/", ".csv", ".txt", ".json", ".npy"]
    )


def _is_string_dtype(dtype: pl.DataType) -> bool:
    return dtype == pl.String


def check_two_storages(
    input_storage: HybridStorage,
    target_storage: Optional[HybridStorage],
) -> None:
    if target_storage is None or target_storage._num_rows == 0:
        return

    for idx in range(target_storage._num_rows):
        row = target_storage.get_row(idx)
        non_id_values = [v for k, v in row.items() if k != "ID"]

        if all(is_null_value(v) for v in non_id_values):
            sample_id = row["ID"]
            raise ValueError(
                f"Expected all observations to have at least one target label "
                f"associated with them, but got empty targets for ID: {sample_id}"
            )

    input_ids = set(input_storage.get_ids())
    target_ids = set(target_storage.get_ids())

    if input_ids != target_ids:
        missing_in_input = target_ids - input_ids
        missing_in_target = input_ids - target_ids
        msg = []
        if missing_in_input:
            msg.append(f"IDs in target but not in input: {missing_in_input}")
        if missing_in_target:
            msg.append(f"IDs in input but not in target: {missing_in_target}")
        raise ValueError(
            "ID mismatch between input and target storage. " + " ".join(msg)
        )


def is_null_value(value: Any) -> bool:
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return False
    if isinstance(value, str):
        return value in ("", "None", "null", "nan", "__NULL__")
    if isinstance(value, (int, float)):
        return np.isnan(value) if isinstance(value, float) else False
    return value is None


def polars_dtype_to_torch_dtype(polars_dtype: Type[pl.DataType]) -> torch.dtype:
    dtype_map = {
        pl.Float64: torch.float64,
        pl.Float32: torch.float32,
        pl.Int64: torch.int64,
        pl.Int32: torch.int32,
        pl.Int16: torch.int16,
        pl.Int8: torch.int8,
        pl.UInt64: torch.uint64,
        pl.UInt32: torch.uint32,
        pl.UInt16: torch.uint16,
        pl.UInt8: torch.uint8,
        pl.Boolean: torch.bool,
    }

    return dtype_map[polars_dtype]


def polars_dtype_to_str_dtype(polars_dtype: Type[pl.DataType]) -> str:
    dtype_map = {
        pl.Float64: "float64",
        pl.Float32: "float32",
        pl.Int64: "int64",
        pl.Int32: "int32",
        pl.Int16: "int16",
        pl.Int8: "int8",
        pl.UInt64: "uint64",
        pl.UInt32: "uint32",
        pl.UInt16: "uint16",
        pl.UInt8: "uint8",
        pl.Boolean: "bool",
        pl.Date: "date",
        pl.Datetime: "datetime",
        pl.Utf8: "str",
    }

    return dtype_map[polars_dtype]
