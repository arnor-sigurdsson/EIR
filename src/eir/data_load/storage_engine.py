import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
import torch
from numpy.typing import DTypeLike

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

NULL_STRINGS = frozenset(("", "None", "null", "nan", "__NULL__"))


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
    array_shape: tuple[int, ...] | None = None
    inner_dtype: str | None = None
    torch_dtype: torch.dtype | None = None
    np_dtype: DTypeLike | None = None


class HybridStorage:
    def __init__(self):
        self.numeric_float_data: torch.Tensor | None = None
        self.numeric_int_data: torch.Tensor | None = None
        self.string_data: np.ndarray | None = None
        self.path_data: np.ndarray | None = None
        self.fixed_array_data: list[torch.Tensor] | None = None
        self.var_array_data: list[np.ndarray] | None = None
        self.object_data: dict[str, list[object]] | None = None

        self.numeric_float_columns: list[ColumnInfo] = []
        self.numeric_int_columns: list[ColumnInfo] = []
        self.string_columns: list[ColumnInfo] = []
        self.path_columns: list[ColumnInfo] = []
        self.fixed_array_columns: list[ColumnInfo] = []
        self.var_array_columns: list[ColumnInfo] = []
        self.object_columns: list[ColumnInfo] = []
        self._num_rows: int = 0

    def from_polars(self, df: pl.DataFrame) -> None:
        self._num_rows = len(df)

        logger.debug("Converting Polars DataFrame to HybridStorage.")

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
                is_float = "float" in dtype_str.lower()
                if is_float:
                    torch_dtype = torch.float32
                    np_dtype_float = np.float32
                    self.numeric_float_columns.append(
                        ColumnInfo(
                            name=col,
                            dtype=ColumnType.NUMERIC,
                            original_dtype=dtype_str,
                            index=idx,
                            torch_dtype=torch_dtype,
                            np_dtype=np_dtype_float,
                        )
                    )
                else:
                    torch_dtype = torch.int64
                    np_dtype_int = np.int64
                    self.numeric_int_columns.append(
                        ColumnInfo(
                            name=col,
                            dtype=ColumnType.NUMERIC,
                            original_dtype=dtype_str,
                            index=idx,
                            torch_dtype=torch_dtype,
                            np_dtype=np_dtype_int,
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

        if self.numeric_float_columns:
            float_data = []
            for col_info in self.numeric_float_columns:
                series = df.get_column(name=col_info.name)
                float_data.append(torch.tensor(series.to_numpy(), dtype=torch.float32))

            final_data = None
            if float_data:
                final_data = torch.stack(float_data).to(dtype=torch.float32)
            self.numeric_float_data = final_data

        if self.numeric_int_columns:
            int_data = []
            for col_info in self.numeric_int_columns:
                series = df.get_column(name=col_info.name)
                # will be cast to float64 with NaN
                # copy as the original data is read-only
                numpy_arr = series.to_numpy().copy()

                numpy_arr[np.isnan(numpy_arr)] = -1
                tensor = torch.tensor(numpy_arr, dtype=torch.int64)
                int_data.append(tensor)

            final_data = None
            if int_data:
                final_data = torch.stack(int_data).to(dtype=torch.int64)
            self.numeric_int_data = final_data

        if self.fixed_array_columns:
            fixed_array_data = []
            for col_info in self.fixed_array_columns:
                series = df.get_column(name=col_info.name)
                samples_np = series.to_numpy().astype(col_info.inner_dtype)
                arr = torch.tensor(samples_np)
                fixed_array_data.append(arr)

            self.fixed_array_data = fixed_array_data

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
            ("Numeric Float", self.numeric_float_columns),
            ("Numeric Int", self.numeric_int_columns),
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
            "numeric_float": len(self.numeric_float_columns),
            "numeric_int": len(self.numeric_int_columns),
            "fixed_array": len(self.fixed_array_columns),
            "var_array": len(self.var_array_columns),
            "string": len(self.string_columns),
            "path": len(self.path_columns),
            "object": len(self.object_columns),
        }

        type_counts = [f"{k}={v}" for k, v in counts.items() if v > 0]

        return f"HybridStorage(rows={self._num_rows}, " + ", ".join(type_counts) + ")"

    def check_data(self) -> None:
        """
        Checks for:
            1. Non-empty storage
            2. Valid ID column
            3. At least one non-null value per row
        """
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
        id_idx = self.string_columns.index(id_col_info)
        id_data = self.string_data[id_idx]

        null_id_mask = np.array([not id_ for id_ in id_data])
        if null_id_mask.any():
            null_indices = np.where(null_id_mask)[0]
            raise ValueError(
                f"Expected all observations to have a sample ID associated "
                f"with them, but got rows with null IDs at indices: "
                f"{null_indices.tolist()}"
            )

        masks = []

        if self.numeric_float_data is not None:
            numeric_mask = torch.isnan(self.numeric_float_data).all(dim=0).cpu().numpy()
            masks.append(numeric_mask)

        if self.numeric_int_data is not None:
            numeric_mask = torch.isnan(self.numeric_int_data).all(dim=0).cpu().numpy()
            masks.append(numeric_mask)

        if self.string_data is not None:
            string_indices = [i for i in range(len(self.string_columns)) if i != id_idx]
            if string_indices:
                string_mask = np.all(
                    [
                        np.array([not bool(x) for x in self.string_data[i]])
                        for i in string_indices
                    ],
                    axis=0,
                )
                masks.append(string_mask)

        if self.fixed_array_data is not None:
            fixed_array_masks = []
            for col_data in self.fixed_array_data:
                nan_mask = torch.isnan(input=col_data)
                other_dims = tuple(range(1, len(nan_mask.shape)))
                fixed_array_mask = (
                    torch.all(input=nan_mask, dim=other_dims).cpu().numpy()
                )
                fixed_array_masks.append(fixed_array_mask)

            masks.append(np.all(fixed_array_masks, axis=0))

        if self.var_array_data:
            var_masks: list[np.ndarray] = []
            for arr_data in self.var_array_data:
                cur_mask: list[bool] = []
                for arr in arr_data:
                    cur_mask.append(arr is None)
                var_masks.append(np.array(cur_mask))

            var_array_mask = np.all(var_masks, axis=0)
            masks.append(var_array_mask)

        if self.path_data is not None:
            path_mask = np.all(check_empty_str_arr(data=self.path_data), axis=0)
            masks.append(path_mask)

        if self.object_data:
            object_mask = np.all(
                [
                    np.array([x is None for x in col_data])
                    for col_data in self.object_data.values()
                ],
                axis=0,
            )
            masks.append(object_mask)

        if masks:
            all_null_mask = np.all(masks, axis=0)
            null_rows = np.where(all_null_mask)[0]

            if len(null_rows) > 0:
                bad_id = id_data[null_rows[0]]
                raise ValueError(
                    f"Expected all observations to have at least one input value "
                    f"associated with them, but got empty inputs for ID: {bad_id}"
                )

    def validate_storage(self, name: str = "") -> int:
        self.check_data()

        memory_bytes = get_total_memory(
            numeric_float_data=self.numeric_float_data,
            numeric_int_data=self.numeric_int_data,
            string_data=self.string_data,
            path_data=self.path_data,
            fixed_array_data=self.fixed_array_data,
            var_array_data=self.var_array_data,
            object_data=self.object_data,
        )

        memory_value, memory_unit = format_memory_size(bytes_size=memory_bytes)

        logger.info(
            f"{name} is using {memory_value:.4f}{memory_unit} of memory "
            f"holding {self._num_rows} samples."
        )

        return memory_bytes

    def get_row(self, idx: int) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if self.numeric_float_data is not None:
            row_values = self.numeric_float_data[:, idx]
            row_isnan = torch.isnan(row_values)

            values_np = row_values.cpu().numpy()
            values_np = np.where(row_isnan.cpu().numpy(), np.nan, values_np)

            for i, col_info in enumerate(self.numeric_float_columns):
                # these are already float32, so no special handling needed for NaN
                result[col_info.name] = values_np[i]

        if self.numeric_int_data is not None:
            row_values = self.numeric_int_data[:, idx]
            values_np = row_values.cpu().numpy()

            for i, col_info in enumerate(self.numeric_int_columns):
                value = values_np[i]
                # Convert -1 sentinel back to None for null values
                if value == -1:
                    result[col_info.name] = None
                else:
                    result[col_info.name] = value.item()

        if self.fixed_array_data is not None:
            for i, col_info in enumerate(self.fixed_array_columns):
                result[col_info.name] = self.fixed_array_data[i][idx].cpu().numpy()

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
            for _, col_info in enumerate(self.object_columns):
                result[col_info.name] = self.object_data[col_info.name][idx]

        return result


def check_empty_str_arr(data: np.ndarray) -> bool:
    str_arr = data.astype(str)
    return (str_arr == "") | (str_arr == "nan")


def _recursive_find_primitive_dtype(dtype: pl.DataType) -> type[pl.DataType]:
    if hasattr(dtype, "inner"):
        return _recursive_find_primitive_dtype(dtype=dtype.inner)
    return type(dtype)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_fixed_array(
    dtype: pl.DataType,
) -> tuple[bool, str | None, tuple | None]:
    if dtype != pl.Array:
        return False, None, None

    assert isinstance(dtype, pl.Array)

    inner_dtype = dtype.inner
    assert isinstance(inner_dtype, pl.DataType)

    inner_type = _recursive_find_primitive_dtype(dtype=inner_dtype)
    inner_dtype_str = polars_dtype_to_str_dtype(polars_dtype=inner_type)
    shape = dtype.shape

    return True, inner_dtype_str, shape


def _is_var_array(dtype: pl.DataType) -> tuple[bool, str | None]:
    if dtype != pl.List:
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
    target_storage: HybridStorage | None,
) -> None:
    if target_storage is None or target_storage._num_rows == 0:
        return

    masks = []

    if target_storage.numeric_float_data is not None:
        numeric_mask = (
            torch.isnan(target_storage.numeric_float_data).all(dim=0).cpu().numpy()
        )
        masks.append(numeric_mask)

    if target_storage.numeric_int_data is not None:
        numeric_mask = (
            torch.isnan(target_storage.numeric_int_data).all(dim=0).cpu().numpy()
        )
        masks.append(numeric_mask)

    if target_storage.string_data is not None:
        id_idx = next(
            i for i, col in enumerate(target_storage.string_columns) if col.name == "ID"
        )
        string_indices = [
            i for i in range(len(target_storage.string_columns)) if i != id_idx
        ]
        if string_indices:
            string_mask = np.all(
                [
                    np.array([not bool(x) for x in target_storage.string_data[i]])
                    for i in string_indices
                ],
                axis=0,
            )
            masks.append(string_mask)

    if target_storage.fixed_array_data is not None:
        fixed_array_masks = []
        for col_data in target_storage.fixed_array_data:
            nan_mask = torch.isnan(input=col_data)
            other_dims = tuple(range(1, len(nan_mask.shape)))
            fixed_array_mask = torch.all(input=nan_mask, dim=other_dims).cpu().numpy()
            fixed_array_masks.append(fixed_array_mask)

        masks.append(np.all(fixed_array_masks, axis=0))

    if target_storage.var_array_data:
        var_masks: list[np.ndarray] = []
        for arr_data in target_storage.var_array_data:
            cur_mask: list[bool] = []
            for arr in arr_data:
                cur_mask.append(arr is None)
            var_masks.append(np.array(cur_mask))

        var_array_mask = np.all(var_masks, axis=0)
        masks.append(var_array_mask)

    if target_storage.object_data:
        object_mask = np.all(
            [
                np.array([x is None for x in col_data])
                for col_data in target_storage.object_data.values()
            ],
            axis=0,
        )
        masks.append(object_mask)

    if masks:
        all_null_mask = np.all(masks, axis=0)
        null_rows = np.where(all_null_mask)[0]
        if len(null_rows) > 0:
            bad_id = target_storage.get_ids()[null_rows[0]]
            raise ValueError(
                f"Expected all observations to have at least one target label "
                f"associated with them, but got empty targets for ID: {bad_id}"
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
    value_type = type(value)
    if value_type in (float, np.float32, np.float64):
        return np.isnan(value)
    if value_type is str:
        return value in NULL_STRINGS
    if value_type is type(None):
        return True
    if value_type is int:
        return False
    return False


def polars_dtype_to_str_dtype(polars_dtype: type[pl.DataType]) -> str:
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


def get_numeric_memory(
    float_data: torch.Tensor | None = None,
    int_data: torch.Tensor | None = None,
) -> int:
    total = 0
    if float_data is not None:
        total += float_data.element_size() * float_data.nelement()
    if int_data is not None:
        total += int_data.element_size() * int_data.nelement()
    return total


def get_fixed_array_memory(data: list[torch.Tensor] | None) -> int:
    if data is None:
        return 0
    return sum(t.element_size() * t.nelement() for t in data)


def get_array_memory(
    data: np.ndarray | None,
) -> int:
    memory_usage = 0
    if data is not None:
        memory_usage += data.nbytes
    return memory_usage


def get_path_memory(data: np.ndarray | None) -> int:
    if data is None:
        return 0
    return data.nbytes


def get_var_array_memory(data: list[np.ndarray] | None) -> int:
    if data is None:
        return 0
    return sum(arr.nbytes for arr in data)


def get_object_memory(data: dict[str, list[object]] | None) -> int:
    if data is None:
        return 0
    return sum(sum(sys.getsizeof(obj) for obj in values) for values in data.values())


def format_memory_size(bytes_size: int) -> tuple[float, str]:
    mb_size = bytes_size / (1024**2)
    if mb_size < 1024:
        return mb_size, "MB"
    return mb_size / 1024, "GB"


def get_total_memory(
    numeric_float_data: torch.Tensor | None = None,
    numeric_int_data: torch.Tensor | None = None,
    string_data: np.ndarray | None = None,
    path_data: np.ndarray | None = None,
    fixed_array_data: list[torch.Tensor] | None = None,
    var_array_data: list[np.ndarray] | None = None,
    object_data: dict[str, list[object]] | None = None,
) -> int:
    return sum(
        [
            get_numeric_memory(
                float_data=numeric_float_data,
                int_data=numeric_int_data,
            ),
            get_fixed_array_memory(data=fixed_array_data),
            get_array_memory(data=string_data),
            get_path_memory(data=path_data),
            get_var_array_memory(data=var_array_data),
            get_object_memory(data=object_data),
        ]
    )
