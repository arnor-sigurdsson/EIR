import reprlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
)

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
    impute_missing_output_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    InputHookOutput,
    get_input_data_loading_hooks,
    prepare_inputs_disk,
    prepare_inputs_memory,
)
from eir.data_load.data_preparation_modules.output_preparation_wrappers import (
    HookOutput,
    get_output_data_loading_hooks,
    prepare_outputs_disk,
    prepare_outputs_memory,
)
from eir.data_load.data_preparation_modules.prepare_tabular import (
    add_tabular_data_to_df,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.data_load.data_source_modules.local_ops import (
    add_sequence_data_from_csv_to_df,
    get_file_sample_id_iterator_basic,
)
from eir.data_load.data_streaming.streaming_dataset import StreamingDataset
from eir.data_load.label_setup import al_target_labels
from eir.data_load.storage_engine import (
    HybridStorage,
    check_two_storages,
    is_null_value,
)
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.setup import config
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.schemas import SequenceInputDataConfig
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.target_setup.target_label_setup import (
        MergedTargetLabels,
        MissingTargetsInfo,
    )

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryDataset", "DiskDataset", "StreamingDataset"]
al_local_datasets = Union["MemoryDataset", "DiskDataset"]
al_dataset_types = type[al_datasets]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_label_dict_target = dict[str, dict[str, int | float | torch.Tensor]]
al_inputs = dict[str, torch.Tensor] | dict[str, Any]
al_getitem_return = tuple[dict[str, torch.Tensor], al_sample_label_dict_target, str]


def set_up_datasets_from_configs(
    configs: config.Configs,
    target_labels: "MergedTargetLabels",
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    train_ids_to_keep: Sequence[str] | None = None,
    valid_ids_to_keep: Sequence[str] | None = None,
    websocket_url: str | None = None,
) -> tuple[al_datasets, al_local_datasets]:
    train_dataset_class: al_dataset_types = (
        MemoryDataset if configs.gc.be.memory_dataset else DiskDataset
    )
    valid_dataset_class: al_dataset_types = (
        MemoryDataset if configs.gc.be.memory_dataset else DiskDataset
    )

    train_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_df=target_labels.train_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=False,
        ids_to_keep=train_ids_to_keep,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    if websocket_url:
        train_dataset_class = StreamingDataset
        train_kwargs = {
            "websocket_url": websocket_url,
            "inputs": inputs_as_dict,
            "outputs": outputs_as_dict,
            "test_mode": False,
            "batch_size": configs.gc.be.batch_size,
        }

    valid_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_df=target_labels.valid_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=True,
        ids_to_keep=valid_ids_to_keep,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    train_dataset: al_datasets = train_dataset_class(**train_kwargs)
    valid_dataset: al_datasets = valid_dataset_class(**valid_kwargs)

    if not websocket_url:
        assert isinstance(train_dataset, MemoryDataset | DiskDataset)
        assert isinstance(valid_dataset, MemoryDataset | DiskDataset)
        _check_valid_and_train_datasets(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )

    assert isinstance(valid_dataset, MemoryDataset | DiskDataset)
    return train_dataset, valid_dataset


def construct_default_dataset_kwargs_from_cl_args(
    target_labels_df: al_target_labels | None,
    inputs: al_input_objects_as_dict,
    outputs: "al_output_objects_as_dict",
    test_mode: bool,
    missing_ids_per_output: "MissingTargetsInfo",
    ids_to_keep: Sequence[str] | None = None,
) -> dict[str, Any]:
    ids_to_keep_set = set(ids_to_keep) if ids_to_keep is not None else None

    dataset_kwargs = {
        "inputs": inputs,
        "outputs": outputs,
        "target_labels_df": target_labels_df,
        "test_mode": test_mode,
        "missing_ids_per_output": missing_ids_per_output,
        "ids_to_keep": ids_to_keep_set,
    }

    return dataset_kwargs


def _check_valid_and_train_datasets(
    train_dataset: al_local_datasets,
    valid_dataset: al_local_datasets,
) -> None:
    if len(train_dataset.input_storage) < len(valid_dataset.input_storage):
        logger.warning(
            "Size of training dataset (size: %d) is smaller than validation dataset ("
            "size: %d). Generally it is the opposite, but if this is intended please "
            "ignore this message.",
            len(train_dataset.input_storage),
            len(valid_dataset.input_storage),
        )

    train_ids = set(train_dataset.input_storage.get_ids())
    valid_ids = set(valid_dataset.input_storage.get_ids())

    if not train_ids.isdisjoint(valid_ids):
        overlapping_ids = train_ids.intersection(valid_ids)
        raise AssertionError(
            f"Found overlapping IDs between training and "
            f"validation sets: {overlapping_ids}"
        )


def _identity(input_: Any) -> Any:
    return input_


def _get_default_output_hook() -> HookOutput:
    return HookOutput(
        hook_callable=_identity,
        return_dtype=None,
    )


def _get_default_input_hook() -> InputHookOutput:
    return InputHookOutput(
        hook_callable=_identity,
        return_dtype=None,
    )


class DatasetBase(Dataset):
    def __init__(
        self,
        inputs: al_input_objects_as_dict,
        outputs: "al_output_objects_as_dict",
        test_mode: bool,
        missing_ids_per_output: "MissingTargetsInfo",
        target_labels_df: al_target_labels | None = None,
        ids_to_keep: set[str] | None = None,
    ):
        super().__init__()

        self.input_df = pl.DataFrame()

        self.inputs = inputs
        self.outputs = outputs
        self.test_mode = test_mode

        self.target_labels_df: pl.DataFrame
        if target_labels_df is None:
            target_labels_df = pl.DataFrame()
        self.target_labels_df = target_labels_df

        self.input_storage = HybridStorage()
        self.target_labels_storage = HybridStorage()

        self.missing_ids_per_output = missing_ids_per_output
        self.ids_to_keep = set(ids_to_keep) if ids_to_keep else None

    def init_label_attributes(self):
        if not self.outputs:
            raise ValueError("Please specify outputs.")

    def set_up_dfs(
        self,
        input_data_loading_hooks: Mapping[str, InputHookOutput] | None = None,
        output_data_loading_hooks: Mapping[str, HookOutput] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        We do an extra filtering step at the end to account for the situation where
        we have a target label file with more samples than there are any inputs
        available for. This is quite likely if we have e.g. pre-split data into
        train/val and test folders.
        """

        mode_str = "evaluation/test" if self.test_mode else "train"
        logger.debug("Setting up dataset in %s mode.", mode_str)

        if not output_data_loading_hooks:
            output_data_loading_hooks = defaultdict(_get_default_output_hook)

        if not self.target_labels_df.is_empty():
            target_labels_df = _target_labels_hook_load_wrapper(
                target_labels_df=self.target_labels_df,
                output_data_loading_hooks=output_data_loading_hooks,
            )
        else:
            target_labels_df = self.target_labels_df

        if not input_data_loading_hooks:
            input_data_loading_hooks = defaultdict(_get_default_input_hook)

        ids_to_keep = initialize_ids_to_keep(
            target_labels_df=target_labels_df,
            ids_to_keep=self.ids_to_keep,
        )

        input_df = initialize_input_df(ids_to_keep=ids_to_keep)

        input_df = add_data_to_df(
            inputs=self.inputs,
            input_df=input_df,
            ids_to_keep=ids_to_keep,
            data_loading_hooks=input_data_loading_hooks,
        )

        input_df = filter_df(
            input_df=input_df,
            target_labels_df=target_labels_df,
        )

        if len(input_df) == 0:
            raise ValueError("No samples found after filtering.")

        _log_missing_samples_between_modalities(
            df=input_df,
            input_keys=self.inputs.keys(),
        )

        if self.target_labels_df is not None and not self.target_labels_df.is_empty():
            target_labels_df = target_labels_df.filter(
                pl.col("ID").is_in(input_df.get_column("ID").to_list())
            )
            target_labels_df = target_labels_df.sort("ID")

        input_df = input_df.sort("ID")

        if self.target_labels_df is not None and not self.target_labels_df.is_empty():
            input_ids = input_df.get_column("ID").to_list()
            target_ids = target_labels_df.get_column("ID").to_list()
            assert input_ids == target_ids

        return input_df, target_labels_df

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def check_samples(self) -> None:
        mode = "Test/Validation" if self.test_mode else "Train"
        self.input_storage.validate_storage(f"{mode} input storage")

        if len(self.target_labels_storage) > 0:
            self.target_labels_storage.validate_storage(f"{mode} target storage")
            check_two_storages(
                input_storage=self.input_storage,
                target_storage=self.target_labels_storage,
            )


def initialize_ids_to_keep(
    target_labels_df: pl.DataFrame | None,
    ids_to_keep: set[str] | None,
) -> set[str] | None:
    if target_labels_df is not None and target_labels_df.height > 0:
        df_ids = set(target_labels_df.get_column("ID").to_list())
        if ids_to_keep:
            return df_ids.intersection(ids_to_keep)
        return df_ids

    return ids_to_keep


def initialize_input_df(ids_to_keep: set[str] | None) -> pl.DataFrame:
    if ids_to_keep is None:
        return pl.DataFrame(schema={"ID": pl.Utf8})

    return pl.DataFrame({"ID": list(ids_to_keep)})


def add_data_to_df(
    inputs: al_input_objects_as_dict,
    input_df: pl.DataFrame,
    ids_to_keep: set[str] | None,
    data_loading_hooks: Mapping[str, InputHookOutput],
) -> pl.DataFrame:
    for input_name, input_object in inputs.items():
        input_info = input_object.input_config.input_info
        input_source = input_info.input_source
        input_inner_key = input_info.input_inner_key

        match input_object:
            case ComputedTabularInputInfo() | ComputedPredictTabularInputInfo():
                input_df = add_tabular_data_to_df(
                    df_tabular=input_object.labels.all_labels,
                    input_df=input_df,
                    ids_to_keep=ids_to_keep,
                    source_name=input_name,
                )
                # TODO: Perhaps delete the input_object.labels.{train,valid}_labels
                #       after adding the data to the input_df.
            case ComputedSequenceInputInfo() if Path(input_source).suffix == ".csv":
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                input_df = add_sequence_data_from_csv_to_df(
                    input_source=input_source,
                    input_name=input_name,
                    input_df=input_df,
                    encode_func=input_object.encode_func,
                    split_on=input_type_info.split_on,
                    ids_to_keep=ids_to_keep,
                )
            case _:
                input_df = _add_data_to_df_wrapper(
                    input_source=input_source,
                    input_name=input_name,
                    input_df=input_df,
                    ids_to_keep=ids_to_keep,
                    data_loading_hook=data_loading_hooks[input_name],
                    deeplake_input_inner_key=input_inner_key,
                )

    return input_df


def filter_df(
    input_df: pl.DataFrame,
    target_labels_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    num_samples_raw = input_df.height
    if len(input_df.columns) <= 1:
        return pl.DataFrame(schema=input_df.schema)

    input_valid_mask = _get_valid_mask(df=input_df)
    has_inputs = input_df.select([pl.col("ID"), input_valid_mask.alias("has_input")])

    filtered_df = input_df.join(has_inputs.filter(pl.col("has_input")), on="ID").drop(
        "has_input"
    )

    if target_labels_df is not None and len(target_labels_df.columns) > 1:
        target_valid_mask = _get_valid_mask(df=target_labels_df)
        has_targets = target_labels_df.select(
            [pl.col("ID"), target_valid_mask.alias("has_target")]
        )
        filtered_df = filtered_df.join(
            has_targets.filter(pl.col("has_target")), on="ID"
        ).drop("has_target")

    num_missing = num_samples_raw - filtered_df.height
    logger.info(
        "Filtered out %d samples that had no %s.",
        num_missing,
        "inputs or no target labels" if target_labels_df is not None else "inputs",
    )

    return filtered_df


def _get_valid_mask(df: pl.DataFrame) -> pl.Expr:
    numeric_cols = [
        col for col in df.columns if col != "ID" and df[col].dtype.is_numeric()
    ]
    other_cols = [
        col for col in df.columns if col != "ID" and not df[col].dtype.is_numeric()
    ]

    numeric_mask = (
        pl.any_horizontal(
            pl.all().exclude("ID", *other_cols).is_not_null()
            & ~pl.all().exclude("ID", *other_cols).is_nan()
        )
        if numeric_cols
        else pl.lit(True)
    )
    other_mask = (
        pl.any_horizontal(pl.all().exclude("ID", *numeric_cols).is_not_null())
        if other_cols
        else pl.lit(True)
    )
    return numeric_mask & other_mask


def _log_missing_samples_between_modalities(
    df: pl.DataFrame,
    input_keys: Iterable[str],
) -> None:
    """
    We have the `if col == key or col.startswith(f"{key}__"):` condition there
    as for e.g. genotype inputs, the column name is just the key itself, but for
    tabular inputs, we have the key as a prefix to the column name, e.g.
    "tabular_input__column1".
    """
    missing_counts = dict.fromkeys(input_keys, 0)
    missing_ids: dict[str, list[str]] = {k: [] for k in input_keys}
    any_missing = False
    no_samples = df.height

    for key in input_keys:
        key_cols = []
        for col in df.columns:
            if col == key or col.startswith(f"{key}__"):
                key_cols.append(col)

        if not key_cols:
            missing_counts[key] = no_samples
            missing_ids[key] = df.get_column("ID").to_list()
            any_missing = True
            continue

        missing_mask = df.select(
            ["ID", pl.all_horizontal(pl.col(key_cols).is_null()).alias("all_null")]
        ).filter(pl.col("all_null"))

        if missing_mask.height > 0:
            any_missing = True
            missing_counts[key] = missing_mask.height
            missing_ids[key] = missing_mask.get_column("ID").to_list()

    message = (
        f"Using total of {no_samples} samples with following counts per "
        f"modality (note missing tabular modalities have been imputed already):\n"
    )

    for key in input_keys:
        cur_missing = missing_counts[key]
        cur_present = no_samples - cur_missing
        cur_missing_ids = reprlib.repr(missing_ids[key])
        cur_str = (
            f"Available {key}: {cur_present} "
            f"(missing: {cur_missing}, missing IDs: {cur_missing_ids})"
        )

        cur_str += "\n"
        message += cur_str

    logger.info(message.rstrip())

    if any_missing:
        warning_message = (
            "There were missing inputs in samples for some modalities. "
            "Please review the info log above for detailed counts and IDs. "
            "If this is expected, ignore this message. If not, possible "
            "causes are (a) different inputs having different sample IDs "
            "present, (b) sample IDs between inputs / targets are not matching, "
            "or (c) something else."
        )
        logger.warning(warning_message)


def _target_labels_hook_load_wrapper(
    target_labels_df: pl.DataFrame,
    output_data_loading_hooks: Mapping[str, HookOutput],
) -> pl.DataFrame:
    """
    We have this slightly roundabout way of using a None placeholder as polars
    raises an error if we try to convert a list that is a mix of None and other
    types (e.g. arrays) to a polars Series directly.
    """
    for column in target_labels_df.columns:
        if "__" not in column or column == "ID":
            continue

        output_name, inner_name = column.split("__", 1)
        if output_name not in output_data_loading_hooks:
            continue

        hook = output_data_loading_hooks[output_name]
        processed_values = []
        none_indices = []

        return_dtype = hook.return_dtype
        none_placeholder: Any
        if isinstance(return_dtype, pl.List):
            none_placeholder = []
        elif return_dtype is not None and return_dtype.is_numeric():
            none_placeholder = np.nan
        elif isinstance(return_dtype, pl.Array):
            none_placeholder = np.full(return_dtype.shape, np.nan)
        else:
            none_placeholder = None

        for i, x in enumerate(target_labels_df.get_column(name=column)):
            if x is None:
                none_indices.append(i)
                processed_values.append(none_placeholder)
                continue

            hook_input = {inner_name: x}
            hook_output = hook.hook_callable(hook_input)
            parsed_value = hook_output[inner_name]
            processed_values.append(parsed_value)

        series = pl.Series(
            name=column,
            values=processed_values,
            dtype=hook.return_dtype,
        )

        # replace placeholders with None / null
        target_labels_df = target_labels_df.with_columns(
            [
                pl.when(pl.arange(0, len(target_labels_df)).is_in(none_indices))
                .then(pl.lit(None))
                .otherwise(pl.lit(series))
                .alias(column)
            ]
        )

    return target_labels_df


def _add_data_to_df_wrapper(
    input_source: str,
    input_name: str,
    input_df: pl.DataFrame,
    ids_to_keep: None | set[str],
    data_loading_hook: InputHookOutput,
    deeplake_input_inner_key: str | None = None,
) -> pl.DataFrame:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_input_inner_key is not None
        return deeplake_ops.add_deeplake_data_to_df(
            input_source=input_source,
            input_name=input_name,
            input_df=input_df,
            ids_to_keep=ids_to_keep,
            deeplake_input_inner_key=deeplake_input_inner_key,
            data_loading_hook=data_loading_hook,
        )
    return _add_file_data_to_df(
        input_source=input_source,
        input_df=input_df,
        ids_to_keep=ids_to_keep,
        data_loading_hook=data_loading_hook,
        input_name=input_name,
    )


def _add_file_data_to_df(
    input_source: str,
    input_name: str,
    input_df: pl.DataFrame,
    ids_to_keep: None | set[str],
    data_loading_hook: InputHookOutput,
) -> pl.DataFrame:
    file_data_iterator = get_file_sample_id_iterator_basic(
        data_source=input_source,
        ids_to_keep=ids_to_keep,
    )

    ids = []
    column_arrays: dict[str, Any] = {}
    hook_callable = data_loading_hook.hook_callable
    hook_dtype = data_loading_hook.return_dtype
    is_list_dype = isinstance(hook_dtype, pl.List)

    for sample_id, file in tqdm(file_data_iterator, desc=input_name):
        sample_data = hook_callable(file)

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

    if not ids:
        return input_df

    df_dict = {"ID": pl.Series(name="ID", values=ids, dtype=pl.Utf8)}

    for col_name, values in column_arrays.items():
        df_dict[col_name] = pl.Series(name=col_name, values=values, dtype=hook_dtype)

    processed_df = pl.DataFrame(df_dict)

    if input_df.height == 0:
        return processed_df
    return input_df.join(processed_df, on="ID", how="full", coalesce=True)


class DiskDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.target_labels_df.is_empty():
            self.init_label_attributes()

        sample_dfs: tuple[pl.DataFrame, pl.DataFrame] = self.set_up_dfs()
        input_df = sample_dfs[0]
        target_labels_df = sample_dfs[1]

        self.input_storage.from_polars(df=input_df)
        self.target_labels_storage.from_polars(df=target_labels_df)

        self.check_samples()

        self.column_map_inputs = _build_column_map(columns=input_df.columns)
        self.column_map_targets = _build_column_map(columns=target_labels_df.columns)

    def __getitem__(self, index: int) -> al_getitem_return:
        input_row = self.input_storage.get_row(idx=index)
        sample_id = input_row["ID"]

        inputs = process_row_values(row=input_row, column_map=self.column_map_inputs)

        inputs_prepared = prepare_inputs_disk(
            inputs=inputs,
            inputs_objects=self.inputs,
            test_mode=self.test_mode,
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs_values=inputs_prepared,
            inputs_objects=self.inputs,
        )

        targets_final = {}
        if len(self.target_labels_storage) > 0:
            target_row = self.target_labels_storage.get_row(idx=index)
            target_labels = process_row_values(
                row=target_row,
                column_map=self.column_map_targets,
            )

            targets_prepared = prepare_outputs_disk(
                outputs=target_labels,
                output_objects=self.outputs,
                test_mode=self.test_mode,
            )

            targets_final = impute_missing_output_modalities_wrapper(
                outputs_values=targets_prepared,
                output_objects=self.outputs,
            )

        return inputs_final, targets_final, sample_id

    def __len__(self):
        return len(self.input_storage)


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.target_labels_df.is_empty():
            self.init_label_attributes()

        input_data_loading_hooks = get_input_data_loading_hooks(inputs=self.inputs)
        output_data_loading_hooks = get_output_data_loading_hooks(outputs=self.outputs)

        sample_dfs: tuple[pl.DataFrame, pl.DataFrame] = self.set_up_dfs(
            input_data_loading_hooks=input_data_loading_hooks,
            output_data_loading_hooks=output_data_loading_hooks,
        )
        input_df = sample_dfs[0]
        target_labels_df = sample_dfs[1]

        self.input_storage.from_polars(df=input_df)
        self.target_labels_storage.from_polars(df=target_labels_df)

        self.check_samples()

        self.column_map_inputs = _build_column_map(columns=input_df.columns)
        self.column_map_targets = _build_column_map(columns=target_labels_df.columns)

    def __getitem__(self, index: int) -> al_getitem_return:
        input_row = self.input_storage.get_row(idx=index)
        sample_id = input_row["ID"]

        inputs = process_row_values(row=input_row, column_map=self.column_map_inputs)

        inputs_prepared = prepare_inputs_memory(
            inputs=inputs,
            inputs_objects=self.inputs,
            test_mode=self.test_mode,
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs_values=inputs_prepared,
            inputs_objects=self.inputs,
        )

        targets_final = {}
        if len(self.target_labels_storage) > 0:
            target_row = self.target_labels_storage.get_row(idx=index)
            target_labels = process_row_values(
                row=target_row,
                column_map=self.column_map_targets,
            )

            targets_prepared = prepare_outputs_memory(
                outputs=target_labels,
                output_objects=self.outputs,
                test_mode=self.test_mode,
            )

            targets_final = impute_missing_output_modalities_wrapper(
                outputs_values=targets_prepared,
                output_objects=self.outputs,
            )

        return inputs_final, targets_final, sample_id

    def __len__(self):
        return len(self.input_storage)


def _build_column_map(columns: list[str]) -> dict[str, tuple[str, str | None]]:
    column_map: dict[str, Any] = {}
    for col in columns:
        if col == "ID":
            continue
        if "__" in col:
            main_key, sub_key = col.split("__", 1)
            column_map[col] = (main_key, sub_key)
        else:
            column_map[col] = (col, None)
    return column_map


def convert_value(value: Any) -> Any:
    """
    Note that this is only needed as when we store arrays in the Array() polars
    dtype, it always returns a list when we get the cell value. If it returned
    a numpy array, this would not be needed.
    """
    return np.array(value) if isinstance(value, list) else value


def process_row_values(
    row: dict,
    column_map: dict[str, tuple[str, str | None]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for col, (main_key, sub_key) in column_map.items():
        value = row[col]
        if is_null_value(value=value):
            continue

        value = convert_value(value=value)
        if sub_key:
            if main_key not in result:
                result[main_key] = {}
            result[main_key][sub_key] = value
        else:
            result[main_key] = value

    return result
