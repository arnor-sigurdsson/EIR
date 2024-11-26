import reprlib
from collections import defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
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
al_dataset_types = Type[al_datasets]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_label_dict_target = dict[str, dict[str, int | float | torch.Tensor]]
al_inputs = dict[str, torch.Tensor] | dict[str, Any]
al_getitem_return = tuple[dict[str, torch.Tensor], al_sample_label_dict_target, str]


def set_up_datasets_from_configs(
    configs: config.Configs,
    target_labels: "MergedTargetLabels",
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    train_ids_to_keep: Optional[Sequence[str]] = None,
    valid_ids_to_keep: Optional[Sequence[str]] = None,
    websocket_url: Optional[str] = None,
) -> Tuple[al_datasets, al_local_datasets]:

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
        assert isinstance(train_dataset, (MemoryDataset, DiskDataset))
        assert isinstance(valid_dataset, (MemoryDataset, DiskDataset))
        _check_valid_and_train_datasets(
            train_dataset=train_dataset, valid_dataset=valid_dataset
        )

    assert isinstance(valid_dataset, (MemoryDataset, DiskDataset))
    return train_dataset, valid_dataset


def construct_default_dataset_kwargs_from_cl_args(
    target_labels_df: Optional[al_target_labels],
    inputs: al_input_objects_as_dict,
    outputs: "al_output_objects_as_dict",
    test_mode: bool,
    missing_ids_per_output: "MissingTargetsInfo",
    ids_to_keep: Optional[Sequence[str]] = None,
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
    train_dataset: al_local_datasets, valid_dataset: al_local_datasets
) -> None:
    if train_dataset.input_df.height < valid_dataset.input_df.height:
        logger.warning(
            "Size of training dataset (size: %d) is smaller than validation dataset ("
            "size: %d). Generally it is the opposite, but if this is intended please "
            "ignore this message.",
            train_dataset.input_df.height,
            valid_dataset.input_df.height,
        )

    train_ids = set(train_dataset.input_df.get_column("ID").to_list())
    valid_ids = set(valid_dataset.input_df.get_column("ID").to_list())

    if not train_ids.isdisjoint(valid_ids):
        overlapping_ids = train_ids.intersection(valid_ids)
        raise AssertionError(
            f"Found overlapping IDs between training and "
            f"validation sets: {overlapping_ids}"
        )


class DatasetBase(Dataset):
    def __init__(
        self,
        inputs: al_input_objects_as_dict,
        outputs: "al_output_objects_as_dict",
        test_mode: bool,
        missing_ids_per_output: "MissingTargetsInfo",
        target_labels_df: Optional[al_target_labels] = None,
        ids_to_keep: Optional[Set[str]] = None,
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

        self.missing_ids_per_output = missing_ids_per_output
        self.ids_to_keep = set(ids_to_keep) if ids_to_keep else None

    def init_label_attributes(self):
        if not self.outputs:
            raise ValueError("Please specify label column name.")

    def set_up_dfs(
        self,
        input_data_loading_hooks: Optional[Mapping[str, InputHookOutput]] = None,
        output_data_loading_hooks: Optional[Mapping[str, HookOutput]] = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        We do an extra filtering step at the end to account for the situation where
        we have a target label file with more samples than there are any inputs
        available for. This is quite likely if we have e.g. pre-split data into
        train/val and test folders.
        """

        def _identity(sample_data: Any) -> Any:
            return sample_data

        mode_str = "evaluation/test" if self.test_mode else "train"
        logger.debug("Setting up dataset in %s mode.", mode_str)

        if not output_data_loading_hooks:
            output_data_loading_hooks = defaultdict(
                lambda: HookOutput(
                    hook_callable=_identity,
                    return_dtype=None,
                )
            )

        if not self.target_labels_df.is_empty():
            self.target_labels_df = _target_labels_hook_load_wrapper(
                target_labels_df=self.target_labels_df,
                output_data_loading_hooks=output_data_loading_hooks,
            )

        if not input_data_loading_hooks:
            input_data_loading_hooks = defaultdict(
                lambda: InputHookOutput(
                    hook_callable=_identity,
                    return_dtype=None,
                )
            )

        ids_to_keep = initialize_ids_to_keep(
            target_labels_df=self.target_labels_df,
            ids_to_keep=self.ids_to_keep,
        )

        input_df = initialize_input_df(ids_to_keep=ids_to_keep)

        input_df = add_data_to_df(
            inputs=self.inputs,
            input_df=input_df,
            ids_to_keep=ids_to_keep,
            data_loading_hooks=input_data_loading_hooks,
        )

        filtered_df = filter_df(
            input_df=input_df,
            target_labels_df=self.target_labels_df,
        )

        _log_missing_samples_between_modalities(
            df=filtered_df, input_keys=self.inputs.keys()
        )

        if self.target_labels_df is not None and not self.target_labels_df.is_empty():
            target_labels_df = self.target_labels_df.filter(
                pl.col("ID").is_in(filtered_df.get_column("ID").to_list())
            )
            self.target_labels_df = target_labels_df
            self.target_labels_df = self.target_labels_df.sort("ID")

        filtered_df = filtered_df.sort("ID")

        return filtered_df, self.target_labels_df

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index: int):
        raise NotImplementedError()

    def check_samples(self) -> None:
        if self.input_df.height == 0:
            raise ValueError(
                "Expected to have at least one sample, but got empty DataFrame. "
                "Possibly there is a mismatch between input IDs and target IDs."
            )

        if self.input_df.get_column("ID").is_null().any():
            missing_ids = (
                self.input_df.filter(pl.col("ID").is_null()).select("ID").row(0)
            )
            raise ValueError(
                f"Expected all observations to have a sample ID associated "
                f"with them, but got rows with null IDs at indices: {missing_ids}"
            )

        input_cols = [col for col in self.input_df.columns if col != "ID"]
        if input_cols:
            empty_inputs = self.input_df.with_columns(
                [pl.all_horizontal(pl.col(input_cols).is_null()).alias("all_null")]
            ).filter(pl.col("all_null"))

            if empty_inputs.height > 0:
                empty_ids = empty_inputs.get_column("ID").to_list()
                raise ValueError(
                    f"Expected all observations to have at least one input value "
                    f"associated with them, but got empty inputs for IDs: {empty_ids}"
                )

        if not self.target_labels_df.is_empty():
            target_cols = [col for col in self.target_labels_df.columns if col != "ID"]
            if target_cols:
                empty_targets = self.target_labels_df.with_columns(
                    [pl.all_horizontal(pl.col(target_cols).is_null()).alias("all_null")]
                ).filter(pl.col("all_null"))

                if empty_targets.height > 0:
                    empty_ids = empty_targets.get_column("ID").to_list()
                    raise ValueError(
                        f"Expected all observations to have at least one target label "
                        f"associated with them, but got empty targets for "
                        f"IDs: {empty_ids}"
                    )


def initialize_ids_to_keep(
    target_labels_df: Optional[pl.DataFrame],
    ids_to_keep: Optional[set[str]],
) -> Optional[set[str]]:

    if target_labels_df is not None and target_labels_df.height > 0:
        df_ids = set(target_labels_df.get_column("ID").to_list())
        if ids_to_keep:
            return df_ids.intersection(ids_to_keep)
        return df_ids

    return ids_to_keep


def initialize_input_df(ids_to_keep: Optional[set[str]]) -> pl.DataFrame:
    if ids_to_keep is None:
        return pl.DataFrame(schema={"ID": pl.Utf8})

    return pl.DataFrame({"ID": list(ids_to_keep)})


def add_data_to_df(
    inputs: al_input_objects_as_dict,
    input_df: pl.DataFrame,
    ids_to_keep: Optional[set[str]],
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
    input_df: pl.DataFrame, target_labels_df: Optional[pl.DataFrame] = None
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
    missing_counts = {k: 0 for k in input_keys}
    missing_ids: dict[str, list[str]] = {k: [] for k in input_keys}
    any_missing = False
    no_samples = df.height

    for key in input_keys:
        key_cols = [col for col in df.columns if col == key]
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
    for column in target_labels_df.columns:
        if "__" not in column or column == "ID":
            continue

        output_name, inner_name = column.split("__", 1)
        if output_name not in output_data_loading_hooks:
            continue

        hook = output_data_loading_hooks[output_name]
        processed_values = [
            hook.hook_callable({inner_name: x})[inner_name]
            for x in target_labels_df.get_column(column)
        ]

        target_labels_df = target_labels_df.with_columns(
            [
                pl.Series(
                    name=column,
                    values=processed_values,
                    dtype=hook.return_dtype,
                )
            ]
        )

    return target_labels_df


def _add_data_to_df_wrapper(
    input_source: str,
    input_name: str,
    input_df: pl.DataFrame,
    ids_to_keep: Union[None, Set[str]],
    data_loading_hook: InputHookOutput,
    deeplake_input_inner_key: Optional[str] = None,
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
    else:
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
    ids_to_keep: Union[None, Set[str]],
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

    for sample_id, file in tqdm(file_data_iterator, desc=input_name):
        sample_data = hook_callable(file)

        if isinstance(sample_data, Path):
            sample_data = str(sample_data)

        if isinstance(sample_data, dict):
            for key in sample_data.keys():
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
    else:
        return input_df.join(processed_df, on="ID", how="full", coalesce=True)


class DiskDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.target_labels_df.is_empty():
            self.init_label_attributes()

        sample_dfs: tuple[pl.DataFrame, pl.DataFrame] = self.set_up_dfs()
        self.input_df = sample_dfs[0]
        self.target_labels_df = sample_dfs[1]
        self.check_samples()

    def __getitem__(self, index: int) -> al_getitem_return:
        input_row = self.input_df.row(index, named=True)
        sample_id = input_row["ID"]

        inputs = process_row_values(row=input_row, columns=self.input_df.columns)

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
        if not self.target_labels_df.is_empty():
            target_row = self.target_labels_df.row(index, named=True)
            target_labels = process_row_values(
                row=target_row, columns=self.target_labels_df.columns
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
        return self.input_df.height


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
        self.input_df = sample_dfs[0]
        self.target_labels_df = sample_dfs[1]
        self.check_samples()

    def __getitem__(self, index: int) -> al_getitem_return:
        input_row = self.input_df.row(index, named=True)
        sample_id = input_row["ID"]

        inputs = process_row_values(row=input_row, columns=self.input_df.columns)

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
        if not self.target_labels_df.is_empty():
            target_row = self.target_labels_df.row(index, named=True)
            target_labels = process_row_values(
                row=target_row, columns=self.target_labels_df.columns
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
        return self.input_df.height


def convert_value(value: Any) -> Any:
    """
    Note that this is only needed as when we store arrays in the Array() polars
    dtype, it always returns a list when we get the cell value. If it returned
    a numpy array, this would not be needed.
    """
    return np.array(value) if isinstance(value, list) else value


def process_row_values(
    row: dict,
    columns: list[str],
    exclude_id: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for col in columns:
        if exclude_id and col == "ID":
            continue

        value = row[col]
        if value is None:
            continue

        if "__" in col:
            main_key, sub_key = col.split("__", 1)
            result.setdefault(main_key, {})[sub_key] = convert_value(value=value)
        else:
            result[col] = convert_value(value=value)

    return result
