import reprlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Union, List, Callable, Any, Sequence

import joblib
import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from eir.data_load.common_ops import ColumnOperation
from eir.setup.schemas import InputConfig

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_all_column_ops = Union[None, Dict[str, Tuple[ColumnOperation, ...]]]
al_train_val_dfs = Tuple[pd.DataFrame, pd.DataFrame]

# e.g. 'Asia' or '5' for categorical or 1.511 for continuous
al_label_values_raw = Union[float, int]
al_sample_labels_raw = Dict[str, al_label_values_raw]
al_label_dict = Dict[str, al_sample_labels_raw]
al_target_columns = Dict[str, List[str]]
al_label_transformers_object = Union[StandardScaler, LabelEncoder]
al_label_transformers = Dict[str, al_label_transformers_object]
al_pd_dtypes = Union[np.ndarray, pd.CategoricalDtype]


@dataclass
class Labels:
    train_labels: al_label_dict
    valid_labels: al_label_dict
    label_transformers: al_label_transformers

    @property
    def all_labels(self):
        return {**self.train_labels, **self.valid_labels}


@dataclass
class TabularFileInfo:
    file_path: Path
    con_columns: Sequence[str]
    cat_columns: Sequence[str]
    parsing_chunk_size: Union[None, int] = None


def set_up_train_and_valid_tabular_data(
    tabular_file_info: TabularFileInfo,
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> Labels:
    """
    Splits and does split based processing (e.g. scaling validation set with training
    set for regression) on the labels.
    """

    if len(tabular_file_info.con_columns) + len(tabular_file_info.cat_columns) < 1:
        raise ValueError(f"No label columns specified in {tabular_file_info}.")

    parse_wrapper = get_label_parsing_wrapper(
        label_parsing_chunk_size=tabular_file_info.parsing_chunk_size
    )
    ids_to_keep = list(train_ids) + list(valid_ids)
    df_labels = parse_wrapper(
        label_file_tabular_info=tabular_file_info,
        ids_to_keep=ids_to_keep,
        custom_label_ops=custom_label_ops,
    )

    df_labels_train, df_labels_valid = _split_df_by_ids(
        df=df_labels, train_ids=train_ids, valid_ids=valid_ids
    )

    df_labels_train, df_labels_valid, label_transformers = _process_train_and_label_dfs(
        tabular_info=tabular_file_info,
        df_labels_train=df_labels_train,
        df_labels_valid=df_labels_valid,
    )

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")

    labels_data_object = Labels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
    )

    return labels_data_object


def _get_fit_label_transformers(
    df_labels: pd.DataFrame, label_columns: al_target_columns
) -> al_label_transformers:

    label_transformers = {}

    for column_type in label_columns:
        target_columns_of_cur_type = label_columns[column_type]

        if target_columns_of_cur_type:
            logger.debug(
                "Fitting transformers on %s label columns %s",
                column_type,
                target_columns_of_cur_type,
            )

        for cur_target_column in target_columns_of_cur_type:
            cur_series = df_labels[cur_target_column]

            cur_target_transformer = _get_transformer(column_type=column_type)
            cur_target_transformer_fit = _fit_transformer_on_label_column(
                column_series=cur_series, transformer=cur_target_transformer
            )

            label_transformers[cur_target_column] = cur_target_transformer_fit

    return label_transformers


def _get_transformer(column_type):
    if column_type in ("con", "extra_con"):
        return StandardScaler()
    elif column_type == "cat":
        return LabelEncoder()

    raise ValueError()


def _fit_transformer_on_label_column(
    column_series: pd.Series,
    transformer: al_label_transformers_object,
) -> al_label_transformers_object:
    """
    TODO:   Possibly use the categorical codes here directly in the fit call. Then we
            don't do another pass over all values, and we ensure that the encoder
            encounters 'NA'.

    If we have a label file, and a column only consists of integers, and we are reading
    it as a categorical columns. Then, the values of that categorical column are
    going to be INT. This is still a categorical column, the key element is, how
    do we change the values of a categorical column?
    """

    if isinstance(column_series.dtype, pd.CategoricalDtype):
        series_values = column_series.cat.categories.values
    else:
        series_values = column_series.values

    series_values_streamlined = _streamline_values_for_transformers(
        transformer=transformer, values=series_values
    )

    transformer.fit(series_values_streamlined)

    return transformer


def _streamline_values_for_transformers(
    transformer: al_label_transformers_object, values: np.ndarray
) -> np.ndarray:
    """
    LabelEncoder() expects a 1D array, whereas StandardScaler() expects a 2D one.
    """

    if isinstance(transformer, StandardScaler):
        values_reshaped = values.reshape(-1, 1)
        return values_reshaped

    return values


def transform_label_df(
    df_labels: pd.DataFrame, label_transformers: al_label_transformers
) -> pd.DataFrame:

    df_labels_copy = df_labels.copy()

    for column_name, transformer_instance in label_transformers.items():

        series_values = df_labels_copy[column_name].values
        series_values_streamlined = _streamline_values_for_transformers(
            transformer=transformer_instance, values=series_values
        )

        transform_func = transformer_instance.transform
        df_labels_copy[column_name] = transform_func(series_values_streamlined)

    return df_labels_copy


def get_label_parsing_wrapper(
    label_parsing_chunk_size: Union[None, int]
) -> Callable[[TabularFileInfo, Sequence[str], al_all_column_ops], pd.DataFrame]:
    if label_parsing_chunk_size is None:
        return label_df_parse_wrapper
    return chunked_label_df_parse_wrapper


def label_df_parse_wrapper(
    label_file_tabular_info: TabularFileInfo,
    ids_to_keep: Union[None, Sequence[str]],
    custom_label_ops: al_all_column_ops = None,
) -> pd.DataFrame:
    """
    Note: Here the genomic arrays are the dominant factor
    in determining whether we drop or not.

    If we start doing multimodal data, we no longer can say that the genomic test set
    is the test set, unless we are careful about syncing all data sources (e.g. images,
    other omics, etc.).
    """

    column_ops = {}
    if custom_label_ops is not None:
        column_ops = custom_label_ops

    all_label_columns, dtypes = _get_all_label_columns_and_dtypes(
        cat_columns=label_file_tabular_info.cat_columns,
        con_columns=label_file_tabular_info.con_columns,
        column_ops=column_ops,
    )

    df_labels = _load_label_df(
        label_fpath=label_file_tabular_info.file_path,
        columns=all_label_columns,
        custom_label_ops=column_ops,
        dtypes=dtypes,
    )

    df_labels_filtered = _filter_ids_from_label_df(
        df_labels=df_labels, ids_to_keep=ids_to_keep
    )

    df_labels_column_op_parsed = _apply_column_operations_to_df(
        df=df_labels_filtered,
        operations_dict=column_ops,
        label_columns=all_label_columns,
    )

    supplied_columns = get_passed_in_columns(tabular_info=label_file_tabular_info)
    df_column_filtered = _drop_not_needed_label_columns(
        df=df_labels_column_op_parsed, needed_label_columns=supplied_columns
    )

    df_cat_str = ensure_categorical_columns_and_format(df=df_column_filtered)

    df_final = _check_parsed_label_df(
        df_labels=df_cat_str, supplied_label_columns=supplied_columns
    )

    return df_final


def chunked_label_df_parse_wrapper(
    label_file_tabular_info: TabularFileInfo,
    ids_to_keep: Union[None, Sequence[str]],
    custom_label_ops: al_all_column_ops = None,
) -> pd.DataFrame:
    """
    Note that we have to recast the dtypes to preserve the categorical dtypes
    after concatenation, otherwise they will be cast to object (i.e. str).
    See: https://github.com/pandas-dev/pandas/issues/25412

    We can be a bit risky in the cast (i.e. only casting columns that are in the DF,
    even if they are specified in the dtypes), because _check_parsed_label_df will
    raise an error if there is any mismatch.

    If this starts to lead to memory issues, we can try using pandas union_categoricals.

    Note that we continue in the case that we have empty dfs after filtering for IDs.
    This is to avoid errors when applying column operations to empty dfs, as that case
    might not be considered in the column operation functions. The case for supporting
    that in column operations might be some manual creation of a df assuming that it
    can be empty, but I think its a very rare use case, so not supported for now.
    """

    column_ops = {}
    if custom_label_ops is not None:
        column_ops = custom_label_ops

    label_columns, dtypes = _get_all_label_columns_and_dtypes(
        cat_columns=label_file_tabular_info.cat_columns,
        con_columns=label_file_tabular_info.con_columns,
        column_ops=column_ops,
    )

    chunk_generator = _get_label_df_chunk_generator(
        chunk_size=label_file_tabular_info.parsing_chunk_size,
        label_fpath=label_file_tabular_info.file_path,
        columns=label_columns,
        custom_label_ops=column_ops,
        dtypes=dtypes,
    )

    processed_chunks = []
    supplied_columns = get_passed_in_columns(tabular_info=label_file_tabular_info)

    for chunk in chunk_generator:

        df_labels_filtered = _filter_ids_from_label_df(
            df_labels=chunk, ids_to_keep=ids_to_keep
        )

        if len(df_labels_filtered) == 0:
            continue

        df_labels_parsed = _apply_column_operations_to_df(
            df=df_labels_filtered,
            operations_dict=column_ops,
            label_columns=label_columns,
        )

        if len(df_labels_parsed) == 0:
            continue

        df_column_filtered = _drop_not_needed_label_columns(
            df=df_labels_parsed, needed_label_columns=supplied_columns
        )

        processed_chunks.append(df_column_filtered)

    df_concat = pd.concat(processed_chunks)

    # TODO: Remove cast after pandas fixes issue, see docstring
    dtype_cast = {k: v for k, v in dtypes.items() if k in df_concat.columns}
    df_concat = df_concat.astype(dtype=dtype_cast)

    df_cat_str = ensure_categorical_columns_and_format(df=df_concat)

    df_final = _check_parsed_label_df(
        df_labels=df_cat_str, supplied_label_columns=supplied_columns
    )

    return df_final


def _get_label_df_chunk_generator(
    chunk_size: int,
    label_fpath: Path,
    columns: Sequence[str],
    custom_label_ops: al_all_column_ops,
    dtypes: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    We accept only loading the available columns at this point because the passed
    in columns might be forward referenced, meaning that they might be created
    by the custom library.
    """

    dtypes = _ensure_id_str_dtype(dtypes=dtypes)

    logger.debug("Reading in labelfile: %s in chunks of %d.", label_fpath, chunk_size)

    columns_with_id_col = ["ID"] + list(columns)
    available_columns = _get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=columns_with_id_col,
        custom_label_ops=custom_label_ops,
    )

    chunks_processed = 0
    for chunk in pd.read_csv(
        label_fpath,
        usecols=available_columns,
        dtype=dtypes,
        low_memory=False,
        chunksize=chunk_size,
    ):
        logger.debug(
            "Processsed %d rows so far in %d chunks.",
            chunk_size * chunks_processed,
            chunks_processed,
        )
        chunks_processed += 1

        chunk = chunk.set_index("ID")
        yield chunk


def ensure_categorical_columns_and_format(df: pd.DataFrame) -> pd.DataFrame:

    df_copy = df.copy()
    for column in df_copy.columns:

        if df_copy[column].dtype == object:
            logger.info(
                "Implicitly casting %s of type 'object' to categorical dtype.", column
            )
            df_copy[column] = df_copy[column].astype(pd.CategoricalDtype())

        if isinstance(df_copy[column].dtype, pd.CategoricalDtype):

            mapping = {k: str(k) for k in df_copy[column].cat.categories}
            df_copy[column] = df_copy[column].cat.rename_categories(mapping)

    return df_copy


def gather_all_ids_from_all_inputs(
    input_configs: Sequence[InputConfig],
) -> Tuple[str, ...]:
    ids = set()
    for input_config in input_configs:

        cur_source = Path(input_config.input_info.input_source)
        cur_type = input_config.input_info.input_type
        if cur_type in ["omics", "sequence"]:
            cur_ids = gather_ids_from_data_source(data_source=cur_source)

        elif cur_type == "tabular":
            cur_ids = gather_ids_from_tabular_file(
                file_path=Path(input_config.input_info.input_source)
            )
        else:
            raise NotImplementedError(
                f"ID gather not implemented for type {cur_type} (source: {cur_source})"
            )

        cur_ids_set = set(cur_ids)
        ids.update(cur_ids_set)

    return tuple(ids)


def gather_ids_from_data_source(data_source: Path, validate: bool = True):
    iterator = get_array_path_iterator(data_source=data_source, validate=validate)
    logger.debug("Gathering IDs from %s.", data_source)
    all_ids = tuple(i.stem for i in tqdm(iterator, desc="Progress"))

    return all_ids


def gather_ids_from_tabular_file(file_path: Path):
    df = pd.read_csv(file_path, usecols=["ID"])
    all_ids = tuple(df["ID"].astype(str))

    return all_ids


def get_array_path_iterator(data_source: Path, validate: bool = True):
    def _file_iterator(file_path: Path):
        with open(str(file_path), "r") as infile:
            for line in infile:
                path = Path(line.strip())

                if validate:
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Could not find array {path} listed in {data_source}."
                        )

                yield path

    if data_source.is_dir():
        return data_source.rglob("*")
    elif data_source.is_file():
        return _file_iterator(file_path=data_source)

    if not data_source.exists():
        raise FileNotFoundError("Could not find data source %s.", data_source)
    raise ValueError(
        "Data source %s is neither recognized as a file nor folder.", data_source
    )


def _get_all_label_columns_and_dtypes(
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    column_ops: al_all_column_ops,
) -> Tuple[Sequence[str], Dict[str, Any]]:

    supplied_label_columns = _get_column_dtypes(
        cat_columns=cat_columns, con_columns=con_columns
    )

    label_columns = list(supplied_label_columns.keys())
    extra_label_parsing_cols = _get_extra_columns(
        label_columns=label_columns, all_column_ops=column_ops
    )

    all_cols_and_dtypes = {**supplied_label_columns, **extra_label_parsing_cols}
    all_cols = tuple(all_cols_and_dtypes.keys())
    supplied_dtypes = {k: v for k, v in all_cols_and_dtypes.items() if v is not None}

    return all_cols, supplied_dtypes


def _get_column_dtypes(
    cat_columns: Sequence[str], con_columns: Sequence[str]
) -> Dict[str, Any]:
    dtypes = {}

    for cat_column in cat_columns:
        dtypes[cat_column] = pd.CategoricalDtype()
    for con_column in con_columns:
        dtypes[con_column] = float

    return dtypes


def _get_extra_columns(
    label_columns: List[str], all_column_ops: al_all_column_ops
) -> Dict[str, Any]:
    """
    We use this to grab extra columns needed for the current run, as specified in the
    COLUMN_OPS, where the keys are the label columns. That is, "for running with these
    specific label columns, what other columns do we need to grab", as specified
    by the extra_columns_deps attribute of each column operation.

    :param label_columns: The target columns we are modelling on.
    :param all_column_ops: The ledger of all column ops to be done for each target
    column.
    :returns A dict of extra columns and their dtypes if a dtype is specified, otherwise
    None as the column dtype.
    """

    extra_columns = {}
    for column_name in label_columns + ["base", "post"]:

        if column_name in all_column_ops:
            cur_column_operations_sequence = all_column_ops.get(column_name)

            for cur_column_operation_object in cur_column_operations_sequence:
                cur_extra_columns = cur_column_operation_object.extra_columns_deps
                cur_column_dtypes = cur_column_operation_object.column_dtypes

                if cur_column_dtypes is None:
                    cur_column_dtypes = {}

                for cur_extra_column in cur_extra_columns:
                    cur_dtype = cur_column_dtypes.get(cur_extra_column, None)
                    extra_columns[cur_extra_column] = cur_dtype

    return extra_columns


def _load_label_df(
    label_fpath: Path,
    columns: Sequence[str],
    custom_label_ops: al_all_column_ops,
    dtypes: Union[Dict[str, Any], None] = None,
) -> pd.DataFrame:
    """
    We accept only loading the available columns at this point because the passed
    in columns might be forward referenced, meaning that they might be created
    by the custom library.
    """

    dtypes = _ensure_id_str_dtype(dtypes=dtypes)

    logger.debug("Reading in labelfile: %s. ID is read as str dtype.", label_fpath)

    columns_with_id_col = ["ID"] + list(columns)
    available_columns = _get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=columns_with_id_col,
        custom_label_ops=custom_label_ops,
    )

    df_labels = pd.read_csv(
        filepath_or_buffer=label_fpath,
        usecols=available_columns,
        dtype=dtypes,
        low_memory=False,
    )

    df_labels = df_labels.set_index("ID")

    return df_labels


def _ensure_id_str_dtype(dtypes: Union[Dict[str, Any], None]) -> Dict[str, Any]:
    if dtypes is None:
        dtypes = {"ID": str}
    elif "ID" not in dtypes.keys():
        dtypes["ID"] = str

    return dtypes


def _get_currently_available_columns(
    label_fpath: Path,
    requested_columns: List[str],
    custom_label_ops: al_all_column_ops,
) -> List[str]:
    """
    If custom label operations are specified, the requested columns could be forward
    references. Hence we should not raise an error if there is a possibility of them
    being created at runtime.

    However if no custom operations are specified, we should fail here if columns
    are not found.
    """

    label_file_columns_set = set(pd.read_csv(label_fpath, dtype={"ID": str}, nrows=0))

    requested_columns_set = set(requested_columns)

    if custom_label_ops is None:
        missing_columns = requested_columns_set - label_file_columns_set
        if missing_columns:
            raise ValueError(
                f"No custom library specified and could not find columns "
                f"{missing_columns} in {label_fpath}."
            )

    available_columns = requested_columns_set.intersection(label_file_columns_set)

    return list(available_columns)


def _filter_ids_from_label_df(
    df_labels: pd.DataFrame, ids_to_keep: Union[None, Tuple[str, ...]] = None
) -> pd.DataFrame:

    if not ids_to_keep:
        return df_labels

    no_labels = df_labels.shape[0]

    mask = df_labels.index.isin(ids_to_keep)
    df_filtered = df_labels.loc[mask, :].copy()

    no_dropped = no_labels - df_filtered.shape[0]

    logger.debug(
        "Removed %d file IDs from label file based on IDs present in data folder.",
        no_dropped,
    )

    return df_filtered


def _apply_column_operations_to_df(
    df: pd.DataFrame, operations_dict: al_all_column_ops, label_columns: Sequence[str]
) -> pd.DataFrame:
    """
    We want to be able to dynamically apply various operations to different columns
    in the label file (e.g. different operations for creating obesity labels or parsing
    country of origin).

    We consider applying a column operation if:

        1. The column is in the df, hence loaded explicitly or as an extra column.
        2. It is not in the df, but in label columns. Hence expected to be created
           by the column op.

    If a column operation is supposed to only be applied if its column is a label
    column, make sure it's not applied in other cases (e.g. if the column is a
    embedding / continuous input to another target).

    Why this 'base'? In the custom column operations, we might have operations that
    should always be called. They have the key 'base' in the column_ops dictionary.
    Same logic for 'post', we might have operations that should only be applied
    after all other stand-alone operations have been applied.

    Note that we have the return condition there for empty dataframes currently. This
    is because currently we do not enforce on the applied operations that they should
    always work with empty dataframes, and in e.g. cases where we are applying row-based
    filtering operations on chunks, it can happens that an empty chunk ensues. In that
    case, we will pass an empty df to operation functions that expect an actual df,
    which in many cases will cause them to raise an error. To avoid this, we immediately
    return the empty df if encountered.

    :param df: Dataframe to perform processing on.
    :param operations_dict: A dictionary of column names, where each value is a list
    of tuples, where each tuple is a callable as the first element and the callable's
    arguments as the second element.
    :param label_columns:
    :return: Parsed dataframe.
    """

    for operation_name, operation_sequences in operations_dict.items():

        if len(df) == 0:
            return df

        if _should_apply_op_sequence(
            operation_name=operation_name,
            columns_in_df=df.columns,
            label_columns=label_columns,
        ):

            df = _apply_operation_sequence(
                df=df,
                operation_sequence=operation_sequences,
                operation_name=operation_name,
                label_columns=label_columns,
            )

    if len(df) == 0:
        return df

    if "post" in operations_dict.keys():
        post_operations = operations_dict.get("post")

        df = _apply_operation_sequence(
            df=df,
            operation_sequence=post_operations,
            operation_name="post",
            label_columns=label_columns,
        )

    return df


def _should_apply_op_sequence(
    operation_name: str, columns_in_df: Sequence[str], label_columns: Sequence[str]
) -> bool:
    column_in_df = operation_name in columns_in_df
    column_expected_to_be_made = (
        operation_name in label_columns and operation_name not in columns_in_df
    )
    is_candidate = (
        column_in_df or column_expected_to_be_made or operation_name in ["base"]
    )
    return is_candidate


def _apply_operation_sequence(
    df: pd.DataFrame,
    operation_sequence: Sequence[ColumnOperation],
    operation_name: str,
    label_columns: Sequence[str],
) -> pd.DataFrame:

    for operation in operation_sequence:

        if _should_apply_single_op(
            column_operation=operation,
            operation_name=operation_name,
            label_columns=label_columns,
        ):

            df = apply_column_op(
                df=df,
                operation=operation,
                operation_name=operation_name,
            )

    return df


def _should_apply_single_op(
    column_operation: ColumnOperation, operation_name: str, label_columns: Sequence[str]
) -> bool:
    only_apply_if_target = column_operation.only_apply_if_target
    not_a_label_col = operation_name not in label_columns
    do_skip = only_apply_if_target and not_a_label_col

    do_call = not do_skip or operation_name in ["base", "post"]
    return do_call


def apply_column_op(
    df: pd.DataFrame,
    operation: ColumnOperation,
    operation_name: str,
) -> pd.DataFrame:

    func, args_dict = operation.function, operation.function_args
    logger.debug(
        "Applying func %s with args %s to column %s in pre-processing.",
        func,
        reprlib.repr(args_dict),
        operation_name,
    )
    logger.debug("Shape before: %s", df.shape)
    df = func(df=df, column_name=operation_name, **args_dict)
    logger.debug("Shape after: %s", df.shape)

    return df


def _check_parsed_label_df(
    df_labels: pd.DataFrame, supplied_label_columns: Sequence[str]
) -> pd.DataFrame:

    missing_columns = set(supplied_label_columns) - set(df_labels.columns)
    if missing_columns:
        raise ValueError(
            f"Columns asked for in CL args ({missing_columns}) "
            f"missing from columns in label dataframe (with columns "
            f"{df_labels.columns}. The missing columns are not "
            f"found in the raw label file and not calculated by a forward "
            f"reference in the supplied custom library."
        )

    assert df_labels.index.dtype == object

    column_dtypes = df_labels.dtypes.to_dict()

    for column, dtype in column_dtypes.items():
        assert isinstance(dtype, pd.CategoricalDtype) or dtype == float, (column, dtype)

        if isinstance(dtype, pd.CategoricalDtype):
            categories = df_labels[column].cat.categories
            assert all(isinstance(i, str) for i in categories), categories

    return df_labels


def get_passed_in_columns(tabular_info: TabularFileInfo) -> Sequence[str]:
    cat_columns = tabular_info.cat_columns
    con_columns = tabular_info.con_columns

    passed_in_columns = list(cat_columns) + list(con_columns)

    return passed_in_columns


def _drop_not_needed_label_columns(
    df: pd.DataFrame, needed_label_columns: Sequence[str]
) -> pd.DataFrame:

    to_drop = [i for i in df.columns if i not in needed_label_columns]

    if to_drop:
        df = df.drop(to_drop, axis=1)

    return df


def _split_df_by_ids(
    df: pd.DataFrame, train_ids: Sequence[str], valid_ids: Sequence[str]
) -> al_train_val_dfs:

    df_labels_train = df.loc[df.index.intersection(train_ids)]
    df_labels_valid = df.loc[df.index.intersection(valid_ids)]
    assert len(df_labels_train) + len(df_labels_valid) == len(df)

    return df_labels_train, df_labels_valid


def split_ids(
    ids: Sequence[str], valid_size: Union[int, float]
) -> Tuple[Sequence[str], Sequence[str]]:

    train_ids, valid_ids = train_test_split(
        list(ids), test_size=valid_size, random_state=0
    )

    assert len(train_ids) + len(valid_ids) == len(ids)

    return train_ids, valid_ids


def _process_train_and_label_dfs(
    tabular_info: TabularFileInfo,
    df_labels_train: pd.DataFrame,
    df_labels_valid: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, al_label_transformers]:
    """
    NOTE: Possibly, we will run into problem here in the future in the scenario that
    we have NA in the df_valid, but not in df_train. Then, we will have to add call
    to a helper function (e.g. _match_df_category_columns). Another even more
    problematic scenario is if we only have NA in the test set (because here we have
    knowledge of both train / valid, so we can match them, but we cannot assume we
    have access to the test set at the time of training).
    """

    train_con_means = _get_con_manual_vals_dict(
        df=df_labels_train, con_columns=tabular_info.con_columns
    )

    df_labels_train_no_nan = handle_missing_label_values_in_df(
        df=df_labels_train,
        cat_label_columns=tabular_info.cat_columns,
        con_label_columns=tabular_info.con_columns,
        con_manual_values=train_con_means,
        name="train df",
    )

    df_labels_valid_no_nan = handle_missing_label_values_in_df(
        df=df_labels_valid,
        cat_label_columns=tabular_info.cat_columns,
        con_label_columns=tabular_info.con_columns,
        con_manual_values=train_con_means,
        name="valid df",
    )

    label_columns = {
        "con": list(tabular_info.con_columns),
        "cat": list(tabular_info.cat_columns),
    }
    fit_label_transformers = _get_fit_label_transformers(
        df_labels=df_labels_train_no_nan, label_columns=label_columns
    )
    df_train_final = transform_label_df(
        df_labels=df_labels_train_no_nan, label_transformers=fit_label_transformers
    )
    df_valid_final = transform_label_df(
        df_labels=df_labels_valid_no_nan, label_transformers=fit_label_transformers
    )

    return df_train_final, df_valid_final, fit_label_transformers


def _get_con_manual_vals_dict(
    df: pd.DataFrame, con_columns: Sequence[str]
) -> Dict[str, float]:
    con_means_dict = {column: df[column].mean() for column in con_columns}
    return con_means_dict


def handle_missing_label_values_in_df(
    df: pd.DataFrame,
    cat_label_columns: Sequence[str],
    con_label_columns: Sequence[str],
    con_manual_values: Union[Dict[str, float], None] = None,
    name: str = "df",
) -> pd.DataFrame:

    df_filled_cat = _fill_categorical_nans(
        df=df, column_names=cat_label_columns, name=name
    )

    df_filled_final = _fill_continuous_nans(
        df=df_filled_cat,
        column_names=con_label_columns,
        name=name,
        con_means_dict=con_manual_values,
    )

    return df_filled_final


def _fill_categorical_nans(
    df: pd.DataFrame,
    column_names: Sequence[str],
    name: str = "df",
) -> pd.DataFrame:
    """
    Note when dealing with categories, we have to make sure it exists in the parent
    category mapping before adding it as a value to the column.
    """

    missing_stats = _get_missing_stats_string(df=df, columns_to_check=column_names)
    logger.debug(
        "Replacing NaNs in categorical columns %s (counts: %s) in %s with 'NA'.",
        column_names,
        missing_stats,
        name,
    )
    for column in column_names:

        na_found = df[column].isna().sum() != 0
        if na_found:
            df[column] = df[column].cat.add_categories("NA").fillna("NA")

    return df


def _fill_continuous_nans(
    df: pd.DataFrame,
    column_names: Sequence[str],
    con_means_dict: Dict[str, float],
    name: str = "df",
) -> pd.DataFrame:

    missing_stats = _get_missing_stats_string(df, column_names)
    logger.debug(
        "Replacing NaNs in continuous columns %s (counts: %s) in %s with %s",
        column_names,
        missing_stats,
        name,
        con_means_dict,
    )

    df[column_names] = df[column_names].fillna(con_means_dict)
    return df


def _get_missing_stats_string(
    df: pd.DataFrame, columns_to_check: Sequence[str]
) -> Dict[str, int]:
    missing_count_dict = {}
    for col in columns_to_check:
        missing_count_dict[col] = int(df[col].isna().sum())

    return missing_count_dict


def get_transformer_path(run_path: Path, transformer_name: str) -> Path:
    transformer_path = run_path / "transformers" / f"{transformer_name}.save"

    return transformer_path


def merge_target_columns(
    target_con_columns: List[str], target_cat_columns: List[str]
) -> al_target_columns:

    if len(target_con_columns + target_cat_columns) == 0:
        raise ValueError("Expected at least 1 label column")

    all_target_columns = {"con": [], "cat": []}

    [all_target_columns["con"].append(i) for i in target_con_columns]
    [all_target_columns["cat"].append(i) for i in target_cat_columns]

    assert len(all_target_columns) > 0

    return all_target_columns


def save_transformer_set(transformers: al_label_transformers, run_folder: Path) -> None:

    for (transformer_name, transformer_object) in transformers.items():
        save_label_transformer(
            run_folder=run_folder,
            transformer_name=transformer_name,
            target_transformer_object=transformer_object,
        )


def save_label_transformer(
    run_folder: Path,
    transformer_name: str,
    target_transformer_object: al_label_transformers_object,
) -> Path:
    target_transformer_outpath = get_transformer_path(
        run_path=run_folder, transformer_name=transformer_name
    )
    ensure_path_exists(target_transformer_outpath)
    joblib.dump(value=target_transformer_object, filename=target_transformer_outpath)

    return target_transformer_outpath
