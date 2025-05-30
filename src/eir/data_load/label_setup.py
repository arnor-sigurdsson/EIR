import reprlib
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
)

import numpy as np
import polars as pl

pl.enable_string_cache()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from tqdm import tqdm

from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    is_deeplake_sample_missing,
    load_deeplake_dataset,
)
from eir.setup.schemas import InputConfig
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.train_utils.utils import get_seed
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_train_val_dfs = tuple[pl.DataFrame, pl.DataFrame]

# e.g. 'Asia' or '5' for categorical or 1.511 for continuous
al_label_values_raw = float | int | Path
al_sample_labels_raw = dict[str, al_label_values_raw]
al_label_dict = dict[str, al_sample_labels_raw]
al_target_labels = pl.DataFrame
al_target_columns = dict[Literal["con", "cat"], list[str]]
type al_label_transformers_object = (
    StandardScaler | LabelEncoder | KBinsDiscretizer | IdentityTransformer
)
al_label_transformers = dict[str, al_label_transformers_object]


@dataclass
class Labels:
    train_labels: pl.DataFrame
    valid_labels: pl.DataFrame
    label_transformers: al_label_transformers

    @property
    def all_labels(self) -> pl.DataFrame:
        return pl.concat([self.train_labels, self.valid_labels])


@dataclass
class TabularFileInfo:
    file_path: Path
    con_columns: Sequence[str]
    cat_columns: Sequence[str]
    parsing_chunk_size: None | int = None


def set_up_train_and_valid_tabular_data(
    tabular_file_info: TabularFileInfo,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    impute_missing: bool = False,
    do_transform_labels: bool = True,
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
    )
    _validate_df(df=df_labels)

    df_labels_train, df_labels_valid = _split_df_by_ids(
        df=df_labels,
        train_ids=list(train_ids),
        valid_ids=list(valid_ids),
    )
    pre_check_label_df(df=df_labels_train, name="Training DataFrame")
    pre_check_label_df(df=df_labels_valid, name="Validation DataFrame")
    check_train_valid_df_sync(
        df_train=df_labels_train,
        df_valid=df_labels_valid,
        cat_columns=tabular_file_info.cat_columns,
    )

    label_transformers: al_label_transformers = {}
    if do_transform_labels:
        df_labels_train, df_labels_valid, label_transformers = (
            _process_train_and_label_dfs(
                tabular_info=tabular_file_info,
                df_labels_train=df_labels_train,
                df_labels_valid=df_labels_valid,
                impute_missing=impute_missing,
            )
        )

    labels_data_object = Labels(
        train_labels=df_labels_train,
        valid_labels=df_labels_valid,
        label_transformers=label_transformers,
    )

    return labels_data_object


def _get_fit_label_transformers(
    df_labels_train: pl.DataFrame,
    df_labels_full: pl.DataFrame,
    label_columns: al_target_columns,
    impute_missing: bool,
) -> al_label_transformers:
    label_transformers = {}

    for column_type in label_columns:
        label_columns_for_cur_type = label_columns[column_type]

        if label_columns_for_cur_type:
            logger.debug(
                "Fitting transformers on %s label columns %s",
                column_type,
                label_columns_for_cur_type,
            )

        for label_column in label_columns_for_cur_type:
            if column_type == "con":
                cur_values = df_labels_train.get_column(name=label_column)
            elif column_type == "cat":
                cur_values = df_labels_full.get_column(name=label_column)
            else:
                raise ValueError(f"Unknown column type: {column_type}")

            cur_transformer = _get_transformer(column_type=column_type)
            cur_target_transformer_fit = _fit_transformer_on_label_column(
                column_series=cur_values,
                transformer=cur_transformer,
                impute_missing=impute_missing,
            )
            label_transformers[label_column] = cur_target_transformer_fit

    return label_transformers


def _get_transformer(column_type):
    if column_type in ("con", "extra_con"):
        return StandardScaler()
    if column_type == "cat":
        return LabelEncoder()

    raise ValueError()


def _fit_transformer_on_label_column(
    column_series: pl.Series,
    transformer: al_label_transformers_object,
    impute_missing: bool,
) -> al_label_transformers_object:
    if column_series.dtype == pl.Categorical or column_series.dtype == pl.Utf8:
        unique_values = column_series.unique().drop_nulls()
        series_values = unique_values.to_numpy()
        if impute_missing:
            series_values = np.append(series_values, "__NULL__")
    else:
        series_values = column_series.drop_nulls().to_numpy()

    series_values_streamlined = streamline_values_for_transformers(
        transformer=transformer,
        values=series_values,
    )

    transformer.fit(series_values_streamlined)
    return transformer


def streamline_values_for_transformers(
    transformer: al_label_transformers_object, values: np.ndarray
) -> np.ndarray:
    """
    LabelEncoder() expects a 1D array, whereas StandardScaler() expects a 2D one.
    """

    if isinstance(transformer, StandardScaler | KBinsDiscretizer | IdentityTransformer):
        values_reshaped = values.reshape(-1, 1)
        return values_reshaped

    return values


def transform_label_df(
    df_labels: pl.DataFrame,
    label_transformers: al_label_transformers,
    impute_missing: bool,
) -> pl.DataFrame:
    expr = []

    for column_name, transformer_instance in label_transformers.items():
        series_values = df_labels.get_column(column_name).to_numpy()
        transform_func = transformer_instance.transform

        if impute_missing:
            series_values_streamlined = streamline_values_for_transformers(
                transformer=transformer_instance,
                values=series_values,
            )
            transformed = transform_func(series_values_streamlined)
            expr.append(pl.lit(transformed.squeeze()).alias(column_name))
        else:
            match transformer_instance:
                case StandardScaler():
                    non_nan_mask = ~df_labels.get_column(column_name).is_null()
                case LabelEncoder():
                    non_nan_mask = ~df_labels.get_column(column_name).is_null()
                case KBinsDiscretizer() | IdentityTransformer():
                    non_nan_mask = ~df_labels.get_column(column_name).is_null()
                case _:
                    raise ValueError(
                        f"Unknown transformer type: {type(transformer_instance)}"
                    )

            non_nan_values = series_values[non_nan_mask.to_numpy()]
            series_values_streamlined = streamline_values_for_transformers(
                transformer=transformer_instance,
                values=non_nan_values,
            )

            transformed_values = transform_func(series_values_streamlined).squeeze()

            result = np.full(len(series_values), np.nan, dtype=np.float32)
            result[non_nan_mask.to_numpy()] = transformed_values

            if isinstance(transformer_instance, LabelEncoder):
                result[~non_nan_mask.to_numpy()] = np.nan

            expr.append(pl.lit(result).alias(column_name))

    df_final = df_labels.with_columns(expr)
    return df_final


class LabelDFParseWrapperProtocol(Protocol):
    def __call__(
        self,
        label_file_tabular_info: TabularFileInfo,
        ids_to_keep: None | Sequence[str],
    ) -> pl.DataFrame: ...


def get_label_parsing_wrapper(
    label_parsing_chunk_size: None | int,
) -> LabelDFParseWrapperProtocol:
    if label_parsing_chunk_size is None:
        return label_df_parse_wrapper
    return chunked_label_df_parse_wrapper


def _validate_df(df: pl.DataFrame) -> None:
    duplicate_counts = (
        df.group_by("ID").agg(pl.count().alias("count")).filter(pl.col("count") > 1)
    )

    if duplicate_counts.height > 0:
        duplicated_ids = duplicate_counts.select("ID").limit(10).to_series().to_list()
        duplicated_indices_str = ", ".join(map(str, duplicated_ids))
        raise ValueError(
            f"Found duplicated indices in the dataframe. "
            f"Random examples: {duplicated_indices_str}. "
            f"Please make sure that the indices in the ID column are unique."
        )


def label_df_parse_wrapper(
    label_file_tabular_info: TabularFileInfo,
    ids_to_keep: None | Sequence[str],
) -> pl.DataFrame:
    all_label_columns, dtypes = _get_all_label_columns_and_dtypes(
        cat_columns=label_file_tabular_info.cat_columns,
        con_columns=label_file_tabular_info.con_columns,
    )

    df_labels = _load_label_df(
        label_fpath=label_file_tabular_info.file_path,
        columns=all_label_columns,
        dtypes=dtypes,
    )

    df_labels_filtered = _filter_ids_from_label_df(
        df_labels=df_labels,
        ids_to_keep=ids_to_keep,
    )

    supplied_columns = get_passed_in_columns(tabular_info=label_file_tabular_info)
    df_column_filtered = _drop_not_needed_label_columns(
        df=df_labels_filtered,
        needed_label_columns=supplied_columns,
    )

    df_cat_str = ensure_categorical_columns_and_format(df=df_column_filtered)

    df_final = _check_parsed_label_df(
        df_labels=df_cat_str,
        supplied_label_columns=supplied_columns,
    )

    return df_final


def chunked_label_df_parse_wrapper(
    label_file_tabular_info: TabularFileInfo,
    ids_to_keep: None | Sequence[str],
) -> pl.DataFrame:
    label_columns, dtypes = _get_all_label_columns_and_dtypes(
        cat_columns=label_file_tabular_info.cat_columns,
        con_columns=label_file_tabular_info.con_columns,
    )

    dtypes = _ensure_id_str_dtype(dtypes=dtypes)

    assert isinstance(label_file_tabular_info.parsing_chunk_size, int)

    stream = pl.read_csv_batched(
        source=label_file_tabular_info.file_path,
        schema_overrides=dtypes,
        has_header=True,
        batch_size=label_file_tabular_info.parsing_chunk_size,
    )

    supplied_columns = get_passed_in_columns(tabular_info=label_file_tabular_info)

    def process_chunk(chunk: pl.DataFrame) -> pl.DataFrame | None:
        if chunk.height == 0:
            return None

        chunk = chunk.select([pl.col("ID").cast(pl.Utf8), pl.all().exclude("ID")])

        df_labels_filtered = _filter_ids_from_label_df(
            df_labels=chunk,
            ids_to_keep=ids_to_keep,
        )

        if df_labels_filtered.height == 0:
            return None

        return _drop_not_needed_label_columns(
            df=df_labels_filtered,
            needed_label_columns=supplied_columns,
        )

    processed_chunks: list[pl.DataFrame] = []

    while True:
        next_batches = stream.next_batches(1)

        if not next_batches:
            break

        for chunk in next_batches:
            processed_chunk = process_chunk(chunk=chunk)
            if processed_chunk is not None:
                pre_check_label_df(
                    df=processed_chunk,
                    name=f"{label_file_tabular_info.file_path}:chunk",
                )
                processed_chunks.append(processed_chunk)

    if not processed_chunks:
        return pl.DataFrame(schema=dtypes)

    df_concat = pl.concat(processed_chunks)

    df_cat_str = ensure_categorical_columns_and_format(df=df_concat)

    df_final = _check_parsed_label_df(
        df_labels=df_cat_str,
        supplied_label_columns=supplied_columns,
    )

    return df_final


def ensure_categorical_columns_and_format(df: pl.DataFrame) -> pl.DataFrame:
    expr = []
    for column in df.columns:
        if column == "ID":
            continue

        col_dtype = df.schema[column]
        if col_dtype in [pl.Object, pl.Categorical, pl.Utf8]:
            expr.append(pl.col(column).cast(pl.Categorical).alias(column))
        else:
            expr.append(pl.col(column))

    return df.with_columns(expr)


def gather_all_ids_from_all_inputs(
    input_configs: Sequence[InputConfig],
) -> tuple[str, ...]:
    ids = set()
    for input_config in input_configs:
        cur_source = Path(input_config.input_info.input_source)
        cur_type = input_config.input_info.input_type
        if cur_type in ["omics", "sequence", "bytes", "image", "array"]:
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


def gather_ids_from_data_source(
    data_source: Path,
    validate: bool = True,
) -> tuple[str, ...]:
    iterator: Iterator[str] | Iterator[Path]
    if is_deeplake_dataset(data_source=str(data_source)):
        iterator = build_deeplake_available_id_iterator(
            data_source=data_source,
            inner_key="ID",
        )
    elif data_source.suffix == ".csv":
        ids = gather_ids_from_tabular_file(file_path=data_source)
        iterator = (str(i) for i in ids)
    else:
        iterator = get_file_path_iterator(data_source=data_source, validate=validate)
        iterator = (i.stem for i in iterator)

    logger.debug("Gathering IDs from %s.", data_source)
    all_ids = tuple(i for i in tqdm(iterator, desc="Progress"))

    return all_ids


def build_deeplake_available_id_iterator(
    data_source: Path, inner_key: str
) -> Generator[str]:
    deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
    columns = {col.name for col in deeplake_ds.schema.columns}
    existence_col = f"{inner_key}_exists"
    for row in deeplake_ds:
        if is_deeplake_sample_missing(
            row=row,
            existence_col=existence_col,
            columns=columns,
        ):
            continue

        id_ = row["ID"]

        yield id_  # type: ignore


@lru_cache
def gather_ids_from_tabular_file(file_path: Path) -> tuple[str, ...]:
    df = pl.read_csv(file_path, columns=["ID"])
    all_ids = tuple(df.select(pl.col("ID").cast(pl.Utf8)).to_series().to_list())
    return all_ids


def get_file_path_iterator(data_source: Path, validate: bool = True) -> Iterator[Path]:
    def _file_iterator(file_path: Path) -> Iterator[Path]:
        with open(str(file_path)) as infile:
            for line in infile:
                path = Path(line.strip())

                if validate and not path.exists():
                    raise FileNotFoundError(
                        f"Could not find array {path} listed in {data_source}."
                    )

                yield path

    if data_source.is_dir():
        return data_source.rglob("*")
    if data_source.is_file():
        return _file_iterator(file_path=data_source)

    if not data_source.exists():
        raise FileNotFoundError("Could not find data source %s.", data_source)
    raise ValueError(
        "Data source %s is neither recognized as a file nor folder.", data_source
    )


def _get_all_label_columns_and_dtypes(
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
) -> tuple[Sequence[str], dict[str, type[pl.Categorical] | type[pl.Float32]]]:
    supplied_label_columns = _get_column_dtypes(
        cat_columns=cat_columns,
        con_columns=con_columns,
    )

    all_cols_and_dtypes = {**supplied_label_columns}
    all_cols = tuple(all_cols_and_dtypes.keys())
    supplied_dtypes = {k: v for k, v in all_cols_and_dtypes.items() if v is not None}

    return all_cols, supplied_dtypes


def _get_column_dtypes(
    cat_columns: Sequence[str], con_columns: Sequence[str]
) -> dict[str, type[pl.Categorical] | type[pl.Float32]]:
    dtypes: dict[str, type[pl.Categorical] | type[pl.Float32]] = {}

    for cat_column in cat_columns:
        dtypes[cat_column] = pl.Categorical
    for con_column in con_columns:
        dtypes[con_column] = pl.Float32

    return dtypes


def _load_label_df(
    label_fpath: Path,
    columns: Sequence[str],
    dtypes: dict[str, type[pl.Categorical] | type[pl.Float32]] | None = None,
) -> pl.DataFrame:
    dtypes = _ensure_id_str_dtype(dtypes=dtypes)

    logger.debug("Reading in labelfile: %s. ID is read as str dtype.", label_fpath)

    columns_with_id_col = ["ID"] + list(columns)
    available_columns = _get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=columns_with_id_col,
    )

    df_labels = pl.read_csv(
        label_fpath,
        columns=available_columns,
        schema_overrides=dtypes,
    )

    df_labels = df_labels.select([pl.col("ID").cast(pl.Utf8), pl.all().exclude("ID")])

    pre_check_label_df(df=df_labels, name=str(label_fpath))

    return df_labels


def pre_check_label_df(df: pl.DataFrame, name: str) -> None:
    for column in df.columns:
        if df.get_column(column).is_null().all():
            raise ValueError(
                f"All values are NULL in column '{column}' from {name}. "
                f"This can either be due to all values in the column actually "
                f"being NULL, or an unfavorable split happened during train/validation "
                f"splitting, causing all values in the training split for the column "
                f"to be NULL. For now this will raise an error, but might be handled "
                f"in the future."
                f" In any case, please remove '{column}' as an input/target for now.",
            )


def check_train_valid_df_sync(
    df_train: pl.DataFrame, df_valid: pl.DataFrame, cat_columns: Sequence[str]
) -> None:
    for col in cat_columns:
        train_values = set(df_train.get_column(col).unique().to_list())
        valid_values = set(df_valid.get_column(col).unique().to_list())

        mismatched_values = valid_values - train_values

        if mismatched_values:
            total_mismatch_count = df_valid.filter(
                pl.col(col).is_in(list(mismatched_values))
            ).height
            total_count = df_valid.height + df_train.height
            percentage = (total_mismatch_count / total_count) * 100

            error_message = (
                f"Mismatched values found in column '{col}': {mismatched_values}. "
                f"Count: {total_mismatch_count}, "
                f"Percentage of total data: {percentage:.2f}%.\n"
                f"This happens as there are values in the validation set that "
                f"are not present in the training set. "
                f"This can happen by chance when working with sparse/rare "
                f"categorical values and/or small datasets. These values will still be "
                f"encoded and used during validation, but might result in nonsensical "
                f"predictions as the model never saw them during training.\n"
                f"One approach to fix this "
                f"is trying a different train/validation split, "
                f"which can be done by:\n"
                f"  1. Running the command with a different seed, "
                f"e.g. EIR_SEED=1 eirtrain ...\n"
                f"  2. Manually specifying the train/validation split, "
                f" using manual_valid_ids_file: <.txt file> in the global "
                f"configuration.\n"
                f"  3. Using a different train/validation split ratio, using "
                f" the valid_size parameter in the global configuration."
                f"Other solutions include:\n"
                f"  4. Skipping the column if it's not crucial.\n"
                f"  5. Binning sparse/rare values into broader categories.\n"
                f"Finally, this will very likely raise an error during testing, "
                f"as the label encoder will likely encounter values "
                f"it has never seen before."
            )
            logger.warning(error_message)


def _ensure_id_str_dtype(dtypes: dict[str, Any] | None) -> dict[str, Any]:
    if dtypes is None:
        dtypes = {"ID": pl.Utf8}
    elif "ID" not in dtypes:
        dtypes["ID"] = pl.Utf8

    return dtypes


def _get_currently_available_columns(
    label_fpath: Path,
    requested_columns: list[str],
) -> list[str]:
    label_file_columns_set = set(pl.read_csv(source=label_fpath, n_rows=0).columns)

    requested_columns_set = set(requested_columns)

    missing_columns = requested_columns_set - label_file_columns_set
    if missing_columns:
        raise ValueError(f"Could not find columns {missing_columns} in {label_fpath}.")

    available_columns = requested_columns_set.intersection(label_file_columns_set)

    return list(available_columns)


def _filter_ids_from_label_df(
    df_labels: pl.DataFrame,
    ids_to_keep: None | Sequence[str] = None,
) -> pl.DataFrame:
    if not ids_to_keep:
        return df_labels

    no_labels = df_labels.height

    df_filtered = df_labels.filter(pl.col("ID").is_in(ids_to_keep))

    no_dropped = no_labels - df_filtered.height

    logger.debug(
        "Removed %d file IDs from label file based on IDs present in data folder.",
        no_dropped,
    )

    return df_filtered


def _check_parsed_label_df(
    df_labels: pl.DataFrame,
    supplied_label_columns: Sequence[str],
) -> pl.DataFrame:
    """
    Validate DataFrame structure and types using Polars.
    """
    missing_columns = set(supplied_label_columns) - set(df_labels.columns)
    if missing_columns:
        raise ValueError(
            f"Columns asked for in CL args ({missing_columns}) "
            f"missing from columns in label dataframe (with columns "
            f"{df_labels.columns}. The missing columns are not "
            f"found in the raw label file."
        )

    id_dtype = df_labels.schema["ID"]
    assert id_dtype == pl.Utf8, f"ID column must be string type, got {id_dtype}"

    for column in df_labels.columns:
        dtype = df_labels.schema[column]

        if column == "ID":
            assert dtype == pl.Utf8, f"ID column must be string type, got {dtype}"
            continue

        assert dtype in [
            pl.Float32,
            pl.Categorical,
        ], f"Column {column} has invalid type {dtype}"

        if dtype == pl.Categorical:
            non_null_vals = (
                df_labels.select([column])
                .filter(pl.col(column).is_not_null())
                .get_column(column)
                .unique()
            )

            if len(non_null_vals) > 0:
                vals_and_types = [(val, type(val)) for val in non_null_vals]
                all_str = all(isinstance(val, str) for val, _ in vals_and_types)

                if not all_str:
                    raise ValueError(
                        f"Non-string values found in string column {column}"
                        f"Got values {vals_and_types}."
                    )

    return df_labels


def get_passed_in_columns(tabular_info: TabularFileInfo) -> Sequence[str]:
    cat_columns = tabular_info.cat_columns
    con_columns = tabular_info.con_columns

    passed_in_columns = list(cat_columns) + list(con_columns)

    return passed_in_columns


def _drop_not_needed_label_columns(
    df: pl.DataFrame, needed_label_columns: Sequence[str]
) -> pl.DataFrame:
    needed_columns = ["ID"] + list(needed_label_columns)
    return df.select(needed_columns)


def _split_df_by_ids(
    df: pl.DataFrame,
    train_ids: list[str],
    valid_ids: list[str],
) -> al_train_val_dfs:
    df_labels_train = df.filter(pl.col("ID").is_in(train_ids))
    df_labels_valid = df.filter(pl.col("ID").is_in(valid_ids))

    msg = (
        "Total number of rows in train and validation "
        "sets doesn't match original DataFrame"
    )
    assert df_labels_train.height + df_labels_valid.height == df.height, msg

    return df_labels_train, df_labels_valid


def split_ids(
    ids: Sequence[str],
    valid_size: int | float,
    manual_valid_ids: None | Sequence[str] = None,
) -> tuple[Sequence[str], Sequence[str]]:
    """
    We sort here to ensure that we get the same splits every time.
    """

    if manual_valid_ids:
        logger.info(
            "Doing a manual split into validation set with %d IDs read from file.",
            len(manual_valid_ids),
        )
        train_ids, valid_ids = _split_ids_manual(
            ids=ids, manual_valid_ids=manual_valid_ids
        )

    else:
        seed, _ = get_seed()
        ids_sorted = sorted(ids)
        train_ids, valid_ids = train_test_split(
            ids_sorted, test_size=valid_size, random_state=seed
        )

    assert len(train_ids) + len(valid_ids) == len(ids)
    assert set(train_ids).isdisjoint(set(valid_ids))

    return train_ids, valid_ids


def _split_ids_manual(
    ids: Sequence[str], manual_valid_ids: Sequence[str]
) -> tuple[Sequence[str], Sequence[str]]:
    ids_set = set(ids)
    not_found = tuple(i for i in manual_valid_ids if i not in ids_set)
    if not_found:
        raise ValueError(
            f"Did not find {len(not_found)} manual validation IDs "
            f"'{reprlib.repr(not_found)}' among those IDs. Possibly some validation "
            f"IDs are not present, or this is a bug."
        )

    train_ids_set = set(ids) - set(manual_valid_ids)
    train_ids = list(train_ids_set)
    valid_ids = list(manual_valid_ids)

    return train_ids, valid_ids


def _process_train_and_label_dfs(
    tabular_info: TabularFileInfo,
    df_labels_train: pl.DataFrame,
    df_labels_valid: pl.DataFrame,
    impute_missing: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, al_label_transformers]:
    train_con_means = _get_con_manual_vals_dict(
        df=df_labels_train,
        con_columns=tabular_info.con_columns,
    )

    df_labels_train_no_nan = handle_missing_label_values_in_df(
        df=df_labels_train,
        cat_label_columns=tabular_info.cat_columns,
        con_label_columns=tabular_info.con_columns,
        con_manual_values=train_con_means,
        name="train df",
        impute_missing=impute_missing,
    )

    df_labels_valid_no_nan = handle_missing_label_values_in_df(
        df=df_labels_valid,
        cat_label_columns=tabular_info.cat_columns,
        con_label_columns=tabular_info.con_columns,
        con_manual_values=train_con_means,
        name="valid df",
        impute_missing=impute_missing,
    )

    df_labels_full = pl.concat([df_labels_train_no_nan, df_labels_valid_no_nan])

    label_columns: al_target_columns = {
        "con": list(tabular_info.con_columns),
        "cat": list(tabular_info.cat_columns),
    }
    fit_label_transformers = _get_fit_label_transformers(
        df_labels_train=df_labels_train_no_nan,
        df_labels_full=df_labels_full,
        label_columns=label_columns,
        impute_missing=impute_missing,
    )

    logger.debug("Transforming label columns in train and validation dataframes.")
    df_train_final = transform_label_df(
        df_labels=df_labels_train_no_nan,
        label_transformers=fit_label_transformers,
        impute_missing=impute_missing,
    )
    df_valid_final = transform_label_df(
        df_labels=df_labels_valid_no_nan,
        label_transformers=fit_label_transformers,
        impute_missing=impute_missing,
    )

    return df_train_final, df_valid_final, fit_label_transformers


def _get_con_manual_vals_dict(
    df: pl.DataFrame,
    con_columns: Sequence[str],
) -> dict[str, float]:
    con_means_dict: dict[str, float] = {}

    for column in con_columns:
        column_mean = (
            df.get_column(column).drop_nans().drop_nulls().cast(pl.Float32).mean()
        )
        if isinstance(column_mean, int | float):
            con_means_dict[column] = float(column_mean)
        else:
            con_means_dict[column] = 0.0

    return con_means_dict


def handle_missing_label_values_in_df(
    df: pl.DataFrame,
    cat_label_columns: Sequence[str],
    con_label_columns: Sequence[str],
    impute_missing: bool,
    con_manual_values: dict[str, float] | None = None,
    name: str = "df",
) -> pl.DataFrame:
    df_filled_cat = _fill_categorical_nans(
        df=df,
        column_names=cat_label_columns,
        name=name,
        impute_missing=impute_missing,
    )

    if con_manual_values is None:
        df_filled_final = df_filled_cat
    else:
        df_filled_final = _fill_continuous_nans(
            df=df_filled_cat,
            column_names=con_label_columns,
            name=name,
            con_means_dict=con_manual_values,
            impute_missing=impute_missing,
        )

    return df_filled_final


def _fill_categorical_nans(
    df: pl.DataFrame,
    column_names: Sequence[str],
    impute_missing: bool,
    name: str = "df",
) -> pl.DataFrame:
    missing_stats = _get_missing_stats_string(df=df, columns_to_check=column_names)

    if not impute_missing:
        return df

    logger.debug(
        "Replacing NaNs in categorical columns %s (counts: %s) in %s with '__NULL__'.",
        column_names,
        missing_stats,
        name,
    )

    expr = [
        pl.when(pl.col(col).is_null())
        .then(pl.lit("__NULL__"))
        .otherwise(pl.col(col))
        .alias(col)
        for col in column_names
    ]

    other_cols = [col for col in df.columns if col not in column_names]
    expr.extend([pl.col(col) for col in other_cols])

    return df.with_columns(expr)


def _fill_continuous_nans(
    df: pl.DataFrame,
    column_names: Sequence[str],
    con_means_dict: dict[str, float],
    impute_missing: bool,
    name: str = "df",
) -> pl.DataFrame:
    missing_stats = _get_missing_stats_string(df=df, columns_to_check=column_names)

    if not impute_missing:
        return df

    logger.debug(
        "Replacing NaNs in continuous columns %s (counts: %s) in %s with %s",
        column_names,
        missing_stats,
        name,
        con_means_dict,
    )

    cols_to_fill = con_means_dict.keys()

    expr = [
        pl.when(pl.col(col).is_null() | pl.col(col).is_nan())
        .then(pl.lit(con_means_dict[col]))
        .otherwise(pl.col(col))
        .alias(col)
        for col in cols_to_fill
    ]

    other_cols = [col for col in df.columns if col not in column_names]
    expr.extend([pl.col(col) for col in other_cols])

    return df.with_columns(expr)


def _get_missing_stats_string(
    df: pl.DataFrame, columns_to_check: Sequence[str]
) -> dict[str, int]:
    missing_count_dict = {
        col: int(df.get_column(col).is_null().sum()) for col in columns_to_check
    }
    return missing_count_dict


def merge_target_columns(
    target_con_columns: list[str], target_cat_columns: list[str]
) -> al_target_columns:
    if len(target_con_columns + target_cat_columns) == 0:
        raise ValueError("Expected at least 1 label column")

    all_target_columns: al_target_columns = {
        "con": target_con_columns,
        "cat": target_cat_columns,
    }

    assert len(all_target_columns) > 0

    return all_target_columns
