from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict, Union, List

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

from human_origins_supervised.data_load.common_ops import ColumnOperation
from human_origins_supervised.train_utils.utils import get_custom_module_submodule

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_all_column_ops = Dict[str, Tuple[ColumnOperation, ...]]
al_train_val_dfs = Tuple[pd.DataFrame, pd.DataFrame]
al_label_dict = Dict[str, Dict[str, Union[str, float]]]
al_target_columns = Dict[str, List[str]]
al_label_transformers = Union[StandardScaler, LabelEncoder]


def set_up_train_and_valid_labels(
    cl_args: Namespace,
) -> Tuple[al_label_dict, al_label_dict]:
    """
    Splits and does split based processing (e.g. scaling validation set with training
    set for regression) on the labels.
    """

    df_labels = label_df_parse_wrapper(cl_args=cl_args)

    df_labels_train, df_labels_valid = _split_df(
        df=df_labels, valid_size=cl_args.valid_size
    )

    df_labels_train, df_labels_valid = _process_train_and_label_dfs(
        cl_args=cl_args,
        df_labels_train=df_labels_train,
        df_labels_valid=df_labels_valid,
    )

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")
    return train_labels_dict, valid_labels_dict


def label_df_parse_wrapper(cl_args: Namespace) -> pd.DataFrame:
    available_ids = _gather_ids_from_folder(Path(cl_args.data_folder))

    column_ops = {}
    if cl_args.custom_lib:
        column_ops = _get_custom_column_ops(custom_lib=cl_args.custom_lib)

    all_cols = _get_all_label_columns_needed(cl_args=cl_args, column_ops=column_ops)

    df_labels = _load_label_df(
        label_fpath=cl_args.label_file, columns=all_cols, custom_lib=cl_args.custom_lib
    )

    df_labels_filtered = _filter_ids_from_label_df(
        df_labels=df_labels, ids_to_keep=available_ids
    )

    label_columns = _get_label_columns_from_cl_args(cl_args=cl_args)
    df_labels_parsed = _parse_label_df(
        df=df_labels_filtered, column_ops=column_ops, label_columns=label_columns
    )

    df_column_filtered = _drop_not_needed_label_columns(
        df=df_labels_parsed, needed_label_columns=label_columns
    )

    df_final = _check_parsed_label_df(
        df_labels=df_column_filtered, supplied_label_columns=label_columns
    )

    return df_final


def _get_all_label_columns_needed(
    cl_args: Namespace, column_ops: al_all_column_ops
) -> List[str]:

    supplied_label_columns = _get_label_columns_from_cl_args(cl_args=cl_args)

    extra_label_parsing_cols = _get_extra_columns(
        label_columns=supplied_label_columns, all_column_ops=column_ops
    )
    all_cols = supplied_label_columns + extra_label_parsing_cols

    return all_cols


def _get_label_columns_from_cl_args(cl_args: Namespace) -> List[str]:
    target_columns = cl_args.target_con_columns + cl_args.target_cat_columns
    extra_input_columns = cl_args.contn_columns + cl_args.embed_columns

    all_label_columns = target_columns + extra_input_columns

    return all_label_columns


def _get_extra_columns(
    label_columns: List[str], all_column_ops: al_all_column_ops
) -> List[str]:
    """
    We use this to grab extra columns needed for the current run, as specified in the
    COLUMN_OPS, where the keys are the label columns. That is, "for running with these
    specific label columns, what other columns do we need to grab", as specified
    by the extra_columns_deps attribute of each column operation.

    :param label_columns: The target columns we are modelling on.
    :param all_column_ops: The ledger of all column ops to be done for each target
    column.
    :returns A list of all extra columns needed from the label file for the current run.
    """

    extra_columns = []
    for column in label_columns:

        if column in all_column_ops:
            cur_ops = all_column_ops.get(column)
            cur_extra_columns = [i.extra_columns_deps for i in cur_ops]

            cur_extra_columns_flat = list(
                column for column_deps in cur_extra_columns for column in column_deps
            )
            extra_columns += cur_extra_columns_flat

    return extra_columns


def _gather_ids_from_folder(data_folder: Path):
    logger.debug("Gathering IDs from %s.", data_folder)
    all_ids = tuple(i.stem for i in tqdm(data_folder.iterdir(), desc="Progress"))

    return all_ids


def _load_label_df(
    label_fpath: Path, columns: List[str], custom_lib: Union[str, None]
) -> pd.DataFrame:
    """
    We accept only loading the available columns at this point because the passed
    in columns might be forward referenced, meaning that they might be created
    by the custom library.
    """

    logger.debug("Reading in labelfile: %s", label_fpath)

    columns_with_id_col = ["ID"] + columns
    available_columns = _get_currently_available_columns(
        label_fpath=label_fpath,
        requested_columns=columns_with_id_col,
        custom_lib=custom_lib,
    )

    df_labels = pd.read_csv(label_fpath, usecols=available_columns, dtype={"ID": str})

    df_labels = df_labels.set_index("ID")

    return df_labels


def _get_currently_available_columns(
    label_fpath: Path, requested_columns: List[str], custom_lib: Union[str, None]
) -> List[str]:

    label_file_columns_set = set(pd.read_csv(label_fpath, dtype={"ID": str}, nrows=0))

    requested_columns_set = set(requested_columns)

    if not custom_lib:
        missing_columns = requested_columns_set - label_file_columns_set
        if missing_columns:
            raise ValueError(
                f"No custom library specified and could not find columns "
                f"{missing_columns} in {label_fpath}."
            )

    available_columns = requested_columns_set.intersection(label_file_columns_set)

    return list(available_columns)


def _filter_ids_from_label_df(
    df_labels: pd.DataFrame, ids_to_keep: Tuple[str, ...] = ()
) -> pd.DataFrame:

    if not ids_to_keep:
        return df_labels

    no_labels = df_labels.shape[0]
    df_filtered = df_labels[df_labels.index.isin(ids_to_keep)]
    no_dropped = no_labels - df_filtered.shape[0]

    logger.debug(
        "Removed %d file IDs from label file based on IDs present in data folder.",
        no_dropped,
    )

    return df_filtered


def _parse_label_df(
    df: pd.DataFrame, column_ops: al_all_column_ops, label_columns: List[str]
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

    :param df: Dataframe to perform processing on.
    :param column_ops: A dictionary of column names, where each value is a list
    of tuples, where each tuple is a callable as the first element and the callable's
    arguments as the second element.
    :param label_columns:
    :return: Parsed dataframe.
    """

    def _is_column_candidate(column_name_: str) -> bool:
        column_in_df = column_name_ in df.columns
        column_expected_to_be_made = (
            column_name_ in label_columns and column_name_ not in df.columns
        )
        is_candidate = column_in_df or column_expected_to_be_made
        return is_candidate

    def _do_call_column_op(column_op_: ColumnOperation, column_name_: str) -> bool:
        only_apply_if_target = column_op_.only_apply_if_target
        not_a_label_col = column_name_ not in label_columns
        do_skip = only_apply_if_target and not_a_label_col

        do_call = not do_skip
        return do_call

    for column_name, ops_funcs in column_ops.items():

        if _is_column_candidate(column_name_=column_name):

            for column_op in ops_funcs:

                if _do_call_column_op(column_op_=column_op, column_name_=column_name):

                    func, args_dict = column_op.function, column_op.function_args
                    logger.debug(
                        "Applying func %s with args %s to column %s in pre-processing.",
                        func,
                        args_dict,
                        column_name,
                    )
                    logger.debug("Shape before: %s", df.shape)
                    df = func(df=df, column_name=column_name, **args_dict)
                    logger.debug("Shape after: %s", df.shape)
    return df


def _check_parsed_label_df(
    df_labels: pd.DataFrame, supplied_label_columns: List[str]
) -> pd.DataFrame:

    missing_columns = set(supplied_label_columns) - set(df_labels.columns)
    if missing_columns:
        raise ValueError(
            f"Columns asked for in CL args ({missing_columns}) "
            f"missing from columns in label dataframe (with columns "
            f"{df_labels.columns}. The missing columns are not"
            f"found in the raw label file and not calculated by a forward"
            f"reference in the supplied custom library."
        )

    return df_labels


def _drop_not_needed_label_columns(
    df: pd.DataFrame, needed_label_columns: List[str]
) -> pd.DataFrame:

    to_drop = [i for i in df.columns if i not in needed_label_columns]

    if to_drop:
        df = df.drop(to_drop, axis=1)

    return df


def _get_custom_column_ops(custom_lib: str) -> al_all_column_ops:
    """
    We want to grab operations from a custom library for the current run, as defined
    by the COLUMN_OPS specifications.

    :param custom_lib: Path to the custom library to try loading custom column
    operations from.
    :return: Loaded CUSTOM_OPS variable to be used by other functions to process label
    columns.
    """
    custom_column_ops_module = get_custom_module_submodule(
        custom_lib, "custom_column_ops"
    )

    # If the user has not defined custom_column_ops, we're fine with that
    if not custom_column_ops_module:
        return {}

    if not hasattr(custom_column_ops_module, "COLUMN_OPS"):
        raise ImportError(
            f"'COLUMN_OPS' variable must be defined in "
            f"{custom_column_ops_module} for custom label operations."
            f""
        )

    column_ops: al_all_column_ops = custom_column_ops_module.COLUMN_OPS

    # Also if they have defined an empty COLUMN_OPS, we don't want things to break
    if column_ops is None:
        return {}

    return column_ops


def _split_df(df: pd.DataFrame, valid_size: Union[int, float]) -> al_train_val_dfs:
    train_ids, valid_ids = train_test_split(list(df.index), test_size=valid_size)

    df_labels_train = df.loc[df.index.intersection(train_ids)]
    df_labels_valid = df.loc[df.index.intersection(valid_ids)]
    assert len(df_labels_train) + len(df_labels_valid) == len(df)

    return df_labels_train, df_labels_valid


def _process_train_and_label_dfs(
    cl_args: Namespace, df_labels_train: pd.DataFrame, df_labels_valid: pd.DataFrame
) -> al_train_val_dfs:

    con_columns = cl_args.target_con_columns + cl_args.contn_columns
    train_con_means = _get_con_manual_vals_dict(
        df=df_labels_train, con_columns=con_columns
    )

    df_labels_train = handle_missing_label_values_in_df(
        df=df_labels_train,
        cl_args=cl_args,
        con_manual_values=train_con_means,
        name="train df",
    )

    df_labels_valid = handle_missing_label_values_in_df(
        df=df_labels_valid,
        cl_args=cl_args,
        con_manual_values=train_con_means,
        name="valid df",
    )

    return df_labels_train, df_labels_valid


def _get_con_manual_vals_dict(
    df: pd.DataFrame, con_columns: List[str]
) -> Dict[str, float]:
    con_means_dict = {column: df[column].mean() for column in con_columns}
    return con_means_dict


def handle_missing_label_values_in_df(
    df: pd.DataFrame,
    cl_args: Namespace,
    con_manual_values: Union[Dict[str, float], None] = None,
    name: str = "df",
) -> pd.DataFrame:

    cat_label_columns = cl_args.embed_columns + cl_args.target_cat_columns
    con_label_columns = cl_args.contn_columns + cl_args.target_con_columns

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
    df: pd.DataFrame, column_names: List[str], name: str = "df"
) -> pd.DataFrame:

    missing_stats = _get_missing_stats_string(df, column_names)
    logger.debug(
        "Replacing NaNs in embedding columns %s (counts: %s) in %s with 'NA'.",
        column_names,
        missing_stats,
        name,
    )
    df[column_names] = df[column_names].fillna("NA")
    return df


def _fill_continuous_nans(
    df: pd.DataFrame,
    column_names: List[str],
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
    df: pd.DataFrame, columns_to_check: List[str]
) -> Dict[str, int]:
    missing_count_dict = {}
    for col in columns_to_check:
        missing_count_dict[col] = int(df[col].isnull().sum())

    return missing_count_dict


def get_transformer_path(run_path: Path, transformer_name: str) -> Path:
    transformer_path = run_path / "transformers" / f"{transformer_name}.save"

    return transformer_path


def set_up_label_transformers(
    labels_dict: al_label_dict, label_columns: al_target_columns
) -> Dict[str, al_label_transformers]:

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
            cur_target_transformer = _fit_transformer_on_label_column(
                labels_dict=labels_dict,
                label_column=cur_target_column,
                column_type=column_type,
            )
            label_transformers[cur_target_column] = cur_target_transformer

    return label_transformers


def _fit_transformer_on_label_column(
    labels_dict: al_label_dict, label_column: str, column_type: str
) -> al_label_transformers:

    transformer = _get_transformer(column_type)

    target_values = np.array([i[label_column] for i in labels_dict.values()])
    target_values_streamlined = _streamline_values_for_transformers(
        transformer=transformer, values=target_values
    )

    transformer.fit(target_values_streamlined)

    return transformer


def _get_transformer(column_type):
    if column_type in ("con", "extra_con"):
        return StandardScaler()
    elif column_type == "cat":
        return LabelEncoder()

    raise ValueError()


def _streamline_values_for_transformers(
    transformer: al_label_transformers, values: np.ndarray
) -> np.ndarray:
    """
    LabelEncoder() expects a 1D array, whereas StandardScaler() expects a 2D one.
    """

    if isinstance(transformer, StandardScaler):
        values_reshaped = values.reshape(-1, 1)
        return values_reshaped
    return values
