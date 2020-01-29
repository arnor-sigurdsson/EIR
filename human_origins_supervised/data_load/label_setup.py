from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict, Union, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from human_origins_supervised.data_load.common_ops import ColumnOperation
from human_origins_supervised.train_utils.utils import get_custom_module_submodule

from aislib.misc_utils import get_logger, ensure_path_exists

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_all_column_ops = Dict[str, Tuple[ColumnOperation, ...]]
al_train_val_dfs = Tuple[pd.DataFrame, pd.DataFrame]
al_label_dict = Dict[str, Dict[str, Union[str, float]]]


def set_up_train_and_valid_labels(
    cl_args: Namespace
) -> Tuple[al_label_dict, al_label_dict]:
    """
    Splits and does split based processing (e.g. scaling validation set with training
    set for regression) on the labels.
    """

    df_labels = label_df_parse_wrapper(cl_args)

    df_labels_train, df_labels_valid = _split_df(df_labels, cl_args.valid_size)

    df_labels_train, df_labels_valid = _process_train_and_label_dfs(
        cl_args, df_labels_train, df_labels_valid
    )

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")
    return train_labels_dict, valid_labels_dict


def label_df_parse_wrapper(cl_args: Namespace) -> pd.DataFrame:
    all_ids = _gather_ids_from_folder(Path(cl_args.data_folder))

    column_ops = {}
    if cl_args.custom_lib:
        column_ops = _get_custom_column_ops(cl_args.custom_lib)

    target_columns = cl_args.target_con_columns + cl_args.target_cat_columns

    extra_label_parsing_cols = _get_extra_columns(target_columns, column_ops)
    extra_embed_cols = cl_args.embed_columns
    extra_contn_cols = cl_args.contn_columns

    all_extra_cols = extra_label_parsing_cols + extra_embed_cols + extra_contn_cols

    df_labels = _load_label_df(
        cl_args.label_file, target_columns, all_ids, all_extra_cols
    )
    df_labels = _parse_label_df(df_labels, column_ops, target_columns)

    # remove columns only used for parsing, so only keep actual label column
    # and extra embedding columns
    to_drop = [
        i
        for i in extra_label_parsing_cols
        if i not in extra_embed_cols + extra_contn_cols
    ]
    if to_drop:
        df_labels = df_labels.drop(to_drop, axis=1)

    return df_labels


def _gather_ids_from_folder(data_folder: Path):
    logger.debug("Gathering IDs from %s.", data_folder)
    all_ids = tuple(i.stem for i in tqdm(data_folder.iterdir(), desc="Progress"))

    return all_ids


def _get_extra_columns(
    target_columns: List[str], all_column_ops: al_all_column_ops
) -> List[str]:
    """
    We use this to grab extra columns needed for the current run, as specified in the
    COLUMN_OPS, where the keys are the target_columns. That is, "for running with these
    specific target columns, what other columns do we need to grab", as specified
    by the extra_columns_deps attribute of each column operation.

    :param target_columns: The target columns we are modelling on.
    :param all_column_ops: The ledger of all column ops to be done for each target
    column.
    :returns A list of all extra columns needed from the label file for the current run.
    """

    extra_columns = []
    for target_column in target_columns:

        if target_column in all_column_ops:
            cur_ops = all_column_ops.get(target_column)
            cur_extra_columns = [i.extra_columns_deps for i in cur_ops]

            cur_extra_columns_flat = list(
                column for column_deps in cur_extra_columns for column in column_deps
            )
            extra_columns += cur_extra_columns_flat

    return extra_columns


def _load_label_df(
    label_fpath: Path,
    target_columns: List[str],
    ids_to_keep: Tuple[str, ...] = (),
    extra_columns: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """
    We want to load the label dataframe to be used for the torch dataset setup.

    :param label_fpath: Path to the labelfile as passed in from the CL arguments.
    :param target_columns: Column we are predicting on.
    :param ids_to_keep: Which IDs in the label dataframe we want to keep.
    :param extra_columns: Extra columns in the label dataframe needed for this runþ
    :return: A dataframe of labels.
    """

    logger.debug("Reading in labelfile: %s", label_fpath)
    df_labels = pd.read_csv(
        label_fpath,
        usecols=["ID"] + target_columns + list(extra_columns),
        dtype={"ID": str},
    )

    df_labels = df_labels.set_index("ID")

    if ids_to_keep:
        no_labels = df_labels.shape[0]
        df_labels = df_labels[df_labels.index.isin(ids_to_keep)]
        no_dropped = no_labels - df_labels.shape[0]
        logger.debug(
            "Removed %d file IDs from label file based on IDs present in data folder.",
            no_dropped,
        )

    return df_labels


def _parse_label_df(
    df: pd.DataFrame, column_ops: al_all_column_ops, target_columns: List[str]
) -> pd.DataFrame:
    """
    We want to be able to dynamically apply various operations to different columns
    in the label file (e.g. different operations for creating obesity labels or parsing
    country of origin).

    If a column operation is supposed to only be applied if its column is the target
    variable, make sure it's not applied in other cases (e.g. if the column is a
    embedding / continuous input to another target).

    :param df: Dataframe to perform processing on.
    :param column_ops: A dictionary of column names, where each value is a list
    of tuples, where each tuple is a callable as the first element and the callable's
    arguments as the second element.
    :param target_column:
    :return: Parsed dataframe.
    """

    for column_name, ops_funcs in column_ops.items():
        if column_name in df.columns:
            for column_op in ops_funcs:
                do_skip = (
                    column_op.only_apply_if_target and column_name not in target_columns
                )
                if not do_skip:
                    func, args_dict = column_op.function, column_op.function_args
                    logger.debug(
                        "Applying func %s with args %s to column in pre-processing.",
                        func,
                        args_dict,
                    )
                    logger.debug("Shape before: %s", df.shape)
                    df = func(df=df, column_name=column_name, **args_dict)
                    logger.debug("Shape after: %s", df.shape)
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
    cl_args, df_labels_train, df_labels_valid
) -> al_train_val_dfs:
    # we make sure not to mess with the passed in CL arg, hence copy
    continuous_columns = cl_args.contn_columns[:]
    run_folder = Path("./runs", cl_args.run_name)

    for continuous_column in continuous_columns:
        df_labels_train, scaler_path = scale_non_target_continuous_columns(
            df_labels_train, continuous_column, run_folder
        )
        df_labels_valid, _ = scale_non_target_continuous_columns(
            df_labels_valid, continuous_column, run_folder, scaler_path
        )

    df_labels_train = handle_missing_label_values(df_labels_train, cl_args, "train df")
    df_labels_valid = handle_missing_label_values(df_labels_valid, cl_args, "valid df")

    return df_labels_train, df_labels_valid


def scale_non_target_continuous_columns(
    df: pd.DataFrame, continuous_column: str, run_folder: Path, scaler_path: Path = None
) -> Tuple[pd.DataFrame, Path]:
    """
    Used to scale continuous columns in label dataframes. For training set, we fit a
    new scaler whereas for validation / testing we are expected to pass in
    `scaler_path`, which loads a fitted scaler and does the transformation on the
    continuous column in question.

    :param df: The dataframe to do the scaling on.
    :param continuous_column: The column of continuous values to do the fit and/or
    scaling on.
    :param run_folder: Folder for the current run, used to set up paths .
    :param scaler_path: If this is passed in, assume we are only transforming and load
    scaler from this path.
    :return:
    """

    def parse_colvals(column: pd.Series) -> np.ndarray:
        return column.values.astype(float).reshape(-1, 1)

    if not scaler_path:
        logger.debug("Fitting standard scaler to training df of shape %s.", df.shape)

        scaler_outpath = get_transformer_path(
            run_folder, continuous_column, "standard_scaler"
        )

        scaler = StandardScaler()
        scaler.fit(parse_colvals(df[continuous_column]))
        ensure_path_exists(scaler_outpath)
        joblib.dump(scaler, scaler_outpath)
        scaler_return_path = scaler_outpath

    else:
        logger.debug(
            "Transforming valid/test (shape %s) df with scaler fitted on training df.",
            df.shape,
        )
        scaler = joblib.load(scaler_path)
        scaler_return_path = scaler_path

    df[continuous_column] = scaler.transform(parse_colvals(df[continuous_column]))

    return df, scaler_return_path


def get_transformer_path(run_path: Path, transformer_name: str, suffix: str) -> Path:
    transformer_path = run_path / "transformers" / f"{transformer_name}_{suffix}.save"

    return transformer_path


def handle_missing_label_values(df: pd.DataFrame, cl_args, name="df"):
    if cl_args.embed_columns:
        missing_stats = _get_missing_stats_string(df, cl_args.embed_columns)
        logger.debug(
            "Replacing NaNs in embedding columns %s (counts: %s) in %s with 'NA'.",
            cl_args.embed_columns,
            missing_stats,
            name,
        )
        df[cl_args.embed_columns] = df[cl_args.embed_columns].fillna("NA")

    if cl_args.contn_columns:
        missing_stats = _get_missing_stats_string(df, cl_args.contn_columns)
        logger.debug(
            "Replacing NaNs in continuous columns %s (counts: %s) in %s with 0.",
            cl_args.contn_columns,
            missing_stats,
            name,
        )
        df[cl_args.contn_columns] = df[cl_args.contn_columns].fillna(0)

    return df


def _get_missing_stats_string(
    df: pd.DataFrame, columns_to_check: List[str]
) -> Dict[str, int]:
    missing_count_dict = {}
    for col in columns_to_check:
        missing_count_dict[col] = int(df[col].isnull().sum())

    return missing_count_dict
