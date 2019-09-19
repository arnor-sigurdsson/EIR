from argparse import Namespace
from pathlib import Path
from typing import Tuple, Dict, Union, List

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from human_origins_supervised.data_load.common_ops import ColumnOperation

try:
    from human_origins_supervised.data_load import COLUMN_OPS
except ModuleNotFoundError:
    COLUMN_OPS = {}

from aislib.misc_utils import get_logger

logger = get_logger(__name__)

# Type Aliases
al_all_column_ops = Dict[str, Tuple[ColumnOperation, ...]]
al_train_val_dfs = Tuple[pd.DataFrame, pd.DataFrame]
al_label_dict = Dict[str, Dict[str, Union[str, float]]]


def get_extra_columns(
    label_column: str, all_column_ops: al_all_column_ops
) -> List[str]:
    extra_columns_flat = []
    if label_column in all_column_ops:
        cur_ops = all_column_ops.get(label_column)
        extra_columns = [i.extra_columns_deps for i in cur_ops]
        extra_columns_flat = list(
            column for column_deps in extra_columns for column in column_deps
        )

    return extra_columns_flat


def load_label_df(
    label_fpath: Path,
    label_column: str,
    ids_to_keep: Tuple[str, ...] = (),
    extra_columns: Tuple[str, ...] = (),
) -> pd.DataFrame:

    logger.debug("Reading in labelfile: %s", label_fpath)
    df_labels = pd.read_csv(
        label_fpath,
        usecols=["ID", label_column] + list(extra_columns),
        dtype={"ID": str},
    )

    df_labels = df_labels.set_index("ID")

    if ids_to_keep:
        no_labels = df_labels.shape[0]
        df_labels = df_labels[df_labels.index.isin(ids_to_keep)]
        no_dropped = no_labels - df_labels.shape[0]
        logger.debug(
            "Removed %d file IDs from label based on IDs present in data folder.",
            no_dropped,
        )

    return df_labels


def parse_label_df(df, column_ops: al_all_column_ops) -> pd.DataFrame:
    """
    We want to be able to dynamically apply various operations to different columns
    in the label file (e.g. different operations for creating obesity labels or parsing
    country of origin).

    :param df: Dataframe to perform processing on.
    :param column_ops: A dictionarity of colum names, where each value is a list
    of tuples, where each tuple is a callable as the first element and the callable's
    arguments as the second element.
    :return: Parsed dataframe.
    """

    for column_name, ops_funcs in column_ops.items():
        if column_name in df.columns:
            for column_op in ops_funcs:
                func, args_dict = column_op.function, column_op.args
                logger.debug(
                    "Applying func %s with args %s to column in pre-processing.",
                    func,
                    args_dict,
                )
                logger.debug("Shape before: %s", df.shape)
                df = func(df=df, column_name=column_name, **args_dict)
                logger.debug("Shape after: %s", df.shape)
    return df


def label_df_parse_wrapper(cl_args: Namespace) -> pd.DataFrame:
    all_ids = tuple(i.stem for i in Path(cl_args.data_folder).iterdir())

    extra_label_parsing_cols = get_extra_columns(cl_args.label_column, COLUMN_OPS)
    extra_embed_cols = cl_args.embed_columns
    all_extra_cols = extra_label_parsing_cols + extra_embed_cols

    df_labels = load_label_df(
        cl_args.label_file, cl_args.label_column, all_ids, all_extra_cols
    )
    df_labels = parse_label_df(df_labels, COLUMN_OPS)

    # remove columns only used for parsing, so only keep actual label column
    # and extra embedding columns
    df_labels = df_labels.drop(extra_label_parsing_cols, axis=1)

    return df_labels


def split_df(df: pd.DataFrame, valid_size: Union[int, float]) -> al_train_val_dfs:
    train_ids, valid_ids = train_test_split(list(df.index), test_size=valid_size)

    df_labels_train = df.loc[df.index.intersection(train_ids)]
    df_labels_valid = df.loc[df.index.intersection(valid_ids)]
    assert len(df_labels_train) + len(df_labels_valid) == len(df)

    return df_labels_train, df_labels_valid


def scale_regression_labels(
    df_labels_train: pd.DataFrame,
    df_labels_valid: pd.DataFrame,
    reg_col: str,
    runs_folder: Path,
) -> al_train_val_dfs:
    """
    Used to scale regression column.
    """

    def parse_colvals(column):
        return column.values.astype(float).reshape(-1, 1)

    logger.debug(
        "Applying standard scaling to column %s in train and valid sets.", reg_col
    )

    scaler = StandardScaler()
    scaler.fit(parse_colvals(df_labels_train[reg_col]))
    scaler_outpath = runs_folder / "standard_scaler.save"
    joblib.dump(scaler, scaler_outpath)

    df_labels_train[reg_col] = scaler.transform(parse_colvals(df_labels_train[reg_col]))
    df_labels_valid[reg_col] = scaler.transform(parse_colvals(df_labels_valid[reg_col]))

    return df_labels_train, df_labels_valid


def process_train_and_label_dfs(
    cl_args, df_labels_train, df_labels_valid
) -> al_train_val_dfs:

    if cl_args.model_task == "reg":
        runs_folder = Path("./runs", cl_args.run_name)
        df_labels_train, df_labels_valid = scale_regression_labels(
            df_labels_train, df_labels_valid, cl_args.label_column, runs_folder
        )

    return df_labels_train, df_labels_valid


def set_up_train_and_valid_labels(
    cl_args: Namespace
) -> Tuple[al_label_dict, al_label_dict]:
    """
    Splits and does split based processing (e.g. scaling validation set with training
    set for regression) on the labels.
    """

    df_labels = label_df_parse_wrapper(cl_args)

    df_labels_train, df_labels_valid = split_df(df_labels, cl_args.valid_size)

    df_labels_train, df_labels_valid = process_train_and_label_dfs(
        cl_args, df_labels_train, df_labels_valid
    )

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")
    return train_labels_dict, valid_labels_dict
