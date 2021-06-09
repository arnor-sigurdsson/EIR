from dataclasses import dataclass
from functools import wraps, partial
from pathlib import Path
from typing import List, Tuple, Any, Union, Callable, Dict

import pandas as pd
import torch
from pandas import DataFrame


@dataclass
class ColumnOperation:
    """
    function: Function to apply on the label dataframe.
    """

    function: Callable[[DataFrame, str], DataFrame]
    function_args: Dict[str, Any]
    extra_columns_deps: Union[Tuple[str, ...], None] = ()
    only_apply_if_target: bool = False
    column_dtypes: Union[Dict[str, Any], None] = None


def streamline_df(
    df_func: Callable[[pd.DataFrame, Any], pd.DataFrame]
) -> Callable[[Any], pd.DataFrame]:
    @wraps(df_func)
    def wrapper(*args, df=None, column_name=None, **kwargs) -> pd.DataFrame:

        df = df_func(*args, df=df, column_name=column_name, **kwargs)

        return df

    return wrapper


@streamline_df
def placeholder_func(df, column_name):
    """
    Dummy function that we can pass to streamline_df so we still get it's functionality.
    """
    return df


def filter_ids_from_array_and_id_lists(
    ids_to_filter: List[str],
    id_list: List[str],
    arrays: Union[torch.Tensor, List[Path]],
    filter_type: str = "keep",
) -> Tuple[List[str], torch.Tensor]:
    """
    We use this function to perform filtering on the ID list and arrays, where these
    are assumed to be in the same order. This function is defined in the case where
    we want to dynamically filter out certain samples when setting up our dataloaders,
    e.g. we might want to fit on origin data with all "European" origin removed.

    :param ids_to_filter: Which IDs to filter on.
    :param id_list: A "master" list of IDs to perform the filtering operation on.
    :param arrays:  A "master" list of arrays to perform the filtering operation on.
    :param filter_type: Whether to keep or remove the IDs in `ids_to_filter`.
    :return: A tuple of (ids, arrays) after performing the filtering operation.
    """

    if filter_type not in ["keep", "skip"]:
        raise ValueError("Expecting either keep or skip for filter type.")

    keep_bool = filter_type == "keep"

    def condition(x):
        cond = x in ids_to_filter
        if keep_bool:
            return cond
        return not cond

    def cast_to_base_type(filtered_arrs):
        base_type = type(arrays)
        if base_type == torch.Tensor:
            return torch.stack(filtered_arrs)
        else:
            return base_type(filtered_arrs)

    filter_generator = (i for i in zip(id_list, arrays) if condition(i[0]))

    ids_filtered, arrays_filtered = zip(*filter_generator)

    return list(ids_filtered), cast_to_base_type(arrays_filtered)


def bucket_column(
    df: pd.DataFrame, column_name: str, n_buckets: int = 5, q_cut=False
) -> pd.DataFrame:
    """
    Splits continuous column into buckets for modelling with multi-class classification.

    :param df: Label dataframe in most cases.
    :param column_name: Column to bucket.
    :param n_buckets: Number of buckets.
    :param q_cut: Whether to do a quantile-based discretization with pd.qcut
    :return: Dataframe with ``column`` replaced with bucket-ized version.
    """
    cut_func = pd.qcut if q_cut else partial(pd.cut, include_lowest=True)
    df[column_name] = cut_func(df[column_name], n_buckets)

    return df


def get_low_count_ids(
    df: pd.DataFrame, column_name: str, threshold: int = 100
) -> List[str]:
    """
    Grabs those IDs where their counts are under a certain threshold. Why? We might
    want to restrict the number of classes during a multi-class experiment to only
    those that are above a certain threshold, i.e. it can happen that in 20k samples
    a certain class has <5 samples.

    Actually returns a ID list, so we can easily use it the output with
    `filter_ids_from_array_and_id_lists`.

    :param df: Dataframe to get counts on.
    :param column_name: Name of column to count on.
    :param threshold: Values under this threshold will be returned.
    :return: List of those IDs that have a value under the threshold.
    """
    counts = df[column_name].value_counts()
    low_count_ids = df[df.isin(counts.index[counts <= threshold]).values].index

    return list(low_count_ids)


def get_ids_from_values(
    df: pd.DataFrame, column_name: str, values_to_grab: List[Any]
) -> List[str]:
    """
    Grabs IDs matching certain values passed in. Why? We might want to manually filter
    out / keep some IDs matching a certain value in a given column.

    :param df: Dataframe to grab values from.
    :param column_name: Which columns to match values to.
    :param values_to_grab: Values to check in `transformer_name`.
    :return: List of IDs matching the values we're interested in.
    """

    ids = df[df[column_name].isin(values_to_grab)].index

    return list(ids)
