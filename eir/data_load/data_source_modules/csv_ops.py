from dataclasses import dataclass
from functools import wraps
from typing import Tuple, Any, Union, Callable, Dict

import pandas as pd
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
