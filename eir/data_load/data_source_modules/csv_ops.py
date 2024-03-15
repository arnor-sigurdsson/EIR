from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import pandas as pd


class DataFrameFunctionProtocol(Protocol):
    def __call__(
        self,
        *args: Any,
        df: pd.DataFrame,
        column_name: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame: ...


@dataclass
class ColumnOperation:
    """
    function: Function to apply on the label dataframe.
    """

    function: DataFrameFunctionProtocol
    function_args: Dict[str, Any]
    extra_columns_deps: Union[Tuple[str, ...], None] = ()
    only_apply_if_target: bool = False
    column_dtypes: Union[Dict[str, Any], None] = None


def streamline_df(df_func: DataFrameFunctionProtocol) -> Callable[[Any], pd.DataFrame]:
    @wraps(df_func)
    def wrapper(*args, df=None, column_name=None, **kwargs) -> pd.DataFrame:
        df = df_func(*args, df=df, column_name=column_name, **kwargs)

        return df

    return wrapper
