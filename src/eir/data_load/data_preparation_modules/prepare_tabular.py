import polars as pl

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def add_tabular_data_to_df(
    df_tabular: pl.DataFrame,
    input_df: pl.DataFrame,
    ids_to_keep: set[str] | None = None,
    source_name: str = "Tabular Data",
) -> pl.DataFrame:
    logger.debug(f"Adding tabular data from {source_name}")

    if ids_to_keep is not None:
        df_tabular = df_tabular.filter(pl.col("ID").is_in(ids_to_keep))

    rename_expr = []
    for col in df_tabular.columns:
        if col == "ID":
            rename_expr.append(pl.col("ID"))
        else:
            new_name = f"{source_name}__{col}"
            rename_expr.append(pl.col(col).alias(new_name))

    df_tabular_renamed = df_tabular.select(rename_expr)

    if input_df.height == 0:
        return df_tabular_renamed
    return input_df.join(df_tabular_renamed, on="ID", how="full", coalesce=True)
