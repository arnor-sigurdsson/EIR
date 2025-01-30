from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from eir.data_load.label_setup import get_file_path_iterator
from eir.setup.input_setup_modules.setup_sequence import get_sequence_split_function
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(name=__name__)


def get_file_sample_id_iterator_basic(
    data_source: str,
    ids_to_keep: None | set[str],
) -> Generator[tuple[str, Path]]:
    base_file_iterator = get_file_path_iterator(
        data_source=Path(data_source), validate=False
    )

    for file in base_file_iterator:
        sample_id = file.stem

        if ids_to_keep:
            if sample_id in ids_to_keep:
                yield sample_id, file
        else:
            yield sample_id, file


def add_sequence_data_from_csv_to_df(
    input_source: str,
    input_df: pl.DataFrame,
    ids_to_keep: None | set[str],
    split_on: str | None,
    encode_func: Callable,
    input_name: str = "CSV File Data",
) -> pl.DataFrame:
    logger.info(
        "Loading sequence data from CSV file %s. Note that this will "
        "load all the sequence data into memory.",
        input_source,
    )

    split_func = get_sequence_split_function(split_on=split_on)
    csv_sequence_iterator = get_csv_id_sequence_iterator(
        data_source=input_source,
        ids_to_keep=ids_to_keep,
    )

    ids = []
    sequences = []

    for sample_id, sequence in tqdm(csv_sequence_iterator, desc=input_name):
        sequence_split = split_func(sequence)
        sequence_encoded = encode_func(sequence_split)

        if isinstance(sequence_encoded, np.ndarray):
            sequence_encoded = sequence_encoded.tolist()

        ids.append(sample_id)
        sequences.append(sequence_encoded)

    if not ids:
        return input_df

    sequence_df = pl.DataFrame(
        {
            "ID": pl.Series(name="ID", values=ids, dtype=pl.Utf8),
            input_name: pl.Series(
                name=input_name, values=sequences, dtype=pl.List(pl.Int64)
            ),
        }
    )

    if input_df.height == 0:
        return sequence_df
    return input_df.join(sequence_df, on="ID", how="full", coalesce=True)


def get_csv_id_sequence_iterator(
    data_source: str, ids_to_keep: set[str] | None = None
) -> Iterator[tuple[str, str]]:
    df = pd.read_csv(data_source, index_col="ID", dtype={"ID": str})

    if "Sequence" not in df.columns:
        raise ValueError(
            f"Expected to find a 'Sequence' column in {data_source}, but didn't."
        )

    if ids_to_keep:
        df = df[df.index.isin(set(ids_to_keep))]

    for row in df.itertuples():
        cur_seq = row.Sequence
        if pd.isna(cur_seq):
            cur_seq = ""

        cur_index = row.Index
        assert isinstance(cur_index, str)
        assert isinstance(cur_seq, str)

        yield cur_index, cur_seq
