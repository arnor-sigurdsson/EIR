from pathlib import Path
from typing import (
    Union,
    Sequence,
    Generator,
    Tuple,
    Any,
    Iterable,
    Callable,
    DefaultDict,
    Optional,
    Set,
    Iterator,
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger

from eir.data_load.data_source_modules.common_utils import add_id_to_samples
from eir.data_load.label_setup import get_file_path_iterator
from eir.setup.input_setup import get_sequence_split_function
from tqdm import tqdm

if TYPE_CHECKING:
    from eir.data_load.datasets import Sample


logger = get_logger(name=__name__)


def get_file_sample_id_iterator(
    data_source: str, ids_to_keep: Union[None, Sequence[str]]
) -> Generator[Tuple[Any, str], None, None]:
    def _id_from_filename(file: Path) -> str:
        return file.stem

    def _filter_ids_callable(item, sample_id):
        if sample_id in ids_to_keep:
            return True
        return False

    base_file_iterator = get_file_path_iterator(
        data_source=Path(data_source), validate=False
    )

    sample_id_and_file_iterator = _get_sample_id_data_iterator(
        base_iterator=base_file_iterator, id_callable=_id_from_filename
    )

    if ids_to_keep:
        final_iterator = _get_filter_iterator(
            base_iterator=sample_id_and_file_iterator,
            filter_callable=_filter_ids_callable,
        )
    else:
        final_iterator = sample_id_and_file_iterator

    yield from final_iterator


def _get_sample_id_data_iterator(
    base_iterator: Iterable[str], id_callable: Callable
) -> Generator[Tuple[Any, str], None, None]:
    for item in base_iterator:
        sample_id = id_callable(item)
        yield item, sample_id


def _get_filter_iterator(base_iterator, filter_callable) -> Generator[Any, None, None]:
    for item in base_iterator:
        if filter_callable(*item):
            yield item


def get_file_sample_id_iterator_basic(
    data_source: str,
    ids_to_keep: Union[None, Set[str]],
) -> Generator[Tuple[Any, str], None, None]:

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


def add_sequence_data_from_csv_to_samples(
    input_object: str,
    samples: DefaultDict[str, "Sample"],
    ids_to_keep: Union[None, Set[str]],
    split_on: str,
    encode_func: Callable,
    source_name: str = "CSV File Data",
) -> DefaultDict[str, "Sample"]:

    logger.info(
        "Loading sequence data from CSV file %s. Note that this will "
        "load all the sequence data into memory.",
        input_object,
    )

    split_func = get_sequence_split_function(split_on=split_on)
    csv_sequence_iterator = get_csv_id_sequence_iterator(
        data_source=input_object, ids_to_keep=ids_to_keep
    )
    file_iterator_tqdm = tqdm(csv_sequence_iterator, desc=source_name)

    for sample_id, sequence in file_iterator_tqdm:

        samples = add_id_to_samples(samples=samples, sample_id=sample_id)

        sequence_split = split_func(sequence)
        sequence_encoded = np.array(encode_func(sequence_split))

        samples[sample_id].inputs[source_name] = sequence_encoded

    return samples


def get_csv_id_sequence_iterator(
    data_source: str, ids_to_keep: Optional[Set[str]] = None
) -> Iterator[Tuple[str, str]]:

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

        yield row.Index, cur_seq
