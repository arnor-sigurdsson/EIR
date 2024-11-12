import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, DefaultDict, Generator, Set, Union

warnings.filterwarnings("ignore", message=".*newer version of deeplake.*")

import deeplake
from tqdm import tqdm

from eir.data_load.data_source_modules.common_utils import add_id_to_samples

if TYPE_CHECKING:
    from eir.data_load.data_utils import Sample


@lru_cache()
def is_deeplake_dataset(data_source: str) -> bool:
    if deeplake.exists(data_source):
        return True
    return False


@lru_cache()
def load_deeplake_dataset(data_source: str) -> deeplake.ReadOnlyDataset:
    dataset = deeplake.open_read_only(
        url=data_source,
        token=None,
    )

    columns = {col.name for col in dataset.schema.columns}
    if "ID" not in columns:
        raise ValueError(
            f"DeepLake dataset at {data_source} does not have an ID column. "
            f"Please add one to the dataset."
        )
    return dataset


def get_deeplake_input_source_iterable(
    deeplake_dataset: deeplake.ReadOnlyDataset, inner_key: str
) -> Generator[Union[deeplake.Column, deeplake.ColumnView], None, None]:
    columns = {col.name for col in deeplake_dataset.schema.columns}

    existence_col = f"{inner_key}_exists"
    for row in deeplake_dataset:
        if is_deeplake_sample_missing(
            row=row,
            existence_col=existence_col,
            columns=columns,
        ):
            continue

        value = row[inner_key]

        yield value


def add_deeplake_data_to_samples(
    input_source: str,
    input_name: str,
    deeplake_input_inner_key: str,
    samples: DefaultDict[str, "Sample"],
    data_loading_hook: Callable,
    ids_to_keep: Union[None, Set[str]],
) -> DefaultDict[str, "Sample"]:
    assert deeplake_input_inner_key is not None, "Deeplake input inner key is None"

    deeplake_ds = load_deeplake_dataset(data_source=input_source)

    columns = {col.name for col in deeplake_ds.schema.columns}
    if deeplake_input_inner_key not in columns:
        raise ValueError(
            f"Input key {deeplake_input_inner_key} not found in deeplake dataset "
            f"{input_source}. Available columns are: {columns}."
        )

    total = len(ids_to_keep) if ids_to_keep is not None else len(deeplake_ds)
    pbar = tqdm(total=total, desc=input_name)

    existence_col = f"{deeplake_input_inner_key}_exists"
    for row in deeplake_ds:
        if is_deeplake_sample_missing(
            row=row,
            existence_col=existence_col,
            columns=columns,
        ):
            continue

        sample_id = row["ID"]

        if ids_to_keep is not None and sample_id not in ids_to_keep:
            continue

        sample_data_pointer = row.row_id
        sample_data = data_loading_hook(sample_data_pointer)

        samples = add_id_to_samples(samples=samples, sample_id=sample_id)
        samples[sample_id].inputs[input_name] = sample_data

        pbar.update(1)

    return samples


def is_deeplake_sample_missing(
    row: deeplake.RowView,
    existence_col: str,
    columns: set[str],
) -> bool:
    if existence_col in columns:
        is_missing = not row[existence_col].item()
        return is_missing

    return False
