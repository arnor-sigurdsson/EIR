from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Union, Callable, Generator, Set, TYPE_CHECKING

import deeplake

from eir.data_load.data_source_modules.common_utils import add_id_to_samples

if TYPE_CHECKING:
    from eir.data_load.datasets import Sample


@lru_cache()
def is_deeplake_dataset(data_source: str) -> bool:
    if Path(data_source, "dataset_meta.json").exists():
        return True

    return False


@lru_cache()
def load_deeplake_dataset(data_source: str) -> deeplake.Dataset:
    dataset = deeplake.load(
        path=data_source,
        read_only=True,
        verbose=False,
    )
    if "ID" not in dataset.tensors.keys():
        raise ValueError(
            f"DeepLake dataset at {data_source} does not have an ID tensor. "
            f"Please add one to the dataset."
        )
    return dataset


def get_deeplake_input_source_iterable(
    deeplake_dataset: deeplake.Dataset, inner_key: str
) -> Generator[deeplake.Tensor, None, None]:
    def _is_empty(x: deeplake.Tensor) -> bool:
        return x.shape[0] == 0

    for deeplake_sample in deeplake_dataset[inner_key]:

        if _is_empty(x=deeplake_sample):
            continue

        yield deeplake_sample


def add_deeplake_data_to_samples(
    input_source: str,
    input_name: str,
    deeplake_input_inner_key: str,
    samples: DefaultDict[str, "Sample"],
    data_loading_hook: Callable,
    ids_to_keep: Union[None, Set[str]],
) -> DefaultDict[str, "Sample"]:
    """
    For normal files in a folder, this is holding the paths, which we can think of as
    as unique pointer to the sample data, for this source.

    In the DeepLake case, this is an integer index pointing to the sample.
    """
    assert deeplake_input_inner_key is not None, "Deeplake input inner key is None"

    def _is_empty(x: deeplake.Tensor) -> bool:
        return x.shape[0] == 0

    deeplake_ds = load_deeplake_dataset(data_source=input_source)
    if deeplake_input_inner_key not in deeplake_ds.tensors.keys():
        raise ValueError(
            f"Input key {deeplake_input_inner_key} not found in deeplake dataset "
            f"{input_source}. Available inputs in deeplake dataset are "
            f"{deeplake_ds.tensors.keys()}."
        )

    for deeplake_sample in deeplake_ds:

        cur_input_from_sample = deeplake_sample[deeplake_input_inner_key]
        if _is_empty(x=cur_input_from_sample):
            continue

        sample_id = deeplake_sample["ID"].numpy().item()

        if ids_to_keep is not None and sample_id not in ids_to_keep:
            continue

        sample_data_pointer = deeplake_sample.index.values[0].value
        sample_data = data_loading_hook(sample_data_pointer)

        samples = add_id_to_samples(samples=samples, sample_id=sample_id)
        samples[sample_id].inputs[input_name] = sample_data

    return samples
