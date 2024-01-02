from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from eir.data_load.data_source_modules.deeplake_ops import (
    get_deeplake_input_source_iterable,
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.data_load.label_setup import get_file_path_iterator


@dataclass
class DataDimensions:
    channels: int
    height: int
    width: int
    extra_dims: tuple[int, ...] = tuple()

    def num_elements(self) -> int:
        base = self.channels * self.height * self.width
        return int(base * np.prod(self.extra_dims))

    def full_shape(self) -> Tuple[int, ...]:
        return (self.channels, self.height, self.width) + self.extra_dims


def get_data_dimension_from_data_source(
    data_source: Path,
    deeplake_inner_key: Optional[str] = None,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    if is_deeplake_dataset(data_source=str(data_source)):
        assert deeplake_inner_key is not None, data_source
        deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
        deeplake_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        shape = next(deeplake_iter).shape
    else:
        iterator = get_file_path_iterator(data_source=data_source)
        path = next(iterator)
        shape = np.load(file=path).shape

    extra_dims: tuple[int, ...] = tuple()
    if len(shape) == 1:
        channels, height, width = 1, 1, shape[0]
    elif len(shape) == 2:
        channels, height, width = 1, shape[0], shape[1]
    elif len(shape) == 3:
        channels, height, width = shape
    else:
        channels, height, width = shape[0], shape[1], shape[2]
        extra_dims = shape[3:]

    return DataDimensions(
        channels=channels, height=height, width=width, extra_dims=extra_dims
    )


def get_dtype_from_data_source(
    data_source: Path,
    deeplake_inner_key: Optional[str] = None,
) -> np.dtype:
    if is_deeplake_dataset(data_source=str(data_source)):
        assert (
            deeplake_inner_key is not None
        ), "Deeplake inner key is required for Deeplake datasets"
        deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
        deeplake_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        data_type = next(deeplake_iter).dtype
    else:
        iterator = get_file_path_iterator(data_source=data_source)
        path = next(iterator)
        data_type = np.load(file=path).dtype

    return data_type


def get_default_sequence_specials() -> List[str]:
    default_specials = ["<bos>", "<unk>", "<mask>", "<pad>", "<eos>"]
    return default_specials
