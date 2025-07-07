from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eir.data_load.label_setup import get_file_path_iterator


@dataclass
class DataDimensions:
    channels: int
    height: int
    width: int
    extra_dims: tuple[int, ...] = ()
    original_shape: tuple[int, ...] | None = None

    def num_elements(self) -> int:
        base = self.channels * self.height * self.width
        return int(base * np.prod(self.extra_dims))

    def full_shape(self) -> tuple[int, ...]:
        return tuple([self.channels, self.height, self.width] + list(self.extra_dims))


def get_data_dimension_from_data_source(
    data_source: Path,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    iterator = get_file_path_iterator(data_source=data_source)
    path = next(iterator)
    shape = np.load(file=path).shape

    extra_dims: tuple[int, ...] = ()
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
        channels=channels,
        height=height,
        width=width,
        extra_dims=extra_dims,
        original_shape=tuple(shape),
    )


def get_dtype_from_data_source(
    data_source: Path,
) -> np.dtype:
    iterator = get_file_path_iterator(data_source=data_source)
    path = next(iterator)
    data_type = np.load(file=path).dtype

    return data_type


def get_default_sequence_specials() -> list[str]:
    default_specials = ["<bos>", "<unk>", "<mask>", "<pad>", "<eos>"]
    return default_specials
