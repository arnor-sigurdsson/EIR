from pathlib import Path

import numpy as np
import torch

from eir.data_load.data_augmentation import (
    make_random_omics_columns_missing,
    shuffle_random_omics_columns,
)


def omics_load_wrapper(
    data_pointer: Path | str | int,
    subset_indices: np.ndarray | None,
) -> np.ndarray:
    assert isinstance(data_pointer, str | Path)
    genotype_array_raw = np.load(str(data_pointer))

    assert isinstance(genotype_array_raw, np.ndarray)

    if subset_indices is not None:
        genotype_array_raw = genotype_array_raw[:, subset_indices]

    genotype_array_raw_bool = genotype_array_raw.astype(bool)

    return genotype_array_raw_bool


def prepare_one_hot_omics_data(
    genotype_array: np.ndarray,
    na_augment_alpha: float,
    na_augment_beta: float,
    shuffle_augment_alpha: float,
    shuffle_augment_beta: float,
    test_mode: bool,
) -> torch.Tensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """

    tensor_bool = torch.BoolTensor(genotype_array).unsqueeze(0).detach().clone()

    if not test_mode and na_augment_alpha > 0 and na_augment_beta > 0:
        tensor_bool = make_random_omics_columns_missing(
            omics_array=tensor_bool,
            na_augment_alpha=na_augment_alpha,
            na_augment_beta=na_augment_beta,
        )

    if not test_mode and shuffle_augment_alpha > 0 and shuffle_augment_beta > 0:
        tensor_bool = shuffle_random_omics_columns(
            omics_array=tensor_bool,
            shuffle_augment_alpha=shuffle_augment_alpha,
            shuffle_augment_beta=shuffle_augment_beta,
        )

    assert tensor_bool.dtype == torch.bool
    return tensor_bool
