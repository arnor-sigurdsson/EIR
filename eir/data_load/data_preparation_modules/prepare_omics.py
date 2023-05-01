from pathlib import Path
from typing import Union, Sequence, Optional

import numpy as np
import torch

from eir.data_load.data_augmentation import make_random_omics_columns_missing
from eir.data_load.data_preparation_modules.common import _load_deeplake_sample
from eir.data_load.data_source_modules import deeplake_ops


def omics_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    subset_indices: Union[Sequence[int], None],
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        genotype_array_raw = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
    else:
        genotype_array_raw = np.load(str(data_pointer))

    if subset_indices is not None:
        genotype_array_raw = genotype_array_raw[:, subset_indices]

    genotype_array_raw_bool = genotype_array_raw.astype(bool)

    return genotype_array_raw_bool


def prepare_one_hot_omics_data(
    genotype_array: np.ndarray,
    na_augment_perc: float,
    na_augment_prob: float,
    test_mode: bool,
) -> torch.BoolTensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """

    tensor_bool = torch.BoolTensor(genotype_array).unsqueeze(0).detach().clone()

    if not test_mode and na_augment_perc > 0 and na_augment_prob > 0:
        tensor_bool = make_random_omics_columns_missing(
            omics_array=tensor_bool,
            percentage=na_augment_perc,
            probability=na_augment_prob,
        )

    return tensor_bool
