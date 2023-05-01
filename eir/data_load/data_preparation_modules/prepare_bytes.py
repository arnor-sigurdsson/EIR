from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch

from eir.data_load.data_preparation_modules.common import (
    _load_deeplake_sample,
    process_tensor_to_length,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.setup.input_setup_modules.setup_bytes import BytesInputInfo


def bytes_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    dtype: str,
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        bytes_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        ).astype(dtype=dtype)
    else:
        bytes_data = np.fromfile(file=data_pointer, dtype=dtype)

    return bytes_data


def prepare_bytes_data(
    bytes_input_object: "BytesInputInfo", bytes_data: np.ndarray, test_mode: bool
) -> torch.Tensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """
    bio = bytes_input_object

    sampling_strat = bio.input_config.input_type_info.sampling_strategy_if_longer
    if test_mode:
        sampling_strat = "from_start"

    bytes_tensor = torch.LongTensor(bytes_data).detach().clone()

    padding_value = bio.vocab.get("<pad>", 0)
    cur_bytes_padded = process_tensor_to_length(
        tensor=bytes_tensor,
        max_length=bio.input_config.input_type_info.max_length,
        sampling_strategy_if_longer=sampling_strat,
        padding_value=padding_value,
    )

    return cur_bytes_padded
