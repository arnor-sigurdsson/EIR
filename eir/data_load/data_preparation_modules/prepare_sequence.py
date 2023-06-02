from pathlib import Path
from typing import Union, Callable, Sequence, List, Optional

import numpy as np
import torch

from eir.data_load.data_preparation_modules.common import (
    _load_deeplake_sample,
    process_tensor_to_length,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    get_sequence_split_function,
)

from eir.setup.schemas import (
    ByteInputDataConfig,
)


def sequence_load_wrapper(
    data_pointer: Union[Path, int, np.ndarray],
    input_source: str,
    split_on: str,
    encode_func: Callable[[Sequence[str]], List[int]],
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:
    """
    In the case of .csv input sources, we have already loaded and tokenized the data.
    """

    split_func = get_sequence_split_function(split_on=split_on)
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        assert isinstance(data_pointer, int)
        text_as_np_array = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        content = text_as_np_array[0]
    elif input_source.endswith(".csv"):
        assert isinstance(data_pointer, np.ndarray)
        return data_pointer
    else:
        assert isinstance(data_pointer, Path)
        content = load_sequence_from_disk(sequence_file_path=data_pointer)

    file_content_split = split_func(content)
    file_content_encoded = encode_func(file_content_split)
    sequence_tokenized = np.array(file_content_encoded)

    return sequence_tokenized


def load_sequence_from_disk(sequence_file_path: Path) -> str:
    with open(sequence_file_path, "r") as infile:
        return infile.read().strip()


def prepare_sequence_data(
    sequence_input_object: "ComputedSequenceInputInfo",
    cur_file_content_tokenized: np.ndarray,
    test_mode: bool,
) -> torch.Tensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """

    sio = sequence_input_object
    input_type_info = sio.input_config.input_type_info
    assert isinstance(input_type_info, ByteInputDataConfig)

    cur_tokens_as_tensor = torch.LongTensor(cur_file_content_tokenized).detach().clone()

    sampling_strategy = input_type_info.sampling_strategy_if_longer
    if test_mode:
        sampling_strategy = "from_start"

    padding_token = getattr(sio.tokenizer, "pad_token", "<pad>")
    padding_value = sio.encode_func([padding_token])[0]
    cur_tokens_padded = process_tensor_to_length(
        tensor=cur_tokens_as_tensor,
        max_length=sio.computed_max_length,
        sampling_strategy_if_longer=sampling_strategy,
        padding_value=padding_value,
    )

    return cur_tokens_padded
