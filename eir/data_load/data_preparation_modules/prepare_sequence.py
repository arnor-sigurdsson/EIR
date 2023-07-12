from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from eir.data_load.data_preparation_modules.common import (
    _load_deeplake_sample,
    process_tensor_to_length,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    al_encode_funcs,
    get_sequence_split_function,
)
from eir.setup.schemas import SequenceInputDataConfig


def sequence_load_wrapper(
    data_pointer: Union[Path, int, np.ndarray],
    input_source: str,
    split_on: Optional[str],
    encode_func: al_encode_funcs,
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
    assert isinstance(input_type_info, SequenceInputDataConfig)

    cur_tokens_as_tensor = torch.LongTensor(cur_file_content_tokenized).detach().clone()

    sampling_strategy = input_type_info.sampling_strategy_if_longer
    if test_mode:
        sampling_strategy = "from_start"

    padding_token = getattr(sio.tokenizer, "pad_token", "<pad>")
    padding_token_parsed = parse_padding_token_encode_func_input(
        split_on=input_type_info.split_on, padding_token=padding_token
    )
    padding_value = sio.encode_func(padding_token_parsed)[0]
    cur_tokens_padded = process_tensor_to_length(
        tensor=cur_tokens_as_tensor,
        max_length=sio.computed_max_length,
        sampling_strategy_if_longer=sampling_strategy,
        padding_value=padding_value,
    )

    return cur_tokens_padded


def parse_padding_token_encode_func_input(
    split_on: Optional[str], padding_token: str
) -> Sequence[str] | str:
    parsed_token: Sequence[str] | str

    if split_on is None:
        parsed_token = padding_token
    else:
        parsed_token = [padding_token]

    return parsed_token
