from typing import Literal

import torch
from torch.nn.functional import pad


def process_tensor_to_length(
    tensor: torch.Tensor,
    max_length: int,
    sampling_strategy_if_longer: Literal["from_start", "uniform"],
    padding_value: int = 0,
) -> torch.Tensor:
    tensor_length = len(tensor)

    if tensor_length > max_length:
        if sampling_strategy_if_longer == "from_start":
            truncated_tensor = tensor[:max_length]
            return truncated_tensor

        if sampling_strategy_if_longer == "uniform":
            uniformly_sampled_tensor = _sample_sequence_uniform(
                tensor=tensor, tensor_length=tensor_length, max_length=max_length
            )
            return uniformly_sampled_tensor

    right_padding = max_length - tensor_length
    padded_tensor = pad(input=tensor, pad=[0, right_padding], value=padding_value)

    return padded_tensor


def _sample_sequence_uniform(
    tensor: torch.Tensor, tensor_length: int, max_length: int
) -> torch.Tensor:
    random_index_start = torch.randperm(max(1, tensor_length - max_length))[0]
    random_index_end = random_index_start + max_length
    return tensor[random_index_start:random_index_end]
