from typing import Dict, Tuple

import torch
from torch._C._nn import pad

from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.train_utils.evaluation_handlers.evaluation_handlers_utils import (
    SpecialTokens,
    get_special_tokens,
)


def prepare_sequence_input_for_sequence_output(
    input_object: ComputedSequenceInputInfo,
    cur_seq: torch.Tensor,
    input_name: str,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    assert input_object.tokenizer is not None
    special_tokens = get_special_tokens(
        tokenizer=input_object.tokenizer,
        vocab=input_object.vocab,
    )

    cur_seq, cur_target = sample_autoregressive_batch(
        batch_tensor=cur_seq,
        batch_size=cur_seq.shape[0],
        special_tokens=special_tokens,
    )
    assert cur_seq.shape == cur_target.shape
    assert cur_seq.dim() == 2
    assert cur_seq.shape[1] == input_object.computed_max_length

    cur_seq = cur_seq.to(device=device)

    cur_target_dict = {input_name: cur_target.to(device=device)}

    return cur_seq, cur_target_dict


def sample_autoregressive_batch(
    batch_tensor: torch.Tensor,
    batch_size: int,
    special_tokens: SpecialTokens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The reason for padding with the BOS token is that the tensor we get here
    is already at max_length. If we had e.g. a full, long sequence, we could
    simply slice that directly (+1 for the target), but here we need to pad
    the input at the beginning, opting for a BOS token.
    """
    st = special_tokens

    inputs = []
    targets = []

    batch_tensor = pad_batch_with_bos(
        batch_tensor=batch_tensor,
        bos_value=st.bos_idx,
    )

    for idx in range(batch_size):
        cur_sample = batch_tensor[idx]

        cur_sample = _switch_first_pad_with_eos(
            sample=cur_sample,
            pad_value=st.pad_idx,
            eos_value=st.eos_idx,
        )

        cur_target = cur_sample[1:]
        cur_sample = cur_sample[:-1]

        inputs.append(cur_sample)
        targets.append(cur_target)

    inputs_tensor = torch.stack(tensors=inputs)
    target_tensor = torch.stack(tensors=targets)
    return inputs_tensor, target_tensor


def _switch_first_pad_with_eos(
    sample: torch.Tensor,
    pad_value: int,
    eos_value: int,
) -> torch.Tensor:
    first_pad_match = (sample == pad_value).nonzero(as_tuple=False)
    if len(first_pad_match) != 0:
        first_pad_index = first_pad_match[0]
        sample[first_pad_index] = eos_value

    return sample


def pad_batch_with_bos(
    batch_tensor: torch.Tensor,
    bos_value: int,
) -> torch.Tensor:
    left_padding = 1
    batch_tensor = pad(input=batch_tensor, pad=[left_padding, 0], value=bos_value)

    return batch_tensor
