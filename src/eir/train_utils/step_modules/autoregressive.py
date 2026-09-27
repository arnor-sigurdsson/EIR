import torch
from torch._C._nn import pad

from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    SpecialTokens,
    get_special_tokens,
)


def prepare_sequence_input_for_sequence_output(
    input_object: ComputedSequenceInputInfo,
    cur_seq: torch.Tensor,
    input_name: str,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

    cur_seq = cur_seq

    cur_target_dict = {input_name: cur_target}

    return cur_seq, cur_target_dict


def sample_autoregressive_batch(
    batch_tensor: torch.Tensor,
    batch_size: int,
    special_tokens: SpecialTokens,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The reason for padding with the BOS token is that the tensor we get here
    is already at max_length. If we had e.g. a full, long sequence, we could
    simply slice that directly (+1 for the target), but here we need to pad
    the input at the beginning, opting for a BOS token.
    """
    st = special_tokens

    # note this returns a new tensor, so we don't modify the original batch_tensor
    batch_tensor_w_pad = pad_batch_with_bos(
        batch_tensor=batch_tensor,
        bos_value=st.bos_idx,
    )

    _switch_first_pad_with_eos(
        batch_tensor_w_pad=batch_tensor_w_pad, pad_idx=st.pad_idx, eos_idx=st.eos_idx
    )

    inputs_tensor = batch_tensor_w_pad[:, :-1]
    target_tensor = batch_tensor_w_pad[:, 1:]

    return inputs_tensor, target_tensor


def _switch_first_pad_with_eos(
    batch_tensor_w_pad: torch.Tensor, pad_idx: int, eos_idx: int
) -> torch.Tensor:
    is_pad = batch_tensor_w_pad == pad_idx
    first_pad = is_pad.int().argmax(dim=1)
    has_pad = is_pad.any(dim=1)

    pad_index = first_pad[has_pad]
    batch_tensor_w_pad[has_pad, pad_index] = eos_idx

    return batch_tensor_w_pad


def pad_batch_with_bos(
    batch_tensor: torch.Tensor,
    bos_value: int,
) -> torch.Tensor:
    left_padding = 1
    batch_tensor = pad(input=batch_tensor, pad=[left_padding, 0], value=bos_value)

    return batch_tensor
