import typing
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Hashable, Literal, Sequence

from eir.setup import schemas
from eir.setup.input_setup_modules.common import get_default_sequence_specials


@dataclass
class ComputedBytesInputInfo:
    input_config: schemas.InputConfig
    vocab: OrderedDict
    computed_max_length: int


def set_up_bytes_input_for_training(
    input_config: schemas.InputConfig, add_specials: bool = True, *args, **kwargs
) -> ComputedBytesInputInfo:
    specials: list[str] = []
    if add_specials:
        specials = get_default_sequence_specials()

    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.ByteInputDataConfig)
    bytes_vocab = build_bytes_vocab(
        byte_encoding=input_type_info.byte_encoding, specials=specials
    )

    if not isinstance(input_type_info.max_length, int):
        raise ValueError(
            "Max length for bytes input only supports raw int values currently."
        )

    bytes_input_info = ComputedBytesInputInfo(
        input_config=input_config,
        vocab=bytes_vocab,
        computed_max_length=input_type_info.max_length,
    )

    return bytes_input_info


def build_bytes_vocab(
    byte_encoding: Literal["uint8"], specials: Sequence[Hashable] = tuple()
) -> typing.OrderedDict[int | Hashable, int]:
    bytes_vocab: typing.OrderedDict[int | Hashable, int] = OrderedDict()

    encoding_to_num_tokens_map = _get_encoding_to_num_tokens_map()
    num_tokens = encoding_to_num_tokens_map[byte_encoding]

    for token in range(num_tokens):
        bytes_vocab[token] = token

    base_num_tokens = len(bytes_vocab)
    for idx, special in enumerate(specials):
        bytes_vocab[special] = base_num_tokens + idx

    return bytes_vocab


def _get_encoding_to_num_tokens_map() -> Dict[str, int]:
    mapping = {"uint8": 256}
    return mapping
