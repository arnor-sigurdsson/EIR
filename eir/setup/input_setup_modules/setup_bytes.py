from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Sequence, Hashable, Dict

from eir.setup import schemas
from eir.setup.input_setup_modules.common import get_default_sequence_specials


@dataclass
class BytesInputInfo:
    input_config: schemas.InputConfig
    vocab: OrderedDict
    computed_max_length: int


def set_up_bytes_input_for_training(
    input_config: schemas.InputConfig, add_specials: bool = True, *args, **kwargs
) -> BytesInputInfo:
    specials = tuple()
    if add_specials:
        specials = get_default_sequence_specials()

    bytes_vocab = build_bytes_vocab(
        byte_encoding=input_config.input_type_info.byte_encoding, specials=specials
    )

    bytes_input_info = BytesInputInfo(
        input_config=input_config,
        vocab=bytes_vocab,
        computed_max_length=input_config.input_type_info.max_length,
    )

    return bytes_input_info


def build_bytes_vocab(
    byte_encoding: Literal["uint8"], specials: Sequence[Hashable] = tuple()
) -> OrderedDict:
    bytes_vocab = OrderedDict()

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
