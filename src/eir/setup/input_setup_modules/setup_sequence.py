import json
import os
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    cast,
    overload,
)

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.tokenization_utils_base import (
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)

from eir.data_load.data_source_modules.deeplake_ops import (
    get_deeplake_input_source_iterable,
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.models.input.sequence.transformer_models import SequenceModelConfig
from eir.setup import schemas
from eir.setup.input_setup_modules.common import get_default_sequence_specials
from eir.setup.input_setup_modules.torchtext_port.utils import (
    get_tokenizer as get_pytorch_tokenizer,
)
from eir.setup.input_setup_modules.torchtext_port.vocab import Vocab
from eir.setup.input_setup_modules.torchtext_port.vocab_factory import (
    build_vocab_from_iterator,
    vocab as pytorch_vocab_builder,
)
from eir.setup.schemas import al_tokenizer_choices
from eir.utils.logging import get_logger


class TokenizerProtocolRaw(Protocol):
    def __call__(self, raw_input: str) -> Sequence[str]: ...

    __closure__: tuple[Any, ...] | None


class TokenizerProtocolPreSplit(Protocol):
    def __call__(self, raw_input_split: Sequence[str]) -> Sequence[str]: ...

    __closure__: tuple[Any, ...] | None


class EncodeFuncProtocol(Protocol):
    def __call__(self, raw_input: Sequence[str] | str) -> Sequence[int]: ...


type al_hf_tokenizer_inputs = TextInput | PreTokenizedInput | EncodedInput
al_sequence_input_objects_basic = tuple[
    Vocab,
    "GatheredSequenceStats",
    TokenizerProtocolPreSplit | TokenizerProtocolRaw,
    EncodeFuncProtocol,
]
al_hf_encode_func = Callable[[al_hf_tokenizer_inputs], Sequence[int]]
al_sequence_input_objects_hf = tuple[
    Vocab,
    "GatheredSequenceStats",
    PreTrainedTokenizer,
    al_hf_encode_func,
]

al_tokenizers = TokenizerProtocolRaw | TokenizerProtocolPreSplit
al_encode_funcs = EncodeFuncProtocol | al_hf_encode_func


logger = get_logger(name=__name__)


@dataclass
class ComputedSequenceInputInfo:
    input_config: schemas.InputConfig
    vocab: Vocab
    computed_max_length: int
    encode_func: al_encode_funcs
    special_tokens: "SpecialTokens"
    tokenizer: al_tokenizers | None


def set_up_computed_sequence_input(
    input_config: schemas.InputConfig,
    mode: Literal["train", "eval"] = "train",
    *args,
    **kwargs,
) -> ComputedSequenceInputInfo:
    model_config = input_config.model_config
    assert isinstance(model_config, SequenceModelConfig)

    sequence_input_object_func = _get_sequence_input_object_func(
        pretrained=model_config.pretrained_model
    )

    vocab, gathered_stats, tokenizer, encode_callable = sequence_input_object_func(
        input_config=input_config,
        mode=mode,
    )

    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.SequenceInputDataConfig)

    gathered_stats = possibly_gather_all_stats_from_input(
        prev_gathered_stats=gathered_stats,
        input_source=input_config.input_info.input_source,
        vocab_file=input_type_info.vocab_file,
        split_on=input_type_info.split_on,
        max_length=input_type_info.max_length,
    )

    computed_max_length = get_max_length(
        max_length_config_value=input_type_info.max_length,
        gathered_stats=gathered_stats,
    )

    special_tokens = get_special_tokens(
        tokenizer=tokenizer,
        vocab=vocab,
    )

    sequence_input_info = ComputedSequenceInputInfo(
        input_config=input_config,
        vocab=vocab,
        computed_max_length=computed_max_length,
        encode_func=encode_callable,
        special_tokens=special_tokens,
        tokenizer=tokenizer,
    )

    return sequence_input_info


class SequenceInputObjectGetterFunctionBasic(Protocol):
    def __call__(
        self,
        input_config: schemas.InputConfig,
        mode: Literal["train", "eval"],
    ) -> al_sequence_input_objects_basic: ...


class SequenceInputObjectGetterFunctionHF(Protocol):
    def __call__(
        self,
        input_config: schemas.InputConfig,
        mode: Literal["train", "eval"],
    ) -> al_sequence_input_objects_hf: ...


def _get_sequence_input_object_func(
    pretrained: bool,
) -> SequenceInputObjectGetterFunctionBasic | SequenceInputObjectGetterFunctionHF:
    if pretrained:
        return get_sequence_input_objects_from_pretrained
    return get_sequence_input_objects_from_input


def get_sequence_input_objects_from_input(
    input_config: schemas.InputConfig,
    mode: Literal["train", "eval"],
) -> al_sequence_input_objects_basic:
    gathered_stats = GatheredSequenceStats()
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.SequenceInputDataConfig)

    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit
    tokenizer, gathered_stats = get_tokenizer(
        input_config=input_config,
        gathered_stats=gathered_stats,
    )

    vocab = init_vocab(
        source=input_config.input_info.input_source,
        inner_key=input_config.input_info.input_inner_key,
        tokenizer_name=input_type_info.tokenizer,
        split_on=input_type_info.split_on,
        vocab_file=input_type_info.vocab_file,
        min_freq=input_type_info.min_freq,
        gathered_stats=gathered_stats,
        tokenizer=tokenizer,
    )

    encode_func: EncodeFuncProtocol
    tokenizer_name = input_type_info.tokenizer
    is_pretrained_tokenizer = tokenizer_name in TOKENIZER_MAPPING_NAMES
    if is_pretrained_tokenizer:
        # here we delegate to the __call__ logic of HFTokenizerWrapper
        assert isinstance(tokenizer, HFTokenizerWrapper)
        encode_func = tokenizer  # THIS RAISES type error
    else:
        encode_func = get_tokenizer_encode_func(
            tokenizer=tokenizer,
            pytorch_vocab=vocab,
        )

    return vocab, gathered_stats, tokenizer, encode_func


class HFTokenizerWrapper:
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer

    def __call__(self, text, *args, **kwargs) -> Sequence[int]:
        encoded = self.tokenizer.encode(text)
        return encoded

    def get_vocab(self):
        return self.tokenizer.get_vocab()


def init_vocab(
    source: str,
    inner_key: str | None,
    tokenizer_name: str | None,
    split_on: str | None,
    vocab_file: str | None,
    min_freq: int,
    gathered_stats: "GatheredSequenceStats",
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
) -> Vocab:
    is_hf_pretrained = tokenizer_name in TOKENIZER_MAPPING_NAMES

    if tokenizer_name == "bpe" or is_hf_pretrained:
        tokenizer_object = extract_tokenizer_object_from_function(
            tokenizer_callable=tokenizer
        )
        vocab = sync_hf_and_pytorch_vocab(hf_tokenizer=tokenizer_object)
    else:
        assert gathered_stats.total_count == 0

        vocab_iter = get_vocab_iterator(
            input_source=source,
            split_on=split_on,
            gathered_stats=gathered_stats,
            vocab_file=vocab_file,
            deeplake_inner_key=inner_key,
        )
        tokenized_vocab_iter = get_tokenized_vocab_iterator(
            vocab_iterator=vocab_iter,
            tokenizer=tokenizer,
            is_from_file=vocab_file is not None,
        )

        min_freq = _init_min_freq(
            vocab_file=vocab_file,
            min_freq=min_freq,
        )

        do_sort_by_freq = not vocab_file
        vocab = build_vocab_from_iterator(
            iterator=tokenized_vocab_iter,
            specials=get_default_sequence_specials(),
            min_freq=min_freq,
            sort_by_freq=do_sort_by_freq,
        )

    vocab.set_default_index(vocab["<unk>"])

    return vocab


def _init_min_freq(
    vocab_file: str | None,
    min_freq: int,
) -> int:
    if vocab_file:
        logger.info(
            "Minimum word/token frequency will be set to 0 as vocabulary is loaded "
            "from file %s.",
            vocab_file,
        )
        min_freq = 1
    return min_freq


def extract_tokenizer_object_from_function(
    tokenizer_callable: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
) -> Tokenizer | PreTrainedTokenizer:
    assert isinstance(tokenizer_callable, TokenizerWrapper | HFTokenizerWrapper), (
        f"Expected TokenizerWrapper instance, got {type(tokenizer_callable).__name__}"
    )

    tokenizer_object = tokenizer_callable.tokenizer
    assert isinstance(
        tokenizer_object, Tokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast
    ), (
        f"Expected Tokenizer or PreTrainedTokenizer, got "
        f"{type(tokenizer_object).__name__}"
    )

    return tokenizer_object


def get_tokenizer(
    input_config: schemas.InputConfig,
    gathered_stats: "GatheredSequenceStats",
) -> tuple[TokenizerProtocolPreSplit | TokenizerProtocolRaw, "GatheredSequenceStats"]:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.SequenceInputDataConfig)
    tokenizer_name = input_type_info.tokenizer

    available_pretrained = TOKENIZER_MAPPING_NAMES.keys()

    tokenizer: TokenizerProtocolPreSplit | TokenizerProtocolRaw
    if tokenizer_name == "bpe":
        vocab_iter = get_vocab_iterator(
            input_source=input_config.input_info.input_source,
            split_on=input_type_info.split_on,
            gathered_stats=gathered_stats,
            vocab_file=input_type_info.vocab_file,
            deeplake_inner_key=input_config.input_info.input_inner_key,
        )

        vocab_file = input_type_info.vocab_file
        tokenizer = get_bpe_tokenizer(
            vocab_iterator=vocab_iter,
            vocab_file=vocab_file,
            vocab_size=input_type_info.adaptive_tokenizer_max_vocab_size,
            split_on=input_type_info.split_on,
        )

    elif tokenizer_name in available_pretrained:
        assert tokenizer_name is not None
        logger.info("Using pretrained tokenizer '%s'.", tokenizer_name)
        hf_tokenizer_base = _get_hf_tokenizer(
            hf_model_name=tokenizer_name,
        )
        tokenizer_base = _add_specials_to_hf_tokenizer(hf_tokenizer=hf_tokenizer_base)
        tokenizer = cast(
            TokenizerProtocolRaw, HFTokenizerWrapper(base_tokenizer=tokenizer_base)
        )

    else:
        tokenizer = get_basic_tokenizer(
            tokenizer_name=tokenizer_name,
            tokenizer_language=input_type_info.tokenizer_language,
        )

    return tokenizer, gathered_stats


def _get_default_specials_map() -> dict:
    mapping = {
        "bos_token": "<bos>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "eos_token": "<eos>",
    }

    default_specials = get_default_sequence_specials()
    assert set(mapping.values()) == set(default_specials)

    return mapping


def get_sequence_input_objects_from_pretrained(
    input_config: schemas.InputConfig,
    mode: Literal["train", "eval"],
) -> al_sequence_input_objects_hf:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.SequenceInputDataConfig)
    vocab_file = input_type_info.vocab_file
    do_not_allow_vocab_file = mode == "train"
    if vocab_file and do_not_allow_vocab_file:
        raise ValueError(
            "Using a vocabulary file not supported when training pre-trained models as "
            "their training vocabulary will be used."
        )

    gathered_stats = GatheredSequenceStats()
    hf_model_name = input_config.model_config.model_type
    hf_tokenizer = _get_hf_tokenizer(hf_model_name=hf_model_name)

    def _passthrough_hf_encode(
        raw_input_split: al_hf_tokenizer_inputs,
    ) -> Sequence[int]:
        encoded = hf_tokenizer.encode(text=raw_input_split, is_split_into_words=True)
        return encoded

    vocab = sync_hf_and_pytorch_vocab(hf_tokenizer=hf_tokenizer)

    return vocab, gathered_stats, hf_tokenizer, _passthrough_hf_encode


def sync_hf_and_pytorch_vocab(hf_tokenizer: Tokenizer | PreTrainedTokenizer) -> Vocab:
    hf_tokenizer_vocab = hf_tokenizer.get_vocab()
    hf_tokenizer_vocab_sorted = OrderedDict(
        dict(sorted(hf_tokenizer_vocab.items(), key=lambda item: item[1]))
    )
    vocab = pytorch_vocab_builder(ordered_dict=hf_tokenizer_vocab_sorted, min_freq=0)

    return vocab


def _get_hf_tokenizer(
    hf_model_name: str,
    add_prefix_space: bool = True,
) -> PreTrainedTokenizer:
    """
    See https://github.com/huggingface/transformers/issues/5486 for why we need to
    set the environment variable.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hf_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=hf_model_name,
        add_prefix_space=add_prefix_space,
    )

    hf_tokenizer = _add_specials_to_hf_tokenizer(hf_tokenizer=hf_tokenizer)
    return hf_tokenizer


def _add_specials_to_hf_tokenizer(
    hf_tokenizer: PreTrainedTokenizer,
) -> PreTrainedTokenizer:
    hf_tokenizer_copy = deepcopy(hf_tokenizer)
    name_special_token_map = _get_default_specials_map()

    specials_tokens_to_add = {}
    for special_token_name, special_token in name_special_token_map.items():
        if special_token_name not in hf_tokenizer_copy.special_tokens_map:
            specials_tokens_to_add[special_token_name] = special_token

    added_tokens = hf_tokenizer_copy.add_special_tokens(
        special_tokens_dict=specials_tokens_to_add
    )
    logger.debug("Special tokens %s added to %s.", added_tokens, hf_tokenizer)

    return hf_tokenizer_copy


class TokenizerWrapper:
    def __init__(self, tokenizer, is_pretokenized: bool):
        self.tokenizer = tokenizer
        self.is_pretokenized = is_pretokenized

    def __call__(self, input_text: str | Sequence[str]) -> Sequence[str]:
        if not input_text or (
            isinstance(input_text, list | tuple) and len(input_text) == 0
        ):
            return []

        if isinstance(input_text, str) and input_text.strip() == "":
            return []

        tokens = self.tokenizer.encode(
            sequence=input_text,
            is_pretokenized=self.is_pretokenized,
        ).tokens
        return tokens


def get_bpe_tokenizer(
    vocab_iterator: Iterator | None,
    vocab_file: str | None,
    vocab_size: int | None,
    split_on: str | None,
) -> TokenizerProtocolRaw | TokenizerProtocolPreSplit:
    tokenizer = _get_bpe_tokenizer_object(
        vocab_iterator=vocab_iterator,
        vocab_file=vocab_file,
        vocab_size=vocab_size,
    )

    if split_on is None:
        return cast(
            TokenizerProtocolRaw,
            TokenizerWrapper(
                tokenizer=tokenizer,
                is_pretokenized=False,
            ),
        )
    return cast(
        TokenizerProtocolPreSplit,
        TokenizerWrapper(
            tokenizer=tokenizer,
            is_pretokenized=True,
        ),
    )


class TokenizerVocabSizeError(Exception):
    pass


@dataclass
class TokenizerValidationResult:
    is_valid: bool
    actual_vocab_size: int
    requested_vocab_size: int | None
    error_message: str | None = None


def validate_tokenizer_vocab_size(
    tokenizer: Tokenizer,
    requested_vocab_size: int | None,
) -> TokenizerValidationResult:
    if requested_vocab_size is None:
        return TokenizerValidationResult(
            is_valid=True,
            actual_vocab_size=tokenizer.get_vocab_size(),
            requested_vocab_size=None,
        )

    actual_size = tokenizer.get_vocab_size()

    if actual_size > requested_vocab_size:
        return TokenizerValidationResult(
            is_valid=False,
            actual_vocab_size=actual_size,
            requested_vocab_size=requested_vocab_size,
            error_message=(
                f"The final vocabulary size ({actual_size}) exceeds the requested "
                f"maximum size ({requested_vocab_size}). This can happen when the "
                f"requested vocabulary size is too small for the complexity of your "
                f"training data. Please increase the vocab_size parameter to at least "
                f"{actual_size} tokens."
            ),
        )

    return TokenizerValidationResult(
        is_valid=True,
        actual_vocab_size=actual_size,
        requested_vocab_size=requested_vocab_size,
    )


def _get_bpe_tokenizer_object(
    vocab_iterator: Iterator | None,
    vocab_file: str | None,
    vocab_size: int | None,
    raise_on_validation_error: bool = True,
) -> Tokenizer:
    if vocab_file:
        logger.info("Loading BPE vocabulary from file %s.", vocab_file)

        if not vocab_file.endswith(".json"):
            raise ValueError(
                "Vocabulary file must be a HuggingFace Tokenizers "
                "compatible .json file."
            )

        tokenizer = Tokenizer.from_file(path=vocab_file)
    else:
        if vocab_iterator is None:
            raise ValueError("Either vocab_iterator or vocab_file must be provided.")

        logger.info("Training BPE tokenizer from source data.")

        tokenizer = Tokenizer(model=BPE(unk_token="<unk>"))

        special_tokens = get_default_sequence_specials()
        if vocab_size is not None:
            trainer = BpeTrainer(
                special_tokens=special_tokens,
                vocab_size=vocab_size,
            )
        else:
            trainer = BpeTrainer(
                special_tokens=special_tokens,
            )

        tokenizer.train_from_iterator(iterator=vocab_iterator, trainer=trainer)

        validation_result = validate_tokenizer_vocab_size(
            tokenizer=tokenizer,
            requested_vocab_size=vocab_size,
        )

        if not validation_result.is_valid:
            logger.warning(validation_result.error_message)

            if raise_on_validation_error:
                raise TokenizerVocabSizeError(validation_result.error_message)

    return tokenizer


def identity_tokenize(raw_input_split: Sequence[str]) -> Sequence[str]:
    return raw_input_split


def join_and_tokenize(
    raw_input_split: Sequence[str], base_tokenizer: Any
) -> Sequence[str]:
    input_joined = " ".join(raw_input_split)
    return base_tokenizer(input_joined)


def make_join_tokenizer(base_tokenizer: Any) -> TokenizerProtocolPreSplit:
    return cast(
        TokenizerProtocolPreSplit,
        partial(join_and_tokenize, base_tokenizer=base_tokenizer),
    )


def get_basic_tokenizer(
    tokenizer_name: al_tokenizer_choices,  # type: ignore
    tokenizer_language: str | None,
) -> TokenizerProtocolPreSplit:
    if not tokenizer_name:
        return cast(TokenizerProtocolPreSplit, identity_tokenize)

    _validate_pytorch_tokenizer_args(
        tokenizer_name=tokenizer_name,
        tokenizer_language=tokenizer_language,
    )
    logger.debug(
        "Using tokenizer '%s' with language '%s'.", tokenizer_name, tokenizer_language
    )

    base_tokenizer = get_pytorch_tokenizer(
        tokenizer=tokenizer_name,
        language=tokenizer_language,
    )

    return cast(TokenizerProtocolPreSplit, make_join_tokenizer(base_tokenizer))


def _validate_pytorch_tokenizer_args(
    tokenizer_name: al_tokenizer_choices,  # type: ignore
    tokenizer_language: str | None,
) -> None:
    tokenizer_language_passed = tokenizer_language is not None
    tokenizer_does_not_support_language = tokenizer_name not in (
        "spacy",
        "basic_english",
    )

    if tokenizer_language_passed and tokenizer_does_not_support_language:
        raise ValueError(
            "Tokenizer '%s' does not support setting a language (got '%s'). "
            "Please leave it as None.",
            tokenizer_name,
            tokenizer_language,
        )


@overload
def encode_text(
    raw_input: str,
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    pytorch_vocab: Vocab,
) -> Sequence[int]: ...


@overload
def encode_text(
    raw_input: Sequence[str],
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    pytorch_vocab: Vocab,
) -> Sequence[int]: ...


def encode_text(
    raw_input: str | Sequence[str],
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    pytorch_vocab: Vocab,
) -> Sequence[int]:
    """
    TODO: Restructure this and related logic to not operate on union types.
    """
    input_tokenized = tokenizer(raw_input)  # type: ignore
    input_as_ids = pytorch_vocab(input_tokenized)
    return input_as_ids


def get_tokenizer_encode_func(
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    pytorch_vocab: Vocab,
) -> EncodeFuncProtocol:
    return partial(encode_text, tokenizer=tokenizer, pytorch_vocab=pytorch_vocab)


def get_tokenized_vocab_iterator(
    vocab_iterator: Iterator[Sequence[str] | str],
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    is_from_file: bool,
) -> Generator[Sequence[str]]:
    @overload
    def _do_tokenize(list_of_words_: str) -> Sequence[str]: ...

    @overload
    def _do_tokenize(list_of_words_: Sequence[str]) -> Sequence[str]: ...

    def _do_tokenize(list_of_words_):
        return tokenizer(list_of_words_)

    if is_from_file:
        yield from vocab_iterator
    else:
        for list_of_words in vocab_iterator:
            yield _do_tokenize(list_of_words_=list_of_words)


def get_vocab_iterator(
    input_source: str,
    split_on: str | None,
    gathered_stats: "GatheredSequenceStats",
    vocab_file: str | None = None,
    deeplake_inner_key: str | None = None,
) -> Generator[Sequence[str]]:
    """
    Note: When using a vocabulary file, we explicitly expect one token per line,
    therefore we do not split on any character.
    """

    if vocab_file is None:
        logger.info(
            "Vocabulary will be collected from input source %s, "
            "splitting tokens on '%s'.",
            input_source,
            split_on,
        )
        vocab_iter = yield_tokens_from_source(
            data_source=input_source,
            split_on=split_on,
            gathered_stats=gathered_stats,
            deeplake_inner_key=deeplake_inner_key,
        )
    else:
        logger.info(
            "Vocabulary for %s will be collected from vocabulary file %s.",
            input_source,
            vocab_file,
        )

        vocab_iter = get_vocab_file_iterator(
            vocab_file=Path(vocab_file),
            gathered_stats=gathered_stats,
        )

    return vocab_iter


def get_vocab_file_iterator(
    vocab_file: Path,
    gathered_stats: "GatheredSequenceStats",
) -> Generator[str]:
    try:
        with vocab_file.open("r") as f:
            vocab_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise e
    except FileNotFoundError as e:
        raise e

    if not isinstance(vocab_dict, dict):
        raise ValueError(f"The content of {vocab_file} is not a dictionary.")

    sorted_tokens = sorted(vocab_dict.items(), key=lambda x: x[1])

    for token, _ in sorted_tokens:
        gathered_stats.total_count += 1
        yield token


@dataclass
class GatheredSequenceStats:
    total_count: int = 0
    total_files: int = 0
    max_length: int = 0


def yield_tokens_from_source(
    data_source: str,
    split_on: str | None,
    gathered_stats: GatheredSequenceStats,
    deeplake_inner_key: str | None = None,
):
    data_source_path = Path(data_source)

    if is_deeplake_dataset(data_source=str(data_source_path)):
        assert deeplake_inner_key is not None
        yield from yield_tokens_from_deeplake_dataset(
            data_source=data_source_path,
            split_on=split_on,
            gathered_stats=gathered_stats,
            inner_key=deeplake_inner_key,
        )

    elif data_source_path.is_dir():
        iterator = tqdm(Path(data_source).iterdir(), desc="Vocabulary Setup")
        for file in iterator:
            preserve_full = split_on is None
            yield from yield_tokens_from_file(
                file_path=str(file),
                split_on=split_on,
                gathered_stats=gathered_stats,
                preserve_full_content=preserve_full,
            )

    elif data_source_path.is_file():
        assert data_source_path.suffix == ".csv"
        yield from yield_tokens_from_csv(
            file_path=data_source,
            split_on=split_on,
            gathered_stats=gathered_stats,
        )

    return gathered_stats


def yield_tokens_from_deeplake_dataset(
    data_source: Path,
    split_on: str | None,
    gathered_stats: GatheredSequenceStats,
    inner_key: str,
) -> Generator[Sequence[str]]:
    deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
    deeplake_iter = get_deeplake_input_source_iterable(
        deeplake_dataset=deeplake_ds,
        inner_key=inner_key,
    )

    split_func = get_sequence_split_function(split_on=split_on)

    for sample in deeplake_iter:
        cur_sequence = str(sample)
        cur_line = split_func(cur_sequence)

        cur_length = len(cur_line)
        gathered_stats.total_count += len(cur_line)

        if cur_length > gathered_stats.max_length:
            gathered_stats.max_length = cur_length

        yield cur_line


def yield_tokens_from_file(
    file_path: str,
    split_on: str | None,
    gathered_stats: GatheredSequenceStats,
    preserve_full_content: bool = False,
) -> Generator[Sequence[str]]:
    gathered_stats.total_files += 1

    if preserve_full_content:
        with open(file_path) as f:
            content = f.read()
            gathered_stats.total_count += 1
            cur_length = len(content)
            if cur_length > gathered_stats.max_length:
                gathered_stats.max_length = cur_length
            yield [content]
        return

    split_func = get_sequence_split_function(split_on=split_on)

    with open(file_path) as f:
        for line in f:
            line_parsed = line[:-1] if line.endswith("\n") else line
            cur_line = split_func(line_parsed)

            cur_length = len(cur_line)

            if split_on is None:
                gathered_stats.total_count += 1
            else:
                gathered_stats.total_count += len(cur_line)

            if cur_length > gathered_stats.max_length:
                gathered_stats.max_length = cur_length

            yield cur_line


def yield_tokens_from_csv(
    file_path: str, split_on: str | None, gathered_stats: GatheredSequenceStats
) -> Generator[Sequence[str]]:
    split_func = get_sequence_split_function(split_on=split_on)

    df = pd.read_csv(filepath_or_buffer=file_path, index_col="ID", dtype={"ID": str})
    if "Sequence" not in df.columns:
        raise ValueError(
            "CSV file '%s' does not have a column named 'Sequence'. "
            "Please ensure that the column name is correct and present.",
            file_path,
        )

    iterator = tqdm(df.itertuples(), desc="Vocabulary Setup")
    for row in iterator:
        cur_sequence = row.Sequence
        if pd.isna(cur_sequence):
            cur_sequence = ""

        cur_line = split_func(cur_sequence)

        cur_length = len(cur_line)
        gathered_stats.total_count += len(cur_line)
        gathered_stats.total_files += 1

        if cur_length > gathered_stats.max_length:
            gathered_stats.max_length = cur_length

        yield cur_line


def get_sequence_split_function(
    split_on: str | None,
) -> Callable[[str], list[str] | str]:
    match split_on:
        case "":
            return lambda x: list(x)
        case None:
            return lambda x: x
        case _:
            return lambda x: x.split(split_on)


class ReturnSavingGenerator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.caught_return_value = yield from self.gen


def possibly_gather_all_stats_from_input(
    prev_gathered_stats: GatheredSequenceStats,
    input_source: str,
    vocab_file: str | None,
    split_on: str | None,
    max_length: schemas.al_max_sequence_length,
) -> GatheredSequenceStats:
    """
    Note that we use all(...) there to exhaust the generator object, so that the
    stats get accumulated in the GatheredSequenceStats().
    """
    gathered_stats = prev_gathered_stats

    dynamic_and_vocab_file = vocab_file and max_length in {"average", "max"}

    if dynamic_and_vocab_file:
        logger.info(
            "Doing a full pass over input data despite vocabulary file %s being "
            "present, as dynamic max length '%s' was requested.",
            vocab_file,
            max_length,
        )
        vocab_iter = yield_tokens_from_source(
            data_source=input_source,
            split_on=split_on,
            gathered_stats=GatheredSequenceStats(),
        )
        value_keeping_gen = ReturnSavingGenerator(gen=vocab_iter)
        all(value_keeping_gen)
        gathered_stats = value_keeping_gen.caught_return_value

    return gathered_stats


def get_max_length(
    max_length_config_value: schemas.al_max_sequence_length,
    gathered_stats: GatheredSequenceStats,
):
    if isinstance(max_length_config_value, int):
        return max_length_config_value

    if max_length_config_value == "max":
        logger.info(
            "Using inferred max length found in sequence data source as %d",
            gathered_stats.max_length,
        )
        return gathered_stats.max_length
    if max_length_config_value == "average":
        average_length = gathered_stats.total_count // gathered_stats.total_files
        logger.info(
            "Using inferred average length found in sequence data source as %d",
            average_length,
        )
        return average_length

    raise ValueError("Unknown max length config value %s.", max_length_config_value)


@dataclass
class SpecialTokens:
    mask_idx: int
    pad_idx: int
    eos_idx: int
    bos_idx: int
    unk_idx: int

    mask_token: str
    pad_token: str
    eos_token: str
    bos_token: str
    unk_token: str


def get_special_tokens(
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit,
    vocab: Vocab,
) -> SpecialTokens:
    keys_and_defaults = (
        ("mask_token", "<mask>"),
        ("pad_token", "<pad>"),
        ("bos_token", "<bos>"),
        ("eos_token", "<eos>"),
        ("unk_token", "<unk>"),
    )

    token_values = {}
    index_values = {}

    has_inner = hasattr(tokenizer, "tokenizer")
    has_specials_map = False
    if has_inner:
        if isinstance(tokenizer, HFTokenizerWrapper):
            has_specials_map = hasattr(tokenizer.tokenizer, "special_tokens_map")

    for key, default in keys_and_defaults:
        if has_inner and has_specials_map:
            assert isinstance(tokenizer, HFTokenizerWrapper)
            wrapper = tokenizer
            inner_tokenizer = wrapper.tokenizer
            cur_token = inner_tokenizer.special_tokens_map.get(key, default)
        else:
            cur_token = getattr(tokenizer, key, default)

        token_values[key] = cur_token

        cur_index = vocab[cur_token]
        assert cur_index >= 0, f"Token '{cur_token}' not found in vocab."
        idx_kwarg_key = key.replace("_token", "_idx")
        index_values[idx_kwarg_key] = cur_index

    special_tokens = SpecialTokens(
        mask_idx=index_values["mask_idx"],
        pad_idx=index_values["pad_idx"],
        eos_idx=index_values["eos_idx"],
        bos_idx=index_values["bos_idx"],
        unk_idx=index_values["unk_idx"],
        mask_token=token_values["mask_token"],
        pad_token=token_values["pad_token"],
        eos_token=token_values["eos_token"],
        bos_token=token_values["bos_token"],
        unk_token=token_values["unk_token"],
    )

    return special_tokens
