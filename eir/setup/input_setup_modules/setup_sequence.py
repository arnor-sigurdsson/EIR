import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, List, Union, Optional, Iterator, Generator, Tuple

import pandas as pd
from aislib.misc_utils import get_logger
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torchtext.data import get_tokenizer as get_pytorch_tokenizer
from torchtext.vocab import (
    Vocab,
    build_vocab_from_iterator,
    vocab as pytorch_vocab_builder,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import (
    TextInput,
    PreTokenizedInput,
    EncodedInput,
)

from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    load_deeplake_dataset,
    get_deeplake_input_source_iterable,
)
from eir.setup import schemas
from eir.setup.input_setup_modules.common import get_default_sequence_specials
from eir.setup.schemas import al_tokenizer_choices

al_hf_tokenizer_inputs = Union[TextInput, PreTokenizedInput, EncodedInput]
al_sequence_input_objects_basic = Tuple[
    Vocab,
    "GatheredSequenceStats",
    Callable[[Sequence[str]], Sequence[str]],
    Callable[[Sequence[str]], List[int]],
]
al_sequence_input_objects_hf = Tuple[
    Vocab,
    "GatheredSequenceStats",
    PreTrainedTokenizer,
    Callable[[al_hf_tokenizer_inputs], Sequence[int]],
]


logger = get_logger(name=__name__)


@dataclass
class ComputedSequenceInputInfo:
    input_config: schemas.InputConfig
    vocab: Vocab
    computed_max_length: int
    encode_func: Callable[[Sequence[str]], List[int]]
    tokenizer: Union[Callable, None] = None


def set_up_sequence_input_for_training(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ComputedSequenceInputInfo:
    sequence_input_object_func = _get_sequence_input_object_func(
        pretrained=input_config.model_config.pretrained_model
    )
    vocab, gathered_stats, tokenizer, encode_callable = sequence_input_object_func(
        input_config=input_config
    )

    gathered_stats = possibly_gather_all_stats_from_input(
        prev_gathered_stats=gathered_stats,
        input_source=input_config.input_info.input_source,
        vocab_file=input_config.input_type_info.vocab_file,
        split_on=input_config.input_type_info.split_on,
        max_length=input_config.input_type_info.max_length,
    )

    computed_max_length = get_max_length(
        max_length_config_value=input_config.input_type_info.max_length,
        gathered_stats=gathered_stats,
    )
    sequence_input_info = ComputedSequenceInputInfo(
        input_config=input_config,
        vocab=vocab,
        computed_max_length=computed_max_length,
        encode_func=encode_callable,
        tokenizer=tokenizer,
    )

    return sequence_input_info


def _get_sequence_input_object_func(
    pretrained: bool,
) -> Callable[
    [schemas.InputConfig],
    Union[al_sequence_input_objects_basic, al_sequence_input_objects_hf],
]:
    if pretrained:
        return get_sequence_input_objects_from_pretrained
    else:
        return get_sequence_input_objects_from_input


def get_sequence_input_objects_from_input(
    input_config: schemas.InputConfig,
) -> al_sequence_input_objects_basic:
    gathered_stats = GatheredSequenceStats()

    vocab_file = input_config.input_type_info.vocab_file

    tokenizer = get_tokenizer(input_config=input_config)

    vocab_iter_kwargs = dict(
        input_source=input_config.input_info.input_source,
        split_on=input_config.input_type_info.split_on,
        gathered_stats=gathered_stats,
        vocab_file=input_config.input_type_info.vocab_file,
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )
    vocab_iter = get_vocab_iterator(**vocab_iter_kwargs)
    tokenized_vocab_iter = get_tokenized_vocab_iterator(
        vocab_iterator=vocab_iter, tokenizer=tokenizer
    )

    min_freq = input_config.input_type_info.min_freq
    if vocab_file:
        logger.info(
            "Minimum word/token frequency will be set to 0 as vocabulary is loaded "
            "from file %s.",
            vocab_file,
        )
        min_freq = 1

    vocab = build_vocab_from_iterator(
        iterator=tokenized_vocab_iter,
        specials=get_default_sequence_specials(),
        min_freq=min_freq,
    )
    vocab.set_default_index(vocab["<unk>"])

    encode_func = get_pytorch_tokenizer_encode_func(
        pytorch_tokenizer=tokenizer, pytorch_vocab=vocab
    )

    return vocab, gathered_stats, tokenizer, encode_func


def get_tokenizer(
    input_config: schemas.InputConfig,
) -> Callable[[Sequence[str]], Sequence[str]]:
    tokenizer_name = input_config.input_type_info.tokenizer

    if tokenizer_name == "bpe":
        vocab_iter_kwargs = dict(
            input_source=input_config.input_info.input_source,
            split_on=input_config.input_type_info.split_on,
            gathered_stats=GatheredSequenceStats(),
            vocab_file=input_config.input_type_info.vocab_file,
            deeplake_inner_key=input_config.input_info.input_inner_key,
        )
        vocab_iter = get_vocab_iterator(**vocab_iter_kwargs)

        vocab_file = input_config.input_type_info.vocab_file
        tokenizer = get_bpe_tokenizer(vocab_iterator=vocab_iter, vocab_file=vocab_file)

    else:
        tokenizer = get_basic_tokenizer(
            tokenizer_name=tokenizer_name,
            tokenizer_language=input_config.input_type_info.tokenizer_language,
        )

    return tokenizer


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
) -> al_sequence_input_objects_hf:
    vocab_file = input_config.input_type_info.vocab_file
    if vocab_file:
        raise ValueError(
            "Using a vocabulary file not supported when using pre-trained models as "
            "their training vocabulary will be used."
        )

    gathered_stats = GatheredSequenceStats()
    hf_model_name = input_config.model_config.model_type
    hf_tokenizer = _get_hf_tokenizer(hf_model_name=hf_model_name)

    def _passthrough_hf_encode(raw_input_split: al_hf_tokenizer_inputs) -> List[int]:
        return hf_tokenizer.encode(text=raw_input_split, is_split_into_words=True)

    vocab = _sync_hf_and_pytorch_vocab(hf_tokenizer=hf_tokenizer)

    return vocab, gathered_stats, hf_tokenizer, _passthrough_hf_encode


def _sync_hf_and_pytorch_vocab(hf_tokenizer: PreTrainedTokenizer) -> Vocab:
    hf_tokenizer_vocab = hf_tokenizer.get_vocab()
    hf_tokenizer_vocab_sorted = OrderedDict(
        {k: v for k, v in sorted(hf_tokenizer_vocab.items(), key=lambda item: item[1])}
    )
    vocab = pytorch_vocab_builder(ordered_dict=hf_tokenizer_vocab_sorted, min_freq=0)

    return vocab


def _get_hf_tokenizer(hf_model_name: str) -> PreTrainedTokenizer:
    """
    See https://github.com/huggingface/transformers/issues/5486 for why we need to
    set the environment variable.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    hf_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=hf_model_name, add_prefix_space=True
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

    hf_tokenizer_copy.add_special_tokens(special_tokens_dict=specials_tokens_to_add)
    logger.debug("Special tokens %s added to %s.", specials_tokens_to_add, hf_tokenizer)

    return hf_tokenizer_copy


def get_bpe_tokenizer(vocab_iterator: Optional[Iterator], vocab_file: Optional[str]):
    tokenizer = _get_bpe_tokenizer_object(
        vocab_iterator=vocab_iterator, vocab_file=vocab_file
    )

    def _tokenize(raw_input_split: Sequence[str]) -> Sequence[str]:
        return tokenizer.encode(raw_input_split, is_pretokenized=True).tokens

    return _tokenize


def _get_bpe_tokenizer_object(
    vocab_iterator: Optional[Iterator], vocab_file: Optional[str]
):
    if vocab_file:
        logger.info("Loading BPE vocabulary from file %s.", vocab_file)

        if not vocab_file.endswith(".json"):
            raise ValueError(
                "Vocabulary file must be a HuggingFace Tokenizers "
                "compatible .json file."
            )

        tokenizer = Tokenizer.from_file(path=vocab_file)
    else:
        assert vocab_iterator is not None

        logger.info("Training BPE tokenizer from source data.")

        tokenizer = Tokenizer(model=BPE(unk_token="<unk>"))

        special_tokens = get_default_sequence_specials()
        trainer = BpeTrainer(special_tokens=special_tokens)

        tokenizer.train_from_iterator(iterator=vocab_iterator, trainer=trainer)

    return tokenizer


def get_basic_tokenizer(
    tokenizer_name: al_tokenizer_choices,
    tokenizer_language: Optional[str],
) -> Callable[[Sequence[str]], Sequence[str]]:
    if not tokenizer_name:
        return lambda x: x

    _validate_tokenizer_args(
        tokenizer_name=tokenizer_name, tokenizer_language=tokenizer_language
    )
    logger.debug(
        "Using tokenizer '%s' with language '%s'.", tokenizer_name, tokenizer_language
    )
    tokenizer = get_pytorch_tokenizer(
        tokenizer=tokenizer_name, language=tokenizer_language
    )

    def _join_and_tokenize(raw_input_split: Sequence[str]) -> Sequence[str]:
        input_joined = " ".join(raw_input_split)
        return tokenizer(input_joined)

    return _join_and_tokenize


def get_pytorch_tokenizer_encode_func(
    pytorch_tokenizer: Callable[[Sequence[str]], Sequence[str]],
    pytorch_vocab: Vocab,
) -> Callable[[Sequence[str]], List[int]]:
    """
    TODO: Possibly deprecate using torchtext and just switch completely to HF.
    """

    def _encode_func(raw_input_split: Sequence[str]) -> List[int]:
        input_tokenized = pytorch_tokenizer(raw_input_split)
        input_as_ids = pytorch_vocab(input_tokenized)
        return input_as_ids

    return _encode_func


def _validate_tokenizer_args(
    tokenizer_name: al_tokenizer_choices, tokenizer_language: Union[str, None]
):
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


def get_tokenized_vocab_iterator(
    vocab_iterator: Iterator[Sequence[str]],
    tokenizer: Callable[[Sequence[str]], Sequence[str]],
) -> Generator[List[str], None, None]:
    for list_of_words in vocab_iterator:
        yield tokenizer(list_of_words)


def get_vocab_iterator(
    input_source: str,
    split_on: str,
    gathered_stats: "GatheredSequenceStats",
    vocab_file: Union[str, None] = None,
    deeplake_inner_key: Optional[str] = None,
) -> Generator[Sequence[str], None, None]:
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
        vocab_iter = yield_tokens_from_file(
            file_path=vocab_file, split_on=" ", gathered_stats=gathered_stats
        )

    return vocab_iter


@dataclass
class GatheredSequenceStats:
    total_count: int = 0
    total_files: int = 0
    max_length: int = 0


def yield_tokens_from_source(
    data_source: str,
    split_on: str,
    gathered_stats: GatheredSequenceStats,
    deeplake_inner_key: Optional[str] = None,
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
            yield from yield_tokens_from_file(
                file_path=str(file), split_on=split_on, gathered_stats=gathered_stats
            )

    elif data_source_path.is_file():
        assert data_source_path.suffix == ".csv"
        yield from yield_tokens_from_csv(
            file_path=data_source, split_on=split_on, gathered_stats=gathered_stats
        )

    return gathered_stats


def yield_tokens_from_deeplake_dataset(
    data_source: Path,
    split_on: str,
    gathered_stats: GatheredSequenceStats,
    inner_key: str,
) -> Generator[Sequence[str], None, None]:
    deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
    deeplake_iter = get_deeplake_input_source_iterable(
        deeplake_dataset=deeplake_ds, inner_key=inner_key
    )

    split_func = get_sequence_split_function(split_on=split_on)

    for sample in deeplake_iter:
        cur_sequence = sample.text()
        cur_line = split_func(cur_sequence)

        cur_length = len(cur_line)
        gathered_stats.total_count += len(cur_line)

        if cur_length > gathered_stats.max_length:
            gathered_stats.max_length = cur_length

        yield cur_line


def yield_tokens_from_file(
    file_path: str, split_on: str, gathered_stats: GatheredSequenceStats
):
    gathered_stats.total_files += 1

    split_func = get_sequence_split_function(split_on=split_on)

    with open(file_path, "r") as f:
        for line in f:
            cur_line = split_func(line.strip())

            cur_length = len(cur_line)
            gathered_stats.total_count += len(cur_line)

            if cur_length > gathered_stats.max_length:
                gathered_stats.max_length = cur_length

            yield cur_line


def yield_tokens_from_csv(
    file_path: str, split_on: str, gathered_stats: GatheredSequenceStats
) -> Generator[Sequence[str], None, None]:
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


def get_sequence_split_function(split_on: str) -> Callable[[str], List[str]]:
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
    vocab_file: str,
    split_on: str,
    max_length: schemas.al_max_sequence_length,
) -> GatheredSequenceStats:
    """
    Note that we use all(...) there to exhaust the generator object, so that the
    stats get accumulated in the GatheredSequenceStats().
    """
    gathered_stats = prev_gathered_stats

    if vocab_file and max_length in {"average", "max"}:
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
    elif max_length_config_value == "average":
        average_length = gathered_stats.total_count // gathered_stats.total_files
        logger.info(
            "Using inferred average length found in sequence data source as %d",
            average_length,
        )
        return average_length

    raise ValueError("Unknown max length config value %s.", max_length_config_value)
