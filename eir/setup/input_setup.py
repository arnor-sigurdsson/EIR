from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Union,
    Generator,
    Sequence,
    Callable,
    Tuple,
    TYPE_CHECKING,
    Iterator,
    List,
)

import dill
import numpy as np
from aislib.misc_utils import get_logger, ensure_path_exists
from torchtext.data.utils import get_tokenizer as get_pytorch_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm

from eir.data_load.label_setup import (
    Labels,
    set_up_train_and_valid_tabular_data,
    TabularFileInfo,
    get_array_path_iterator,
    save_transformer_set,
)
from eir.setup import schemas
from eir.setup.schemas import al_tokenizer_choices

if TYPE_CHECKING:
    from eir.train import Hooks

logger = get_logger(__name__)

al_input_objects_as_dict = Dict[
    str, Union["OmicsInputInfo", "TabularInputInfo", "SequenceInputInfo"]
]


def set_up_inputs_for_training(
    inputs_configs: schemas.al_input_configs,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> al_input_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_input_name_config_iterator(input_configs=inputs_configs)
    for name, input_config in name_config_iter:
        cur_input_data_config = input_config.input_info
        setup_func = get_input_setup_function(
            input_type=cur_input_data_config.input_type
        )
        logger.info(
            "Setting up %s inputs '%s' from %s.",
            cur_input_data_config.input_type,
            cur_input_data_config.input_name,
            cur_input_data_config.input_source,
        )
        set_up_input = setup_func(
            input_config=input_config,
            train_ids=train_ids,
            valid_ids=valid_ids,
            hooks=hooks,
        )
        all_inputs[name] = set_up_input

    return all_inputs


def get_input_name_config_iterator(input_configs: schemas.al_input_configs):
    for input_config in input_configs:
        cur_input_data_config = input_config.input_info
        cur_name = (
            f"{cur_input_data_config.input_type}_{cur_input_data_config.input_name}"
        )
        yield cur_name, input_config


def get_input_setup_function(input_type) -> Callable:
    mapping = get_input_setup_function_map()

    return mapping[input_type]


def get_input_setup_function_map() -> Dict[str, Callable]:
    setup_mapping = {
        "omics": set_up_omics_input,
        "tabular": set_up_tabular_input_for_training,
        "sequence": set_up_sequence_input_for_training,
    }

    return setup_mapping


@dataclass
class SequenceInputInfo:
    input_config: schemas.InputConfig
    vocab: Vocab
    tokenizer: Callable
    computed_max_length: int


def set_up_sequence_input_for_training(
    input_config: schemas.InputConfig, *args, **kwargs
):

    vocab, gathered_stats, tokenizer = get_vocab_wrapper(input_config=input_config)

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
    sequence_input_info = SequenceInputInfo(
        input_config=input_config,
        vocab=vocab,
        computed_max_length=computed_max_length,
        tokenizer=tokenizer,
    )

    return sequence_input_info


@dataclass
class GatheredSequenceStats:
    total_count: int = 0
    total_files: int = 0
    max_length: int = 0


def get_vocab_wrapper(
    input_config: schemas.InputConfig,
) -> Tuple[Vocab, GatheredSequenceStats, Callable]:
    gathered_stats = GatheredSequenceStats()

    vocab_file = input_config.input_type_info.vocab_file
    vocab_iter = get_vocab_iterator(
        input_source=input_config.input_info.input_source,
        split_on=input_config.input_type_info.split_on,
        gathered_stats=gathered_stats,
        vocab_file=input_config.input_type_info.vocab_file,
    )
    tokenizer = get_tokenizer(
        tokenizer_name=input_config.input_type_info.tokenizer,
        tokenizer_language=input_config.input_type_info.tokenizer_language,
    )
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
        specials=["<unk>"],
        min_freq=min_freq,
    )
    vocab.set_default_index(vocab["<unk>"])

    return vocab, gathered_stats, tokenizer


def get_tokenizer(
    tokenizer_name: al_tokenizer_choices,
    tokenizer_language: Union[str, None],
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

    def _join_and_tokenize(raw_input: Sequence[str]) -> Sequence[str]:
        input_joined = " ".join(raw_input)
        return tokenizer(input_joined)

    return _join_and_tokenize


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


def yield_tokens_from_source(
    data_source: str, split_on: str, gathered_stats: GatheredSequenceStats
):
    iterator = tqdm(Path(data_source).iterdir(), desc="Vocabulary Setup")

    for file in iterator:
        yield from yield_tokens_from_file(
            file_path=str(file), split_on=split_on, gathered_stats=gathered_stats
        )
    return gathered_stats


def yield_tokens_from_file(
    file_path: str, split_on: str, gathered_stats: GatheredSequenceStats
):
    gathered_stats.total_files += 1

    split_func = _get_split_func(split_on=split_on)

    with open(file_path, "r") as f:
        for line in f:
            cur_line = split_func(line.strip())

            cur_length = len(cur_line)
            gathered_stats.total_count += len(cur_line)

            if cur_length > gathered_stats.max_length:
                gathered_stats.max_length = cur_length

            yield cur_line


def _get_split_func(split_on: str) -> Callable[[str], List[str]]:
    if split_on == "":
        return lambda x: list(x)
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


@dataclass
class TabularInputInfo:
    labels: Labels
    input_config: schemas.InputConfig


def set_up_tabular_input_for_training(
    input_config: schemas.InputConfig,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> TabularInputInfo:
    tabular_file_info = get_tabular_input_file_info(
        input_source=input_config.input_info.input_source,
        tabular_data_type_config=input_config.input_type_info,
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    tabular_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_file_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    tabular_input_info = TabularInputInfo(
        labels=tabular_labels, input_config=input_config
    )

    return tabular_input_info


def get_tabular_input_file_info(
    input_source: str,
    tabular_data_type_config: schemas.TabularInputDataConfig,
) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=Path(input_source),
        con_columns=tabular_data_type_config.extra_con_columns,
        cat_columns=tabular_data_type_config.extra_cat_columns,
        parsing_chunk_size=tabular_data_type_config.label_parsing_chunk_size,
    )

    return table_info


@dataclass
class OmicsInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"


def set_up_omics_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> OmicsInputInfo:

    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source)
    )
    omics_input_info = OmicsInputInfo(
        input_config=input_config, data_dimensions=data_dimensions
    )

    return omics_input_info


@dataclass
class DataDimensions:
    channels: int
    height: int
    width: int

    def num_elements(self):
        return self.channels * self.height * self.width


def get_data_dimension_from_data_source(
    data_source: Path,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    iterator = get_array_path_iterator(data_source=data_source)
    path = next(iterator)
    shape = np.load(file=path).shape

    if len(shape) == 1:
        channels, height, width = 1, 1, shape[0]
    elif len(shape) == 2:
        channels, height, width = 1, shape[0], shape[1]
    elif len(shape) == 3:
        channels, height, width = shape
    else:
        raise ValueError("Currently max 3 dimensional inputs supported")

    return DataDimensions(channels=channels, height=height, width=width)


def serialize_all_input_transformers(
    inputs_dict: al_input_objects_as_dict, run_folder: Path
):
    for input_name, input_ in inputs_dict.items():
        if input_name.startswith("tabular_"):
            save_transformer_set(
                transformers=input_.labels.label_transformers, run_folder=run_folder
            )


def serialize_all_sequence_inputs(
    inputs_dict: al_input_objects_as_dict, run_folder: Path
):
    for input_name, input_ in inputs_dict.items():
        if input_name.startswith("sequence_"):
            outpath = get_sequence_input_serialization_path(
                run_folder=run_folder, sequence_input_name=input_name
            )
            ensure_path_exists(path=outpath, is_folder=False)
            with open(outpath, "wb") as outfile:
                dill.dump(obj=input_, file=outfile)


def get_sequence_input_serialization_path(
    run_folder: Path, sequence_input_name: str
) -> Path:
    path = (
        run_folder
        / "serializations"
        / f"sequence_input_serializations/{sequence_input_name}.dill"
    )

    return path
