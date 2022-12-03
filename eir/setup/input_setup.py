from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import (
    Dict,
    Union,
    Generator,
    Sequence,
    Callable,
    Optional,
    Literal,
    Type,
    Hashable,
    Tuple,
    TYPE_CHECKING,
    Iterator,
    List,
    Any,
)

import pandas as pd
import numpy as np
from aislib.misc_utils import get_logger
from timm.models.registry import _model_pretrained_cfgs
from torchtext.data.utils import get_tokenizer as get_pytorch_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.vocab import vocab as pytorch_vocab_builder
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    TextInput,
    PreTokenizedInput,
    EncodedInput,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from eir.data_load.label_setup import (
    Labels,
    set_up_train_and_valid_tabular_data,
    TabularFileInfo,
    get_file_path_iterator,
)
from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    load_deeplake_dataset,
    get_deeplake_input_source_iterable,
)
from eir.experiment_io.experiment_io import (
    load_serialized_input_object,
    load_transformers,
    get_run_folder_from_model_path,
)
from eir.models.tabular.tabular import get_unique_values_from_transformers
from eir.setup import schemas
from eir.setup.schemas import al_tokenizer_choices
from eir.setup.setup_utils import collect_stats

if TYPE_CHECKING:
    from eir.train import Hooks

logger = get_logger(__name__)

al_input_objects = Union[
    "OmicsInputInfo",
    "TabularInputInfo",
    "SequenceInputInfo",
    "BytesInputInfo",
    "ImageInputInfo",
]
al_input_objects_as_dict = Dict[str, al_input_objects]
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

al_serializable_input_objects = Union[
    "SequenceInputInfo",
    "ImageInputInfo",
    "BytesInputInfo",
]

al_serializable_input_classes = Union[
    Type["SequenceInputInfo"],
    Type["ImageInputInfo"],
    Type["BytesInputInfo"],
]


def set_up_inputs_general(
    inputs_configs: schemas.al_input_configs,
    hooks: Union["Hooks", None],
    setup_func_getter: Callable[[schemas.InputConfig], Callable[..., al_input_objects]],
    setup_func_kwargs: Dict[str, Any],
) -> al_input_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_input_name_config_iterator(input_configs=inputs_configs)

    for name, input_config in name_config_iter:
        setup_func = setup_func_getter(input_config=input_config)

        cur_input_data_config = input_config.input_info
        logger.info(
            "Setting up %s inputs '%s' from %s.",
            cur_input_data_config.input_type,
            cur_input_data_config.input_name,
            cur_input_data_config.input_source,
        )

        set_up_input = setup_func(
            input_config=input_config, hooks=hooks, **setup_func_kwargs
        )
        all_inputs[name] = set_up_input

    return all_inputs


def set_up_inputs_for_training(
    inputs_configs: schemas.al_input_configs,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> al_input_objects_as_dict:

    train_input_setup_kwargs = {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
    }
    all_inputs = set_up_inputs_general(
        inputs_configs=inputs_configs,
        hooks=hooks,
        setup_func_getter=get_input_setup_function_for_train,
        setup_func_kwargs=train_input_setup_kwargs,
    )

    return all_inputs


def get_input_name_config_iterator(input_configs: schemas.al_input_configs):

    for input_config in input_configs:
        cur_input_data_config = input_config.input_info
        cur_name = cur_input_data_config.input_name
        yield cur_name, input_config


def get_input_setup_function_for_train(
    input_config: schemas.InputConfig,
) -> Callable[..., al_input_objects]:

    input_type = input_config.input_info.input_type
    pretrained_config = input_config.pretrained_config

    from_scratch_mapping = get_input_setup_function_map()

    if pretrained_config:
        pretrained_run_folder = get_run_folder_from_model_path(
            model_path=pretrained_config.model_path
        )
        from_pretrained_mapping = get_input_setup_from_pretrained_function_map(
            run_folder=pretrained_run_folder,
            load_module_name=pretrained_config.load_module_name,
        )
        return from_pretrained_mapping[input_type]

    return from_scratch_mapping[input_type]


def get_input_setup_function_map() -> Dict[str, Callable[..., al_input_objects]]:
    setup_mapping = {
        "omics": set_up_omics_input,
        "tabular": set_up_tabular_input_for_training,
        "sequence": set_up_sequence_input_for_training,
        "bytes": set_up_bytes_input_for_training,
        "image": set_up_image_input_for_training,
    }

    return setup_mapping


def set_up_tabular_input_from_pretrained(
    input_config: schemas.InputConfig,
    custom_input_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> "TabularInputInfo":

    tabular_input_object = set_up_tabular_input_for_training(
        input_config=input_config, train_ids=train_ids, valid_ids=valid_ids, hooks=hooks
    )

    pretrained_run_folder = get_run_folder_from_model_path(
        model_path=input_config.pretrained_config.model_path
    )

    loaded_transformers = load_transformers(
        run_folder=pretrained_run_folder, transformers_to_load=None
    )
    loaded_transformers_input = loaded_transformers[custom_input_name]

    tabular_input_object.labels.label_transformers = loaded_transformers_input

    return tabular_input_object


def get_input_setup_from_pretrained_function_map(
    run_folder: Path, load_module_name: str
) -> Dict[str, Callable]:
    pretrained_setup_mapping = {
        "omics": set_up_omics_input,
        "tabular": partial(
            set_up_tabular_input_from_pretrained, custom_input_name=load_module_name
        ),
        "sequence": partial(
            load_serialized_input_object,
            input_class=SequenceInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "bytes": partial(
            load_serialized_input_object,
            input_class=BytesInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
        "image": partial(
            load_serialized_input_object,
            input_class=ImageInputInfo,
            run_folder=run_folder,
            custom_input_name=load_module_name,
        ),
    }

    return pretrained_setup_mapping


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
        specials = _get_default_specials()

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


@dataclass
class PretrainedImageModelInfo:
    url: str
    num_classes: int
    input_size: Sequence[int]
    pool_size: Sequence[int]
    mean: Sequence[float]
    std: Sequence[float]
    first_conv: str
    classifier: str


def get_timm_configs() -> Dict[str, PretrainedImageModelInfo]:
    default_configs = {}
    field_names = {i.name for i in fields(PretrainedImageModelInfo)}
    for name, dict_ in _model_pretrained_cfgs.items():
        common = {k: v for k, v in dict_.items() if k in field_names}
        default_configs[name] = PretrainedImageModelInfo(**common)

    return default_configs


@dataclass
class ImageInputInfo:
    input_config: schemas.InputConfig
    base_transforms: Compose
    all_transforms: Compose
    normalization_stats: "ImageNormalizationStats"
    num_channels: int


def set_up_image_input_for_training(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ImageInputInfo:
    input_type_info = input_config.input_type_info

    num_channels = input_type_info.num_channels
    if not num_channels:
        num_channels = infer_num_channels(
            data_source=input_config.input_info.input_source,
            deeplake_inner_key=input_config.input_info.input_inner_key,
        )

    normalization_stats = get_image_normalization_values(input_config=input_config)

    base_transforms, all_transforms = get_image_transforms(
        target_size=input_config.input_type_info.size,
        normalization_stats=normalization_stats,
        auto_augment=input_type_info.auto_augment,
    )

    image_input_info = ImageInputInfo(
        input_config=input_config,
        base_transforms=base_transforms,
        all_transforms=all_transforms,
        normalization_stats=normalization_stats,
        num_channels=num_channels,
    )

    return image_input_info


def infer_num_channels(data_source: str, deeplake_inner_key: str) -> int:

    if is_deeplake_dataset(data_source=data_source):
        deeplake_ds = load_deeplake_dataset(data_source=data_source)
        deeplake_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        test_image_array = next(deeplake_iter).numpy()
        data_pointer = (
            f"[deeplake dataset {data_source}, input {deeplake_inner_key}, "
            f"image ID: {deeplake_ds['ID'][0].text()}]"
        )
    else:
        test_file = next(Path(data_source).iterdir())
        test_image = default_loader(path=str(test_file))
        test_image_array = np.array(test_image)
        data_pointer = test_file.name

    if test_image_array.ndim == 2:
        num_channels = 1
    else:
        num_channels = test_image_array.shape[-1]

    logger.info(
        "Inferring number of channels from source %s (using %s) as: %d",
        data_source,
        data_pointer,
        num_channels,
    )

    return num_channels


@dataclass
class ImageNormalizationStats:
    channel_means: Sequence[float]
    channel_stds: Sequence[float]


def get_image_normalization_values(
    input_config: schemas.InputConfig,
) -> ImageNormalizationStats:
    input_type_info = input_config.input_type_info

    pretrained_model_configs = get_timm_configs()

    means = input_type_info.mean_normalization_values
    stds = input_type_info.stds_normalization_values

    model_config = input_config.model_config

    if model_config.pretrained_model:
        cur_config = pretrained_model_configs[model_config.model_type]

        if not means:
            logger.info(
                "Using inferred image channel means (%s) from base on training "
                "statistics from pretrained '%s' model.",
                cur_config.mean,
                model_config.model_type,
            )
            means = cur_config.mean
        else:
            logger.warning(
                "Got manual values for channel means (%s) when using "
                "pretrained model '%s'. Usually one would use the means "
                "from the training data when '%s' was trained.",
                means,
                model_config.model_type,
                model_config.model_type,
            )
        if not stds:
            logger.info(
                "Using inferred image channel standard deviations (%s) from base on "
                "training statistics from pretrained '%s' model.",
                cur_config.std,
                model_config.model_type,
            )
            stds = cur_config.std
        else:
            logger.warning(
                "Got manual values for channel standard deviations (%s) "
                "when using pretrained model '%s'. Usually one would use "
                "the means from the training data when '%s' was trained.",
                stds,
                model_config.model_type,
                model_config.model_type,
            )
    else:
        if not means or not stds:
            input_source = input_config.input_info.input_source
            deeplake_inner_key = input_config.input_info.input_inner_key
            logger.info(
                "Not using a pretrained model and no mean and standard deviation "
                "statistics passed in. Gathering running image means and standard "
                "deviations from %s.",
                input_source,
            )

            if is_deeplake_dataset(data_source=input_source):
                deeplake_ds = load_deeplake_dataset(data_source=input_source)
                image_iter = get_deeplake_input_source_iterable(
                    deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
                )
                tensor_iterator = (to_tensor(i.numpy()) for i in image_iter)
            else:
                file_iterator = Path(input_source).rglob("*")
                image_iterator = (default_loader(str(f)) for f in file_iterator)
                tensor_iterator = (to_tensor(i) for i in image_iterator)

            gathered_stats = collect_stats(tensor_iterable=tensor_iterator)
            means = gathered_stats.mean
            stds = gathered_stats.std
            logger.info(
                "Gathered the following means: %s and standard deviations: %s "
                "from %s.",
                means,
                stds,
                input_source,
            )

    stats = ImageNormalizationStats(channel_means=means, channel_stds=stds)

    return stats


def get_image_transforms(
    target_size: Sequence[int],
    normalization_stats: ImageNormalizationStats,
    auto_augment: bool,
) -> Tuple[Compose, Compose]:
    random_transforms = transforms.TrivialAugmentWide()
    target_resize = [int(i * 1.5) for i in target_size]

    base = [
        transforms.Resize(size=target_resize),
        transforms.CenterCrop(size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalization_stats.channel_means,
            std=normalization_stats.channel_stds,
        ),
    ]

    base_transforms = transforms.Compose(transforms=base)
    if auto_augment:
        logger.info("Image will be auto augmented with TrivialAugment during training.")
        all_transforms = transforms.Compose(transforms=[random_transforms] + base)
    else:
        all_transforms = base_transforms

    return base_transforms, all_transforms


@dataclass
class SequenceInputInfo:
    input_config: schemas.InputConfig
    vocab: Vocab
    computed_max_length: int
    encode_func: Callable[[Sequence[str]], List[int]]
    tokenizer: Union[Callable, None] = None


def set_up_sequence_input_for_training(
    input_config: schemas.InputConfig, *args, **kwargs
) -> SequenceInputInfo:

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
    sequence_input_info = SequenceInputInfo(
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


@dataclass
class GatheredSequenceStats:
    total_count: int = 0
    total_files: int = 0
    max_length: int = 0


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
        specials=_get_default_specials(),
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

    default_specials = _get_default_specials()
    assert set(mapping.values()) == set(default_specials)

    return mapping


def _get_default_specials() -> List[str]:
    default_specials = ["<bos>", "<unk>", "<mask>", "<pad>", "<eos>"]
    return default_specials


def get_sequence_input_objects_from_pretrained(
    input_config: schemas.InputConfig,
) -> al_sequence_input_objects_hf:
    vocab_file = input_config.input_type_info.vocab_file
    if vocab_file:
        raise ValueError(
            "Using a vocabulary file not supported when using pre-trained models "
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

    hf_tokenizer_copy.add_special_tokens(specials_tokens_to_add)
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
        tokenizer.pre_tokenizer = Whitespace()

        special_tokens = _get_default_specials()
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
    gathered_stats.total_files += 1

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

        if cur_length > gathered_stats.max_length:
            gathered_stats.max_length = cur_length

        yield cur_line


def get_sequence_split_function(split_on: str) -> Callable[[str], List[str]]:
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
    total_num_features: int


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
        include_missing=True,
    )

    total_num_features = get_tabular_num_features(
        label_transformers=tabular_labels.label_transformers,
        cat_columns=input_config.input_type_info.input_cat_columns,
        con_columns=input_config.input_type_info.input_con_columns,
    )
    tabular_input_info = TabularInputInfo(
        labels=tabular_labels,
        input_config=input_config,
        total_num_features=total_num_features,
    )

    return tabular_input_info


def get_tabular_input_file_info(
    input_source: str,
    tabular_data_type_config: schemas.TabularInputDataConfig,
) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=Path(input_source),
        con_columns=tabular_data_type_config.input_con_columns,
        cat_columns=tabular_data_type_config.input_cat_columns,
        parsing_chunk_size=tabular_data_type_config.label_parsing_chunk_size,
    )

    return table_info


def get_tabular_num_features(
    label_transformers: Dict, cat_columns: Sequence[str], con_columns: Sequence[str]
) -> int:
    unique_cat_values = get_unique_values_from_transformers(
        transformers=label_transformers,
        keys_to_use=cat_columns,
    )
    cat_num_features = sum(
        len(unique_values) for unique_values in unique_cat_values.values()
    )
    total_num_features = cat_num_features + len(con_columns)

    return total_num_features


@dataclass
class OmicsInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"
    subset_indices: Union[None, Sequence[int]]


def set_up_omics_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> OmicsInputInfo:

    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source),
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    subset_indices = None
    input_type_info = input_config.input_type_info
    if input_type_info.subset_snps_file:
        df_bim = read_bim(bim_file_path=input_type_info.snp_file)
        snps_to_subset = read_subset_file(
            subset_snp_file_path=input_type_info.subset_snps_file
        )
        subset_indices = _setup_snp_subset_indices(
            df_bim=df_bim,
            snps_to_subset=snps_to_subset,
            snp_file_name=input_type_info.snp_file,
            subset_file_name=input_type_info.subset_snps_file,
        )
        data_dimensions = DataDimensions(
            channels=data_dimensions.channels,
            height=data_dimensions.height,
            width=len(subset_indices),
        )

    omics_input_info = OmicsInputInfo(
        input_config=input_config,
        data_dimensions=data_dimensions,
        subset_indices=subset_indices,
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
    deeplake_inner_key: Optional[str] = None,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    if is_deeplake_dataset(data_source=str(data_source)):
        assert deeplake_inner_key is not None, data_source
        deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
        deeplake_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        shape = next(deeplake_iter).shape
    else:
        iterator = get_file_path_iterator(data_source=data_source)
        path = next(iterator)
        shape = np.load(file=path).shape

    if len(shape) == 1:
        channels, height, width = 1, 1, shape[0]
    elif len(shape) == 2:
        channels, height, width = 1, shape[0], shape[1]
    elif len(shape) == 3:
        channels, height, width = shape
    else:
        raise ValueError("Currently max 3 dimensional inputs supported.")

    return DataDimensions(channels=channels, height=height, width=width)


def _setup_snp_subset_indices(
    df_bim: pd.DataFrame,
    snps_to_subset: List[str],
    subset_file_name: str = "",
    snp_file_name: str = "",
) -> np.ndarray:
    """
    .bim columns: ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    """

    df_subset = df_bim[df_bim["VAR_ID"].isin(snps_to_subset)]

    if len(df_subset) < len(snps_to_subset):
        num_missing = len(snps_to_subset) - len(df_subset)
        missing = [i for i in snps_to_subset if i not in df_subset["VAR_ID"]]
        logger.warning(
            "Did not find all SNPs in subset file '%s' in base .bim file '%s'. "
            "Number of missing SNPs: %d. Example: '%s'.",
            subset_file_name,
            snp_file_name,
            num_missing,
            missing[:3],
        )
    else:
        logger.info(
            "Using %d SNPs from subset file %s.", len(df_subset), subset_file_name
        )

    return df_subset.index


def read_subset_file(subset_snp_file_path: str) -> List[str]:
    with open(subset_snp_file_path, "r") as infile:
        snps_to_subset = infile.read().split()

    return snps_to_subset


def read_bim(bim_file_path: str) -> pd.DataFrame:
    bim_headers = _get_bim_headers()
    df_bim = pd.read_csv(bim_file_path, names=bim_headers, sep=r"\s+")
    df_bim["VAR_ID"] = df_bim["VAR_ID"].astype(str)

    if not len(df_bim.columns) == 6:
        raise ValueError(
            "Expected 6 columns in bim file '%s', got %d.",
            bim_file_path,
            len(df_bim.columns),
        )

    return df_bim


def _get_bim_headers() -> List[str]:
    bim_headers = ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    return bim_headers
