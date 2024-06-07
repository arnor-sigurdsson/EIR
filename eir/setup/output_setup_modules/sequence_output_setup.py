from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vocab

from eir.models.input.sequence.transformer_models import SequenceModelConfig
from eir.setup.input_setup_modules import setup_sequence
from eir.setup.schemas import InputConfig, OutputConfig, SequenceOutputTypeConfig
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.input_setup import al_input_objects_as_dict
    from eir.setup.output_setup import al_output_objects_as_dict

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class ComputedSequenceOutputInfo:
    output_config: OutputConfig
    vocab: Vocab
    embedding_dim: int
    computed_max_length: int
    encode_func: setup_sequence.al_encode_funcs
    tokenizer: Optional[setup_sequence.al_tokenizers]


def set_up_sequence_output(
    output_config: OutputConfig,
    input_objects: "al_input_objects_as_dict",
    *args,
    **kwargs,
) -> ComputedSequenceOutputInfo:
    output_name = output_config.output_info.output_name
    matching_seq_auto_set_up_input_object = input_objects[output_name]

    model_config = matching_seq_auto_set_up_input_object.input_config.model_config
    assert isinstance(model_config, SequenceModelConfig)

    embedding_dim = model_config.embedding_dim
    matching_input_config = matching_seq_auto_set_up_input_object.input_config

    sequence_input_object_func = get_sequence_input_objects_from_output

    vocab, gathered_stats, tokenizer, encode_callable = sequence_input_object_func(
        output_config=output_config, matching_input_config=matching_input_config
    )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, SequenceOutputTypeConfig)

    gathered_stats = setup_sequence.possibly_gather_all_stats_from_input(
        prev_gathered_stats=gathered_stats,
        input_source=output_config.output_info.output_source,
        vocab_file=output_type_info.vocab_file,
        split_on=output_type_info.split_on,
        max_length=output_type_info.max_length,
    )

    computed_max_length = setup_sequence.get_max_length(
        max_length_config_value=output_type_info.max_length,
        gathered_stats=gathered_stats,
    )

    sequence_output = ComputedSequenceOutputInfo(
        output_config=output_config,
        vocab=vocab,
        embedding_dim=embedding_dim,
        computed_max_length=computed_max_length,
        encode_func=encode_callable,
        tokenizer=tokenizer,
    )

    return sequence_output


def get_sequence_input_objects_from_output(
    output_config: OutputConfig, matching_input_config: InputConfig
) -> setup_sequence.al_sequence_input_objects_basic:
    gathered_stats = setup_sequence.GatheredSequenceStats()
    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, SequenceOutputTypeConfig)

    tokenizer, gathered_stats = setup_sequence.get_tokenizer(
        input_config=matching_input_config,
        gathered_stats=gathered_stats,
    )

    vocab = setup_sequence.init_vocab(
        source=output_config.output_info.output_source,
        inner_key=output_config.output_info.output_inner_key,
        tokenizer_name=output_type_info.tokenizer,
        split_on=output_type_info.split_on,
        vocab_file=output_type_info.vocab_file,
        min_freq=output_type_info.min_freq,
        gathered_stats=gathered_stats,
        tokenizer=tokenizer,
    )

    encode_func = setup_sequence.get_tokenizer_encode_func(
        tokenizer=tokenizer, pytorch_vocab=vocab
    )

    return vocab, gathered_stats, tokenizer, encode_func


def converge_sequence_input_and_output(
    inputs: "al_input_objects_as_dict",
    outputs: "al_output_objects_as_dict",
) -> "al_input_objects_as_dict":
    inputs_copy = deepcopy(inputs)

    for output_name, output_object in outputs.items():
        cur_out_type_info = output_object.output_config.output_type_info
        if getattr(cur_out_type_info, "sequence_operation", None) not in (
            "autoregressive",
            "mlm",
        ):
            continue

        logger.info(f"Converging input and output for {output_name}.")

        cur_input = inputs_copy[output_name]
        assert isinstance(cur_input, setup_sequence.ComputedSequenceInputInfo)
        assert isinstance(output_object, ComputedSequenceOutputInfo)

        cur_input.computed_max_length = output_object.computed_max_length
        cur_input.vocab = output_object.vocab
        cur_input.tokenizer = output_object.tokenizer
        cur_input.encode_func = output_object.encode_func

        inputs_copy[output_name] = cur_input

    return inputs_copy
