from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterator, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from aislib.misc_utils import ensure_path_exists
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from eir.data_load.datasets import al_getitem_return
from eir.models.model_training_utils import predict_on_batch
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
)
from eir.setup.schemas import OutputConfig, SequenceOutputTypeConfig
from eir.train_utils import utils
from eir.train_utils.evaluation_handlers.evaluation_handlers_utils import (
    convert_model_inputs_to_raw,
    decode_tokens,
    extract_input_types,
    general_pre_process_prepared_inputs,
    get_dataset_loader_single_sample_generator,
    get_special_tokens,
    post_prepare_manual_inputs,
    prepare_base_input,
    prepare_manual_sample_data,
    remove_special_tokens_from_string,
    serialize_raw_inputs,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.predict import PredictExperiment, PredictHooks
    from eir.serve import ServeExperiment
    from eir.train import Experiment, al_input_objects_as_dict
    from eir.train_utils.step_logic import Hooks

logger = get_logger(name=__name__)


@dataclass
class SequenceOutputEvalSamples:
    auto_samples: Dict[str, list["SequenceOutputEvalSample"]]
    manual_samples: Dict[str, list["SequenceOutputEvalSample"]]


@dataclass()
class SequenceOutputEvalSample:
    inputs_to_model: Dict[str, Any]
    target_labels: Dict[str, Any]
    sample_id: str


def sequence_out_single_sample_evaluation_wrapper(
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    input_objects: "al_input_objects_as_dict",
    auto_dataset_to_load_from: Dataset,
    iteration: int,
    output_folder: str,
) -> None:
    default_eir_hooks = experiment.hooks
    assert default_eir_hooks is not None

    output_configs = experiment.configs.output_configs

    if not any(i.sampling_config for i in output_configs):
        return

    manual_samples = get_sequence_output_manual_input_samples(
        output_configs=output_configs, input_objects=input_objects
    )

    auto_validation_generator = get_dataset_loader_single_sample_generator(
        dataset=auto_dataset_to_load_from
    )
    auto_samples = get_sequence_output_auto_validation_samples(
        output_configs=output_configs,
        input_objects=input_objects,
        eval_sample_iterator=auto_validation_generator,
    )
    eval_samples = SequenceOutputEvalSamples(
        auto_samples=auto_samples, manual_samples=manual_samples
    )

    for config in output_configs:
        if config.output_info.output_type != "sequence":
            continue

        cur_input_name = config.output_info.output_name
        cur_output_name = config.output_info.output_name

        not_in_manual_samples = cur_output_name not in eval_samples.manual_samples
        not_in_auto_samples = cur_output_name not in eval_samples.auto_samples
        if not_in_manual_samples and not_in_auto_samples:
            continue

        cur_sample_output_folder = utils.prepare_sample_output_folder(
            output_folder=output_folder,
            output_name=cur_output_name,
            column_name=cur_input_name,
            iteration=iteration,
        )

        output_type_info = config.output_type_info
        assert isinstance(output_type_info, SequenceOutputTypeConfig)

        assert config.sampling_config is not None

        if output_type_info.sequence_operation == "autoregressive":
            sample_generator = _get_eval_sample_generator(
                eval_samples=eval_samples, output_name=cur_output_name
            )
            assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

            for idx, (eval_type, eval_sample) in enumerate(sample_generator):
                generated_tokens = autoregressive_sequence_generation(
                    input_objects=input_objects,
                    eval_sample=eval_sample,
                    seq_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    sampling_config=config.sampling_config,
                )

                cur_input_object = input_objects[cur_input_name]
                assert isinstance(cur_input_object, ComputedSequenceInputInfo)
                assert cur_input_object.tokenizer is not None

                generated_sample = decode_tokens(
                    tokens=generated_tokens,
                    vocab=cur_input_object.vocab,
                    split_on=output_type_info.split_on,
                )
                special_tokens = get_special_tokens(
                    tokenizer=cur_input_object.tokenizer,
                    vocab=cur_input_object.vocab,
                )
                generated_sample = remove_special_tokens_from_string(
                    string=generated_sample,
                    special_tokens=special_tokens,
                )

                cur_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_generated.txt"
                )
                ensure_path_exists(path=cur_output_path)
                cur_output_path.write_text(data=generated_sample)

                cur_inputs_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_inputs"
                )

                raw_inputs = convert_model_inputs_to_raw(
                    inputs_to_model=eval_sample.inputs_to_model,
                    input_objects=input_objects,
                )

                serialize_raw_inputs(
                    raw_inputs=raw_inputs,
                    input_objects=input_objects,
                    output_path=cur_inputs_output_path,
                )


def get_sequence_output_manual_input_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, list[SequenceOutputEvalSample]]:
    prepared_samples: dict[str, list[SequenceOutputEvalSample]] = {}

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "sequence":
            continue

        assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

        sample_data_from_yaml = config.sampling_config.manual_inputs
        output_name = config.output_info.output_name

        prepared_samples[output_name] = []

        for idx, single_sample_inputs in enumerate(sample_data_from_yaml):
            prepared_inputs = prepare_manual_sample_data(
                sample_inputs=single_sample_inputs,
                input_objects=input_objects,
            )

            final_inputs = post_prepare_manual_inputs(
                prepared_inputs=prepared_inputs,
                output_name=output_name,
                input_objects=input_objects,
            )

            cur_eval_sample = SequenceOutputEvalSample(
                inputs_to_model=final_inputs,
                target_labels={},
                sample_id=f"manual_{idx}",
            )

            prepared_samples[output_name].append(cur_eval_sample)

    return prepared_samples


def get_sequence_output_auto_validation_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
    eval_sample_iterator: Iterator[al_getitem_return],
) -> Dict[str, list[SequenceOutputEvalSample]]:
    prepared_eval_samples: dict[str, list[SequenceOutputEvalSample]] = {}
    input_types = extract_input_types(input_objects=input_objects)

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "sequence":
            continue

        assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        for i in range(config.sampling_config.n_eval_inputs):
            input_to_model, target_labels, cur_id = next(eval_sample_iterator)

            cur_inputs_masked = _mask_targets_for_auto_eval_generation(
                inputs=input_to_model,
                output_name=output_name,
                input_types=input_types,
            )

            cur_eval_sample = SequenceOutputEvalSample(
                inputs_to_model=cur_inputs_masked,
                target_labels=target_labels,
                sample_id=cur_id,
            )

            prepared_eval_samples[output_name].append(cur_eval_sample)

    return prepared_eval_samples


def _mask_targets_for_auto_eval_generation(
    inputs: Dict[str, Any],
    output_name: str,
    input_types: Dict[str, str],
) -> Dict[str, Any]:
    raw_inputs_masked = {}

    for input_name, raw_input in inputs.items():
        if input_name == output_name:
            if input_types[input_name] == "sequence":
                raw_inputs_masked[output_name] = torch.tensor([[]], dtype=torch.long)
            else:
                raise NotImplementedError()

        else:
            raw_inputs_masked[input_name] = raw_input

    return raw_inputs_masked


def _get_eval_sample_generator(
    eval_samples: SequenceOutputEvalSamples, output_name: str
) -> Generator[Tuple[str, SequenceOutputEvalSample], None, None]:
    cur_config_auto_samples = eval_samples.auto_samples[output_name]
    cur_config_manual_samples = eval_samples.manual_samples[output_name]

    for eval_sample in cur_config_auto_samples:
        yield "auto", eval_sample

    for eval_sample in cur_config_manual_samples:
        yield "manual", eval_sample


@torch.inference_mode()
def autoregressive_sequence_generation(
    input_objects: "al_input_objects_as_dict",
    eval_sample: SequenceOutputEvalSample,
    seq_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
    sampling_config: SequenceOutputSamplingConfig,
) -> list[int]:
    """
    Note that it's currently a bit weird / perhaps suboptimal how we are dealing
    with the input data here. While normally one could have the final, prepared
    input (including embeddings) as a direct input to the model, we have here
    what is more similar to what is returned from the data loader, namely e.g.
    token IDs. The sequence of IDs is then used in the loop and appended to,
    which in turn goes through the general batch preparation hook, which e.g.
    looks up the embeddings.
    """
    output_object = experiment.outputs[seq_output_name]
    input_object = input_objects[seq_output_name]

    assert isinstance(output_object, ComputedSequenceOutputInfo)
    assert isinstance(input_object, ComputedSequenceInputInfo)

    _check_vocab_consistency(
        output_object_vocab=output_object.vocab,
        input_object_vocab=input_object.vocab,
    )

    assert output_object.tokenizer is not None
    st = get_special_tokens(
        tokenizer=output_object.tokenizer,
        vocab=output_object.vocab,
    )

    prepared_sample_inputs = prepare_base_input(
        prepared_inputs=eval_sample.inputs_to_model,
    )

    generated_tokens_base = _extract_base_generated_tokens(
        prepared_inputs=prepared_sample_inputs,
        seq_output_name=seq_output_name,
    )

    generated_tokens = _ensure_no_extra_padding(
        tokens=generated_tokens_base,
        pad_idx=st.pad_idx,
    )

    for i in range(len(generated_tokens), sampling_config.generated_sequence_length):
        prepared_sample_inputs = _prepare_current_autoregressive_input(
            prepared_sample_inputs=prepared_sample_inputs,
            generated_tokens=generated_tokens,
            seq_output_name=seq_output_name,
            max_length=output_object.computed_max_length,
            pad_idx=st.pad_idx,
        )

        target_index = _compute_target_index(
            current_generated_length=i,
            max_length=output_object.computed_max_length,
        )

        batch = general_pre_process_prepared_inputs(
            prepared_inputs=prepared_sample_inputs,
            target_labels=eval_sample.target_labels,
            sample_id=eval_sample.sample_id,
            experiment=experiment,
            custom_hooks=default_eir_hooks,
        )

        outputs = predict_on_batch(
            model=experiment.model,
            inputs=batch.inputs,
        )

        next_token_index = sample_next_token_index_from_output(
            outputs=outputs,
            seq_output_name=seq_output_name,
            sampling_config=sampling_config,
            current_target_index=target_index,
        )

        if next_token_index == st.eos_idx:
            break

        generated_tokens.append(next_token_index)

    return generated_tokens


def _check_vocab_consistency(
    output_object_vocab: Vocab, input_object_vocab: Vocab
) -> None:
    assert output_object_vocab.get_stoi() == input_object_vocab.get_stoi()
    assert output_object_vocab.get_itos() == input_object_vocab.get_itos()


def _extract_base_generated_tokens(
    prepared_inputs: Dict[str, torch.Tensor], seq_output_name: str
) -> list[int]:
    """
    Note that we are expecting / enforcing the token IDs being passed in here,
    not the embeddings.
    """
    tensor_base = prepared_inputs[seq_output_name]
    assert tensor_base.dim() == 2, tensor_base.dim()

    list_base = tensor_base.squeeze(0).tolist()
    assert isinstance(list_base, list)

    return list_base


def _ensure_no_extra_padding(tokens: list[int], pad_idx: int) -> list[int]:
    """
    Needed in case the sequence is already at max due to padding, as the autoregressive
    functionality itself takes care of shifting, bos insertion, padding etc. So, it
    expects just the sequence as given, without any padding. However, some processing
    code might have added padding to the end, so we have this little safeguard here.
    """
    generated_tokens = [i for i in tokens if i != pad_idx]
    if generated_tokens:
        assert generated_tokens[-1] != pad_idx

    return generated_tokens


def _compute_target_index(current_generated_length: int, max_length: int) -> int:
    if current_generated_length < max_length:
        return current_generated_length
    else:
        return max_length - 1


def _prepare_current_autoregressive_input(
    prepared_sample_inputs: dict[str, Any],
    generated_tokens: list[int],
    seq_output_name: str,
    max_length: int,
    pad_idx: int,
) -> dict[str, torch.Tensor]:
    """
    Assuming max_length of 5:

    Truncate (from start):
        - [A, B, C, D, E] -> [B, C, D, E]
    Pad:
        - [B, C, D, E] -> [B, C, D, E, pad]
    In autoregressive batch preparation hook:
        - [B, C, D, E, pad] -> [bos, B, C, D, E, pad] -> [bos, B, C, D, E]

    So essentially the pad is a placeholder to that we need here due to how
    the batch is later (in the batch preparation hook) prepared, namely
    as it inserts the bos at the beginning and cuts off at the end, so in the case
    where the sequence is already at max_length, we need to pad it to make sure
    that the latest token is preserved.
    """
    current_sequence = torch.tensor(generated_tokens, dtype=torch.long)
    current_sequence = _maybe_truncate_autoregressive_sequence(
        current_sequence=current_sequence,
        max_length=max_length,
    )

    assert current_sequence.dim() == 1, current_sequence.shape

    pad_size = max_length - len(current_sequence)
    assert pad_size >= 1, pad_size

    current_sequence = F.pad(current_sequence, (0, pad_size), value=pad_idx)
    current_sequence = current_sequence.unsqueeze(0)

    assert current_sequence.dim() == 2, current_sequence.shape

    current_inputs_copy = copy(prepared_sample_inputs)
    current_inputs_copy[seq_output_name] = current_sequence

    return current_inputs_copy


def _maybe_truncate_autoregressive_sequence(
    current_sequence: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    +1 to account for bos token being inserted and last token removed later.
    If already at max length, we truncate from the start and leave 1 element
    missing (+1) to allow for padding.
    """
    sequence_length = len(current_sequence)

    if sequence_length >= max_length:
        start = sequence_length - max_length + 1
        current_sequence = current_sequence[start:]

    return current_sequence


def sample_next_token_index_from_output(
    outputs: dict[str, dict[str, torch.Tensor]],
    seq_output_name: str,
    sampling_config: SequenceOutputSamplingConfig,
    current_target_index: int,
) -> int:
    cur_logits = outputs[seq_output_name][seq_output_name].squeeze()
    cur_position_logits = cur_logits[current_target_index]

    filtered_logits = top_k_top_p_filtering(
        logits=cur_position_logits,
        top_k=sampling_config.top_k,
        top_p=sampling_config.top_p,
    )
    probabilities = F.softmax(input=filtered_logits, dim=-1)
    next_token_index = torch.multinomial(input=probabilities, num_samples=1).item()

    return int(next_token_index)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits
