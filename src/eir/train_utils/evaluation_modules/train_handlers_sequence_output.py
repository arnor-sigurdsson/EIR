import json
from collections import Counter
from collections.abc import Generator, Iterator, Sequence
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import torch
import torch.nn.functional as F
from aislib.misc_utils import ensure_path_exists
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.datasets import al_getitem_return
from eir.models.model_training_utils import predict_on_batch, recursive_to_device
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    SpecialTokens,
    get_special_tokens,
)
from eir.setup.input_setup_modules.torchtext_port.vocab import Vocab
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
)
from eir.setup.schemas import OutputConfig, SequenceOutputTypeConfig
from eir.train_utils import utils
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import (
    convert_model_inputs_to_raw,
    decode_tokens,
    extract_input_types,
    general_pre_process_prepared_inputs,
    get_batch_generator,
    get_dataset_loader_single_sample_generator,
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
    auto_samples: dict[str, list["SequenceOutputEvalSample"]]
    manual_samples: dict[str, list["SequenceOutputEvalSample"]]


@dataclass()
class SequenceOutputEvalSample:
    inputs_to_model: dict[str, Any]
    target_labels: dict[str, Any]
    sample_id: str


def sequence_out_single_sample_evaluation_wrapper(
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    input_objects: "al_input_objects_as_dict",
    auto_dataset_to_load_from: Dataset,
    iteration: int,
    output_folder: str,
) -> None:
    gc = experiment.configs.global_config

    default_eir_hooks = experiment.hooks
    assert default_eir_hooks is not None

    output_configs = experiment.configs.output_configs

    if not any(i.sampling_config for i in output_configs):
        return

    manual_samples = get_sequence_output_manual_input_samples(
        output_configs=output_configs,
        input_objects=input_objects,
    )

    auto_validation_generator = get_dataset_loader_single_sample_generator(
        dataset=auto_dataset_to_load_from,
        fabric=experiment.fabric,
    )
    auto_samples = get_sequence_output_auto_validation_samples(
        output_configs=output_configs,
        input_objects=input_objects,
        eval_sample_iterator=auto_validation_generator,
    )
    eval_samples_base = SequenceOutputEvalSamples(
        auto_samples=auto_samples,
        manual_samples=manual_samples,
    )

    for config in output_configs:
        if config.output_info.output_type != "sequence":
            continue

        cur_input_name = config.output_info.output_name
        cur_output_name = config.output_info.output_name

        not_in_manual_samples = cur_output_name not in eval_samples_base.manual_samples
        not_in_auto_samples = cur_output_name not in eval_samples_base.auto_samples
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
                eval_samples=eval_samples_base,
                output_name=cur_output_name,
            )

            batch_generator = get_batch_generator(
                iterator=enumerate(sample_generator),
                batch_size=gc.be.batch_size,
            )

            assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

            meta = {}

            for batch in batch_generator:
                batch_indices, batch_eval_data = zip(*batch, strict=False)
                batch_eval_types, batch_eval_samples = zip(
                    *batch_eval_data, strict=False
                )

                batch_generated_tokens = autoregressive_sequence_generation(
                    input_objects=input_objects,
                    eval_samples=batch_eval_samples,
                    seq_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    sampling_config=config.sampling_config,
                )

                cur_input_object = input_objects[cur_input_name]
                assert isinstance(cur_input_object, ComputedSequenceInputInfo)
                assert cur_input_object.tokenizer is not None

                for eval_type, idx, eval_sample, generated_tokens in zip(
                    batch_eval_types,
                    batch_indices,
                    batch_eval_samples,
                    batch_generated_tokens,
                    strict=False,
                ):
                    generated_sample = decode_tokens(
                        tokens=generated_tokens,
                        vocab=cur_input_object.vocab,
                        split_on=output_type_info.split_on,
                        tokenizer=cur_input_object.tokenizer,
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

                    cur_id = eval_sample.sample_id
                    meta[cur_id] = {
                        "generated": str(cur_output_path.relative_to(output_folder)),
                        "inputs": str(
                            cur_inputs_output_path.relative_to(output_folder)
                        ),
                        "index": idx,
                    }

                meta_path = cur_sample_output_folder / "meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)


def get_sequence_output_manual_input_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> dict[str, list[SequenceOutputEvalSample]]:
    prepared_samples: dict[str, list[SequenceOutputEvalSample]] = {}

    for _config_idx, config in enumerate(output_configs):
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

            imputed_inputs = impute_missing_modalities_wrapper(
                inputs_values=prepared_inputs,
                inputs_objects=input_objects,
            )

            final_inputs = post_prepare_manual_inputs(
                prepared_inputs=imputed_inputs,
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
) -> dict[str, list[SequenceOutputEvalSample]]:
    prepared_eval_samples: dict[str, list[SequenceOutputEvalSample]] = {}
    input_types = extract_input_types(input_objects=input_objects)

    for _config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "sequence":
            continue

        assert isinstance(config.sampling_config, SequenceOutputSamplingConfig)

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        for _i in range(config.sampling_config.n_eval_inputs):
            input_to_model, target_labels, cur_id = next(eval_sample_iterator)

            cur_inputs_masked = _mask_targets_for_auto_eval_generation(
                inputs=input_to_model,
                output_name=output_name,
                input_types=input_types,
            )

            cur_eval_sample = SequenceOutputEvalSample(
                inputs_to_model=cur_inputs_masked,
                target_labels=target_labels,
                sample_id=cur_id[0],
            )

            prepared_eval_samples[output_name].append(cur_eval_sample)

    return prepared_eval_samples


def _mask_targets_for_auto_eval_generation(
    inputs: dict[str, Any],
    output_name: str,
    input_types: dict[str, str],
) -> dict[str, Any]:
    raw_inputs_masked = {}

    for input_name, raw_input in inputs.items():
        if input_name == output_name:
            if input_types[input_name] == "sequence":
                raw_inputs_masked[output_name] = torch.tensor([], dtype=torch.long)
            else:
                raise NotImplementedError()

        else:
            raw_inputs_masked[input_name] = raw_input

    return raw_inputs_masked


def _get_eval_sample_generator(
    eval_samples: SequenceOutputEvalSamples, output_name: str
) -> Generator[tuple[str, SequenceOutputEvalSample]]:
    cur_config_auto_samples = eval_samples.auto_samples[output_name]
    cur_config_manual_samples = eval_samples.manual_samples[output_name]

    for eval_sample in cur_config_auto_samples:
        yield "auto", eval_sample

    for eval_sample in cur_config_manual_samples:
        yield "manual", eval_sample


@torch.inference_mode()
def autoregressive_sequence_generation(
    input_objects: "al_input_objects_as_dict",
    eval_samples: tuple[SequenceOutputEvalSample, ...],
    seq_output_name: str,
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    default_eir_hooks: Union["Hooks", "PredictHooks"],
    sampling_config: SequenceOutputSamplingConfig,
) -> list[list[int]]:
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
    max_length = sampling_config.generated_sequence_length

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

    device = str(experiment.fabric.device)
    autoregressive_pre_batch = prepare_autoregressive_sampling_batch(
        eval_samples=eval_samples,
        seq_output_name=seq_output_name,
        special_tokens=st,
        device=device,
    )
    prepared_sample_inputs = autoregressive_pre_batch.prepared_inputs
    prepared_targets = autoregressive_pre_batch.prepared_targets
    generated_tokens = autoregressive_pre_batch.generated_tokens
    indices = autoregressive_pre_batch.indices

    indices_has_finished: dict[int, bool] = {}
    for _i in range(0, sampling_config.generated_sequence_length):
        if len(indices_has_finished) == len(eval_samples):
            break

        target_indices = []
        for sample_index, _ in enumerate(eval_samples):
            cur_generated_tokens = generated_tokens[sample_index]

            cur_prepared_sample_inputs = prepared_sample_inputs[sample_index]
            cur_prepared_sample_inputs = _prepare_current_autoregressive_input(
                prepared_sample_inputs=cur_prepared_sample_inputs,
                generated_tokens=cur_generated_tokens,
                seq_output_name=seq_output_name,
                max_length=output_object.computed_max_length,
                pad_idx=st.pad_idx,
            )
            cur_prepared_sample_inputs = recursive_to_device(
                obj=cur_prepared_sample_inputs,
                device=device,
            )

            cur_target_index = _compute_target_index(
                current_generated_length=indices[sample_index],
                max_length=output_object.computed_max_length,
            )

            target_indices.append(cur_target_index)
            prepared_sample_inputs[sample_index] = cur_prepared_sample_inputs
            indices[sample_index] += 1

        all_inputs = default_collate(prepared_sample_inputs)
        all_targets = default_collate(prepared_targets)
        all_ids = [eval_sample.sample_id for eval_sample in eval_samples]

        batch = general_pre_process_prepared_inputs(
            prepared_inputs=all_inputs,
            target_labels=all_targets,
            sample_ids=all_ids,
            experiment=experiment,
            custom_hooks=default_eir_hooks,
        )

        outputs = predict_on_batch(
            model=experiment.model,
            inputs=batch.inputs,
        )

        next_token_indices = sample_next_token_index_from_output(
            outputs=outputs,
            seq_output_name=seq_output_name,
            sampling_config=sampling_config,
            current_target_indices=target_indices,
            generated_tokens_history=generated_tokens,
        )

        for sample_index, _ in enumerate(eval_samples):
            if sample_index in indices_has_finished:
                continue

            next_token_index = next_token_indices[sample_index]

            is_eos = next_token_index == st.eos_idx
            is_at_max_length = indices[sample_index] == max_length

            if is_eos or is_at_max_length:
                indices_has_finished[sample_index] = True
                continue

            generated_tokens[sample_index].append(next_token_index)

    return generated_tokens


@dataclass(frozen=False)
class AutoRegressiveSamplingBatch:
    prepared_inputs: list[dict[str, torch.Tensor]]
    prepared_targets: list[dict[str, torch.Tensor]]
    indices: list[int]
    generated_tokens: list[list[int]]


def prepare_autoregressive_sampling_batch(
    eval_samples: Sequence[SequenceOutputEvalSample],
    seq_output_name: str,
    special_tokens: SpecialTokens,
    device: str,
) -> AutoRegressiveSamplingBatch:
    prepared_sample_inputs = []
    prepared_targets = []
    generated_tokens = []
    indices = []

    for eval_sample in eval_samples:
        cur_inputs = eval_sample.inputs_to_model
        cur_prepared = prepare_base_input(prepared_inputs=cur_inputs)
        prepared_sample_inputs.append(cur_prepared)
        prepared_targets.append(eval_sample.target_labels)

        cur_generated_tokens_base = _extract_base_generated_tokens(
            prepared_inputs=cur_prepared,
            seq_output_name=seq_output_name,
        )

        cur_generated_tokens = _ensure_no_extra_padding(
            tokens=cur_generated_tokens_base,
            pad_idx=special_tokens.pad_idx,
        )

        generated_tokens.append(cur_generated_tokens)
        indices.append(len(cur_generated_tokens))

    prepared_sample_inputs = recursive_to_device(
        obj=prepared_sample_inputs,
        device=device,
    )

    prepared_targets = recursive_to_device(
        obj=prepared_targets,
        device=device,
    )

    return AutoRegressiveSamplingBatch(
        prepared_inputs=prepared_sample_inputs,
        prepared_targets=prepared_targets,
        indices=indices,
        generated_tokens=generated_tokens,
    )


def _check_vocab_consistency(
    output_object_vocab: Vocab, input_object_vocab: Vocab
) -> None:
    assert output_object_vocab.get_stoi() == input_object_vocab.get_stoi()
    assert output_object_vocab.get_itos() == input_object_vocab.get_itos()


def _extract_base_generated_tokens(
    prepared_inputs: dict[str, torch.Tensor],
    seq_output_name: str,
) -> list[int]:
    """
    Note that we are expecting / enforcing the token IDs being passed in here,
    not the embeddings.
    """
    tensor_base = prepared_inputs[seq_output_name]

    assert tensor_base.dim() == 1, (tensor_base, tensor_base.dim())

    list_base = tensor_base.tolist()
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

    The reason for padding with the BOS token is that the tensor we get here
    is already at max_length. If we had e.g. a full, long sequence, we could
    simply slice that directly (+1 for the target), but here we need to pad
    the input at the beginning, opting for a BOS token.
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
    assert current_sequence.dim() == 1, current_sequence.shape

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
    current_target_indices: list[int],
    generated_tokens_history: list[list[int]] | None = None,
) -> list[int]:
    cur_logits = outputs[seq_output_name][seq_output_name]
    batch_indices = torch.arange(cur_logits.size(0))
    cur_position_logits = cur_logits[batch_indices, current_target_indices, :]

    repetition_penalty = sampling_config.repetition_penalty
    if generated_tokens_history is not None and repetition_penalty != 1.0:
        for i in range(cur_position_logits.size(0)):
            cur_position_logits[i] = apply_repetition_penalty(
                logits=cur_position_logits[i],
                generated_tokens=generated_tokens_history[i],
                penalty=repetition_penalty,
                max_window=sampling_config.repetition_penalty_max_window,
            )

    frequency_penalty = sampling_config.frequency_penalty
    if generated_tokens_history is not None and repetition_penalty > 0.0:
        for i in range(cur_position_logits.size(0)):
            cur_position_logits[i] = apply_frequency_penalty(
                logits=cur_position_logits[i],
                generated_tokens=generated_tokens_history[i],
                penalty=frequency_penalty,
                max_window=sampling_config.frequency_penalty_max_window,
            )

    temperature = sampling_config.temperature
    if temperature != 1.0:
        cur_position_logits = cur_position_logits / temperature

    filtered_logits = top_k_top_p_filtering(
        logits=cur_position_logits,
        top_k=sampling_config.top_k,
        top_p=sampling_config.top_p,
    )

    tau = sampling_config.tau
    filter_value = -float("Inf")
    if tau < 1.0:
        valid_tokens_mask = filtered_logits != filter_value
        valid_tokens_count = valid_tokens_mask.sum(dim=-1)

        batch_indices = torch.where(valid_tokens_count > 1)[0]
        if len(batch_indices) > 0:
            batch_logits = filtered_logits[batch_indices]
            batch_filtered = locally_typical_sampling(
                logits=batch_logits,
                tau=tau,
                filter_value=filter_value,
            )
            filtered_logits[batch_indices] = batch_filtered

    probabilities = F.softmax(input=filtered_logits, dim=-1)
    next_token_indices = torch.multinomial(input=probabilities, num_samples=1)
    next_token_indices_list = next_token_indices.squeeze(1).tolist()

    return next_token_indices_list


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: list[int],
    penalty: float = 1.2,
    max_window: int = 64,
) -> torch.Tensor:
    if not generated_tokens:
        return logits

    generated_tokens = generated_tokens[-max_window:]
    unique_tokens = torch.tensor(list(set(generated_tokens)), device=logits.device)

    valid_tokens = unique_tokens[unique_tokens < logits.size(0)]

    if len(valid_tokens) == 0:
        return logits

    logits_modified = logits.clone()

    token_logits = logits_modified[valid_tokens]

    token_logits = torch.where(
        token_logits >= 0,
        token_logits / penalty,
        token_logits * penalty,
    )

    logits_modified[valid_tokens] = token_logits

    return logits_modified


def apply_frequency_penalty(
    logits: torch.Tensor,
    generated_tokens: list[int],
    penalty: float = 0.1,
    max_window: int = 128,
) -> torch.Tensor:
    if not generated_tokens or penalty <= 0.0:
        return logits

    recent_tokens = generated_tokens[-max_window:]

    token_counts = Counter(recent_tokens)

    vocab_size = logits.size(0)
    valid_tokens = []
    penalties = []

    for token, count in token_counts.items():
        if token < vocab_size and count > 1:
            valid_tokens.append(token)
            penalties.append(penalty * (1.0 - 1.0 / count))

    if not valid_tokens:
        return logits

    valid_tokens_tensor = torch.tensor(valid_tokens, device=logits.device)
    penalties_tensor = torch.tensor(penalties, device=logits.device)

    logits_modified = logits.clone()

    logits_modified[valid_tokens_tensor] -= penalties_tensor

    return logits_modified


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    assert logits.dim() == 2, (
        f"Expected 2D tensor (batch, vocab_scores). Got: {logits.dim()}D."
    )

    batch_size, vocab_size = logits.size()

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

        for i in range(batch_size):
            indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = filter_value

    return logits


def locally_typical_sampling(
    logits: torch.Tensor,
    tau: float = 0.95,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    filtered_logits = logits.clone()

    probs = F.softmax(filtered_logits, dim=-1)

    eps = 1e-10

    # Calculate entropy of the distribution
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1, keepdim=True)

    # Calculate typicality score
    typical_score = torch.abs(torch.log(probs + eps) + entropy)

    # Keep only tokens with typicality score below threshold
    sorted_scores, _ = torch.sort(typical_score, dim=-1)

    for i in range(filtered_logits.size(0)):
        num_tokens = sorted_scores[i].size(-1)
        threshold_idx = int((num_tokens - 1) * tau)

        threshold = sorted_scores[i, threshold_idx]

        filtered_logits[i, typical_score[i] > threshold] = filter_value

    return filtered_logits
