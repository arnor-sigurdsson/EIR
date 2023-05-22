from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Sequence, Callable, Generator, Dict, Any, TYPE_CHECKING

import torch
import torch.nn.functional as F
from aislib.misc_utils import get_logger, ensure_path_exists
from torch.nn.functional import pad
from torchtext.vocab import Vocab
from transformers.tokenization_utils import PreTrainedTokenizerBase

from eir.data_load import datasets
from eir.data_load.data_preparation_modules.preparation_wrappers import (
    prepare_sequence_output_manual_sample_data,
)
from eir.data_load.data_utils import Batch
from eir.data_load.datasets import (
    impute_missing_modalities_wrapper,
    prepare_inputs_memory,
)
from eir.interpretation.interpret_image import un_normalize
from eir.interpretation.interpret_sequence import extract_raw_inputs_from_tokens
from eir.models.model_training_utils import predict_on_batch
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
)
from eir.setup.schemas import OutputConfig
from eir.train_utils import utils
from eir.train_utils.utils import call_hooks_stage_iterable

if TYPE_CHECKING:
    from eir.train_utils.step_logic import Hooks
    from eir.train import al_input_objects_as_dict, Experiment

logger = get_logger(name=__name__)


@dataclass
class SequenceOutputEvalSamples:
    auto_samples: Dict[str, Sequence["SequenceOutputEvalSample"]]
    manual_samples: Dict[str, Sequence["SequenceOutputEvalSample"]]


@dataclass()
class SequenceOutputEvalSample:
    raw_inputs: Dict[str, Any]
    prepared_inputs: Dict[str, Any]

    sample_id: str


def sequence_out_manual_sample_evaluation_wrapper(
    experiment: "Experiment", iteration: int
):
    inputs = experiment.inputs
    default_eir_hooks = experiment.hooks

    output_configs = experiment.configs.output_configs

    if not any(i.sampling_config for i in output_configs):
        return

    manual_samples = get_sequence_output_manual_input_samples(
        output_configs=output_configs, input_objects=inputs
    )
    auto_samples = get_sequence_output_auto_eval_samples(
        output_configs=output_configs,
        input_objects=inputs,
        eval_dataset=experiment.valid_dataset,
    )
    eval_samples = SequenceOutputEvalSamples(
        auto_samples=auto_samples, manual_samples=manual_samples
    )

    for config in output_configs:
        cur_input_name = config.output_info.output_name
        cur_output_name = config.output_info.output_name

        not_in_manual_samples = cur_output_name not in eval_samples.manual_samples
        not_in_auto_samples = cur_output_name not in eval_samples.auto_samples
        if not_in_manual_samples and not_in_auto_samples:
            continue

        cur_sample_output_folder = utils.prep_sample_outfolder(
            output_folder=experiment.configs.global_config.output_folder,
            output_name=cur_output_name,
            column_name=cur_input_name,
            iteration=iteration,
        )

        if config.output_type_info.sequence_operation == "autoregressive":
            sample_generator = _get_eval_sample_generator(
                eval_samples=eval_samples, output_name=cur_output_name
            )

            for idx, (eval_type, eval_sample) in enumerate(sample_generator):
                generated_sample, generated_tokens = autoregressive_sequence_generation(
                    eval_sample=eval_sample,
                    seq_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    sampling_config=config.sampling_config,
                )

                tokenizer = experiment.inputs[cur_input_name].tokenizer
                if isinstance(tokenizer, PreTrainedTokenizerBase):
                    generated_sample = tokenizer.decode(token_ids=generated_tokens)

                cur_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_generated.txt"
                )
                ensure_path_exists(path=cur_output_path)

                cur_output_path.write_text(data=generated_sample)

                cur_inputs_output_path = (
                    cur_sample_output_folder / eval_type / f"{idx}_inputs"
                )
                _serialize_raw_inputs(
                    raw_inputs=eval_sample.raw_inputs,
                    input_objects=experiment.inputs,
                    output_path=cur_inputs_output_path,
                )


def get_sequence_output_manual_input_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, Sequence[SequenceOutputEvalSample]]:
    prepared_samples = {}

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config:
            continue

        sample_data_from_yaml = config.sampling_config.manual_inputs
        output_name = config.output_info.output_name

        prepared_samples[output_name] = []

        for idx, single_sample_inputs in enumerate(sample_data_from_yaml):
            input_to_model = prepare_sequence_output_manual_sample_data(
                sample_inputs=single_sample_inputs, input_objects=input_objects
            )

            cur_raw_inputs = convert_prepared_input_to_raw(
                inputs=input_to_model, input_objects=input_objects
            )
            cur_raw_inputs_post_processed = _post_prepare_manual_inputs(
                raw_inputs=cur_raw_inputs,
                output_name=output_name,
                input_objects=input_objects,
            )

            cur_eval_sample = SequenceOutputEvalSample(
                raw_inputs=cur_raw_inputs_post_processed,
                prepared_inputs=input_to_model,
                sample_id=f"manual_{idx}",
            )

            prepared_samples[output_name].append(cur_eval_sample)

    return prepared_samples


def _post_prepare_manual_inputs(
    raw_inputs: Dict[str, Any],
    output_name: str,
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, Any]:
    raw_inputs_masked = {}

    for input_name, raw_input in raw_inputs.items():
        input_object = input_objects[input_name]
        input_info = input_object.input_config.input_info

        if input_name == output_name:
            if input_info.input_type == "sequence":
                raw_inputs_masked[output_name] = [i for i in raw_input if i != "<pad>"]
            else:
                raise NotImplementedError()

        else:
            raw_inputs_masked[input_name] = raw_input

    return raw_inputs_masked


def get_sequence_output_auto_eval_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
    eval_dataset: datasets.DatasetBase,
) -> Dict[str, Sequence[SequenceOutputEvalSample]]:
    prepared_eval_samples = {}

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config:
            continue

        output_name = config.output_info.output_name

        prepared_eval_samples[output_name] = []

        for i in range(config.sampling_config.n_eval_inputs):
            input_to_model, target_labels, cur_id = eval_dataset[i]

            if config.output_info.output_type == "sequence":
                if isinstance(eval_dataset, datasets.DiskDataset):
                    input_to_model = datasets.prepare_inputs_disk(
                        inputs=input_to_model,
                        inputs_objects=input_objects,
                        test_mode=True,
                    )

                cur_raw_inputs = convert_prepared_input_to_raw(
                    inputs=input_to_model, input_objects=input_objects
                )
                cur_raw_inputs_masked = _mask_targets_for_auto_eval_generation(
                    raw_inputs=cur_raw_inputs,
                    output_name=output_name,
                    input_objects=input_objects,
                )

                cur_eval_sample = SequenceOutputEvalSample(
                    raw_inputs=cur_raw_inputs_masked,
                    prepared_inputs=input_to_model,
                    sample_id=cur_id,
                )
            elif config.output_info.output_type in ("omics", "image"):
                cur_eval_sample = SequenceOutputEvalSample(
                    raw_inputs=input_to_model,
                    prepared_inputs=input_to_model,
                    sample_id=cur_id,
                )

            else:
                raise NotImplementedError()

            prepared_eval_samples[output_name].append(cur_eval_sample)

    return prepared_eval_samples


def convert_prepared_input_to_raw(
    inputs: Dict[str, Any], input_objects: "al_input_objects_as_dict"
) -> Dict[str, Any]:
    raw_inputs = {}
    for name, data in inputs.items():
        input_object = input_objects[name]

        input_type = input_object.input_config.input_info.input_type

        if input_type == "omics":
            raise NotImplementedError()

        elif input_type == "sequence":
            raw_input = extract_raw_inputs_from_tokens(
                tokens=data, vocab=input_object.vocab
            )
            raw_inputs[name] = raw_input

        elif input_type == "bytes":
            raise NotImplementedError()

        elif input_type == "image":
            raw_input = un_normalize(
                normalized_img=data,
                normalization_stats=input_object.normalization_stats,
            )
            raw_inputs[name] = raw_input

        elif input_type == "tabular":
            raise NotImplementedError()

    return raw_inputs


def _mask_targets_for_auto_eval_generation(
    raw_inputs: Dict[str, Any],
    output_name: str,
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, Any]:
    raw_inputs_masked = {}

    for input_name, raw_input in raw_inputs.items():
        input_object = input_objects[input_name]
        input_info = input_object.input_config.input_info

        if input_name == output_name:
            if input_info.input_type == "sequence":
                raw_inputs_masked[output_name] = []
            else:
                raise NotImplementedError()

        else:
            raw_inputs_masked[input_name] = raw_input

    return raw_inputs_masked


def _serialize_raw_inputs(
    raw_inputs: Dict[str, Any],
    input_objects: "al_input_objects_as_dict",
    output_path: Path,
) -> None:
    for input_name, cur_input in raw_inputs.items():
        cur_input_object = input_objects[input_name]
        cur_input_type = cur_input_object.input_config.input_info.input_type
        cur_output_path = output_path / f"{input_name}"
        ensure_path_exists(path=cur_output_path)

        if cur_input_type == "sequence":
            cur_output_path = cur_output_path.with_suffix(".txt")
            cur_split_on = cur_input_object.input_config.input_type_info.split_on
            cur_text = cur_split_on.join(cur_input)
            cur_output_path.write_text(data=cur_text)

        else:
            logger.warning(
                "Not serializing input of type %s. Not yet implemented.", cur_input_type
            )


def _get_eval_sample_generator(
    eval_samples: SequenceOutputEvalSamples, output_name: str
) -> Generator[Tuple[str, SequenceOutputEvalSample], None, None]:
    cur_config_auto_samples = eval_samples.auto_samples[output_name]
    cur_config_manual_samples = eval_samples.manual_samples[output_name]

    for eval_sample in cur_config_auto_samples:
        yield "auto", eval_sample

    for eval_sample in cur_config_manual_samples:
        yield "manual", eval_sample


@torch.no_grad()
def autoregressive_sequence_generation(
    eval_sample: SequenceOutputEvalSample,
    seq_output_name: str,
    experiment: "Experiment",
    default_eir_hooks: "Hooks",
    sampling_config: SequenceOutputSamplingConfig,
) -> Tuple[str, List[int]]:
    output_object = experiment.outputs[seq_output_name]
    input_object = experiment.inputs[seq_output_name]

    assert output_object.vocab == input_object.vocab

    st = get_special_tokens(
        tokenizer=output_object.tokenizer, vocab=output_object.vocab
    )
    output_config = experiment.outputs[seq_output_name].output_config
    split_on_string = output_config.output_type_info.split_on

    raw_sample_inputs_copy = deepcopy(eval_sample.raw_inputs)
    if len(raw_sample_inputs_copy[seq_output_name]) == 0:
        raw_sample_inputs_copy[seq_output_name].append(st.bos_token)

    generated_string = split_on_string.join(raw_sample_inputs_copy[seq_output_name])
    generated_token_indices = []
    for i in range(sampling_config.generated_sequence_length):
        current_sequence = raw_sample_inputs_copy[seq_output_name]

        assert len(current_sequence) != 0, "Must start with <bos> seed."

        # ABCDE -> BCDE -> [bos, C, D, E]
        while len(current_sequence) >= output_object.computed_max_length:
            current_sequence = current_sequence[1:]
            current_sequence[0] = st.bos_token
            raw_sample_inputs_copy[seq_output_name] = current_sequence

        batch = general_pre_process(
            raw_inputs=raw_sample_inputs_copy,
            experiment=experiment,
            custom_hooks=default_eir_hooks,
            output_configs=experiment.configs.output_configs,
        )

        outputs = predict_on_batch(model=experiment.model, inputs=batch.inputs)

        cur_logits = outputs[seq_output_name][seq_output_name].squeeze()
        # cur_target_logit_index = len(current_sequence) - 1
        cur_target_logit_index = len(current_sequence)
        cur_position_logits = cur_logits[cur_target_logit_index]

        filtered_logits = top_k_top_p_filtering(
            logits=cur_position_logits,
            top_k=sampling_config.top_k,
            top_p=sampling_config.top_p,
        )
        probabilities = F.softmax(input=filtered_logits, dim=-1)
        next_token_index = torch.multinomial(input=probabilities, num_samples=1).item()

        if next_token_index == st.eos_idx:
            break

        generated_token_indices.append(next_token_index)

        cur_vocab = input_object.vocab
        next_token = cur_vocab.lookup_token(index=next_token_index)

        generated_string = generated_string + next_token + split_on_string
        current_sequence += [next_token]

        raw_sample_inputs_copy[seq_output_name] = current_sequence

    return generated_string.removeprefix(st.bos_token), generated_token_indices


def pad_batch_with_bos(batch_tensor: torch.Tensor, bos_value: int) -> torch.Tensor:
    left_padding = 1
    batch_tensor = pad(input=batch_tensor, pad=[left_padding, 0], value=bos_value)

    return batch_tensor


def general_pre_process(
    raw_inputs: dict,
    experiment: "Experiment",
    custom_hooks: "Hooks",
    output_configs: Sequence[OutputConfig],
) -> Batch:
    """
    The custom hooks here make sure we are doing basic processing,
    i.e. not the sequence output specific things we define here in the train module.
    """
    exp = experiment

    inputs_prepared_for_memory = {}
    for name, cur_input in raw_inputs.items():
        input_object = experiment.inputs[name]

        if input_object.input_config.input_info.input_type == "sequence":
            cur_input = input_object.encode_func(cur_input)

        inputs_prepared_for_memory[name] = cur_input

    inputs_prepared = prepare_inputs_memory(
        inputs=inputs_prepared_for_memory, inputs_objects=exp.inputs, test_mode=True
    )

    inputs_final = impute_missing_modalities_wrapper(
        inputs_values=inputs_prepared, inputs_objects=exp.inputs
    )

    inputs_final = {k: v.unsqueeze(0) for k, v in inputs_final.items()}

    loader_batch = (inputs_final, None, None)

    batch_prep_hook_kwargs = {"experiment": exp}
    state = call_hooks_stage_iterable(
        hook_iterable=custom_hooks.step_func_hooks.base_prepare_batch,
        common_kwargs={"loader_batch": loader_batch, **batch_prep_hook_kwargs},
        state=None,
    )
    batch = state["batch"]

    batch_final = Batch(inputs=batch.inputs, target_labels={}, ids=list())

    return batch_final


def pad_seq_output_samples_with_bos(
    inputs_prepared: dict,
    ssl_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
) -> dict:
    inputs_processed = {}

    for input_name, input_value in inputs_prepared.items():
        if input_name in (i.output_info.output_name for i in ssl_configs):
            input_object = input_objects[input_name]
            st = get_special_tokens(
                tokenizer=input_object.tokenizer, vocab=input_object.vocab
            )

            cur_input = inputs_prepared[input_name]
            shifted_right = cur_input[1:]

            inputs_processed[input_name] = pad_batch_with_bos(
                batch_tensor=shifted_right, bos_value=st.bos_idx
            )

        else:
            inputs_processed[input_name] = input_value

    return inputs_processed


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


def get_special_tokens(tokenizer: Callable, vocab: Vocab) -> SpecialTokens:
    keys_and_defaults = (
        ("mask_token", "<mask>"),
        ("pad_token", "<pad>"),
        ("bos_token", "<bos>"),
        ("eos_token", "<eos>"),
        ("unk_token", "<unk>"),
    )
    kwargs = {}

    for key, default in keys_and_defaults:
        cur_token = getattr(tokenizer, key, default)
        kwargs[key] = cur_token

        cur_index = vocab[cur_token]
        idx_kwarg_key = key.replace("_token", "_idx")
        kwargs[idx_kwarg_key] = cur_index

    special_tokens = SpecialTokens(**kwargs)

    return special_tokens
