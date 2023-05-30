import json
from copy import deepcopy, copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Tuple,
    Iterator,
    Sequence,
    Callable,
    Union,
    Generator,
    Dict,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from aislib.misc_utils import get_logger, ensure_path_exists
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from transformers.tokenization_utils import PreTrainedTokenizerBase

from eir.data_load.data_preparation_modules.prepare_array import (
    array_load_wrapper,
    prepare_array_data,
)
from eir.data_load.data_preparation_modules.prepare_bytes import (
    bytes_load_wrapper,
    prepare_bytes_data,
)
from eir.data_load.data_preparation_modules.prepare_image import (
    image_load_wrapper,
    prepare_image_data,
)
from eir.data_load.data_preparation_modules.prepare_omics import (
    omics_load_wrapper,
    prepare_one_hot_omics_data,
)
from eir.data_load.data_preparation_modules.prepare_sequence import (
    prepare_sequence_data,
)
from eir.data_load.data_utils import Batch
from eir.data_load.datasets import (
    impute_missing_modalities_wrapper,
    prepare_inputs_memory,
    al_getitem_return,
)
from eir.data_load.label_setup import (
    al_label_transformers,
    _streamline_values_for_transformers,
)
from eir.interpretation.interpret_image import un_normalize
from eir.models.model_training_utils import predict_on_batch
from eir.setup.input_setup_modules.setup_image import ImageNormalizationStats
from eir.setup.input_setup_modules.setup_sequence import get_sequence_split_function
from eir.setup.schema_modules.output_schemas_sequence import (
    SequenceOutputSamplingConfig,
)
from eir.setup.schemas import OutputConfig
from eir.train_utils import utils
from eir.train_utils.utils import call_hooks_stage_iterable

if TYPE_CHECKING:
    from eir.train_utils.step_logic import Hooks
    from eir.train import al_input_objects_as_dict, Experiment
    from eir.predict import PredictExperiment

logger = get_logger(name=__name__)


@dataclass
class SequenceOutputEvalSamples:
    auto_samples: Dict[str, Sequence["SequenceOutputEvalSample"]]
    manual_samples: Dict[str, Sequence["SequenceOutputEvalSample"]]


@dataclass()
class SequenceOutputEvalSample:
    inputs_to_model: Dict[str, Any]
    target_labels: Dict[str, Any]
    sample_id: str


def sequence_out_single_sample_evaluation_wrapper(
    experiment: Union["Experiment", "PredictExperiment"],
    input_objects: "al_input_objects_as_dict",
    auto_dataset_to_load_from: Dataset,
    iteration: int,
    output_folder: str,
) -> None:
    default_eir_hooks = experiment.hooks

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

        if config.output_type_info.sequence_operation == "autoregressive":
            sample_generator = _get_eval_sample_generator(
                eval_samples=eval_samples, output_name=cur_output_name
            )

            for idx, (eval_type, eval_sample) in enumerate(sample_generator):
                generated_tokens = autoregressive_sequence_generation(
                    input_objects=input_objects,
                    eval_sample=eval_sample,
                    seq_output_name=cur_output_name,
                    experiment=experiment,
                    default_eir_hooks=default_eir_hooks,
                    sampling_config=config.sampling_config,
                )

                generated_sample = decode_tokens(
                    tokens=generated_tokens,
                    tokenizer=input_objects[cur_input_name].tokenizer,
                    vocab=input_objects[cur_input_name].vocab,
                    split_on=config.output_type_info.split_on,
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

                _serialize_raw_inputs(
                    raw_inputs=raw_inputs,
                    input_objects=input_objects,
                    output_path=cur_inputs_output_path,
                )


def convert_model_inputs_to_raw(
    inputs_to_model: Dict[str, torch.Tensor | dict[str, torch.Tensor]],
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, str | np.ndarray]:
    raw_inputs = {}
    for name, data in inputs_to_model.items():
        input_object = input_objects[name]

        input_type = input_object.input_config.input_info.input_type

        match input_type:
            case "tabular":
                raw_input = convert_tabular_input_to_raw(
                    data=data,
                    input_transformers=input_object.labels.label_transformers,
                )
            case "omics" | "array" | "bytes":
                raw_input = data.numpy().squeeze()
            case "sequence":
                raw_input = decode_tokens(
                    tokens=data.numpy().squeeze().tolist(),
                    vocab=input_object.vocab,
                    split_on=input_object.input_config.input_type_info.split_on,
                    tokenizer=input_object.tokenizer,
                )
            case "image":
                raw_input = convert_image_input_to_raw(
                    data=data, normalization_stats=input_object.normalization_stats
                )
            case _:
                raise NotImplementedError()

        raw_inputs[name] = raw_input

    return raw_inputs


def convert_image_input_to_raw(
    data: torch.Tensor, normalization_stats: ImageNormalizationStats
) -> Image:
    data = data.numpy()
    assert data.ndim == 4, "Input should be 4D"
    assert data.shape[0] == 1, "The batch dimension should be 1"

    cur_input = un_normalize(
        normalized_img=data,
        normalization_stats=normalization_stats,
    )

    cur_input = cur_input.squeeze(axis=0)
    cur_input_hwc = np.moveaxis(cur_input, 0, -1)
    raw_input_uint = (cur_input_hwc * 255).astype(np.uint8)
    return Image.fromarray(raw_input_uint)


def convert_tabular_input_to_raw(
    data: Dict[str, torch.Tensor], input_transformers: al_label_transformers
) -> Dict[str, np.ndarray]:
    all_reversed = {}
    for col_name, tensor_data in data.items():
        transformer = input_transformers[col_name]
        tensor_data_reshaped = tensor_data.numpy().reshape(-1, 1)
        assert tensor_data_reshaped.shape[0] > 0, "Empty tensor"

        cur_reversed = transformer.inverse_transform(tensor_data_reshaped)

        cur_reversed = cur_reversed.squeeze()
        assert cur_reversed.ndim == 0

        cur_reversed = cur_reversed.item()
        all_reversed[col_name] = cur_reversed
    return all_reversed


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
            prepared_inputs = prepare_sequence_output_manual_sample_data(
                sample_inputs=single_sample_inputs, input_objects=input_objects
            )

            final_inputs = _post_prepare_manual_inputs(
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


def get_dataset_loader_single_sample_generator(
    dataset: Dataset,
) -> Iterator[al_getitem_return]:
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    yield from loader


def get_sequence_output_auto_validation_samples(
    output_configs: Sequence[OutputConfig],
    input_objects: "al_input_objects_as_dict",
    eval_sample_iterator: Iterator[al_getitem_return],
) -> Dict[str, Sequence[SequenceOutputEvalSample]]:
    prepared_eval_samples = {}
    input_types = _extract_input_types(input_objects=input_objects)

    for config_idx, config in enumerate(output_configs):
        if not config.sampling_config or config.output_info.output_type != "sequence":
            continue

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


def _extract_input_types(input_objects: "al_input_objects_as_dict") -> dict[str, str]:
    input_types = {}
    for input_name, input_object in input_objects.items():
        input_types[input_name] = input_object.input_config.input_info.input_type
    return input_types


def _serialize_raw_inputs(
    raw_inputs: Dict[str, Union[np.ndarray, str, Image]],
    input_objects: "al_input_objects_as_dict",
    output_path: Path,
) -> None:
    warned_types = set()
    for input_name, cur_input in raw_inputs.items():
        cur_input_object = input_objects[input_name]
        cur_input_type = cur_input_object.input_config.input_info.input_type
        cur_output_path = output_path / f"{input_name}"
        ensure_path_exists(path=cur_output_path)

        match cur_input_type:
            case "omics" | "array":
                assert isinstance(cur_input, np.ndarray)
                cur_output_path = cur_output_path.with_suffix(".npy")
                np.save(str(cur_output_path), cur_input)
            case "tabular":
                assert isinstance(cur_input, dict)
                cur_output_path = cur_output_path.with_suffix(".json")
                cur_output_path.write_text(data=json.dumps(cur_input))
            case "sequence":
                assert isinstance(cur_input, str)
                cur_output_path = cur_output_path.with_suffix(".txt")
                cur_output_path.write_text(data=cur_input)
            case "image":
                assert isinstance(cur_input, Image.Image)
                cur_output_path = cur_output_path.with_suffix(".png")
                cur_input.save(cur_output_path)
            case "bytes":
                assert isinstance(cur_input, np.ndarray)
                cur_output_path = cur_output_path.with_suffix(".bin")
                cur_input.tofile(cur_output_path)
            case _:
                if cur_input_type not in warned_types:
                    warned_types.add(cur_input_type)
                    logger.warning(
                        "Not serializing input of type %s. Not yet implemented.",
                        cur_input_type,
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
    input_objects: "al_input_objects_as_dict",
    eval_sample: SequenceOutputEvalSample,
    seq_output_name: str,
    experiment: "Experiment",
    default_eir_hooks: "Hooks",
    sampling_config: SequenceOutputSamplingConfig,
) -> list[int]:
    output_object = experiment.outputs[seq_output_name]
    input_object = input_objects[seq_output_name]

    _check_vocab_consistency(
        output_object_vocab=output_object.vocab, input_object_vocab=input_object.vocab
    )

    st = get_special_tokens(
        tokenizer=output_object.tokenizer, vocab=output_object.vocab
    )

    prepared_sample_inputs = _prepare_base_input(
        prepared_inputs=eval_sample.inputs_to_model,
    )

    generated_tokens = _extract_base_generated_tokens(
        prepared_inputs=prepared_sample_inputs, seq_output_name=seq_output_name
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
            current_generated_length=i, max_length=output_object.computed_max_length
        )

        batch = general_pre_process_prepared_inputs(
            prepared_inputs=prepared_sample_inputs,
            target_labels=eval_sample.target_labels,
            sample_id=eval_sample.sample_id,
            experiment=experiment,
            custom_hooks=default_eir_hooks,
        )

        outputs = predict_on_batch(model=experiment.model, inputs=batch.inputs)

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


def _prepare_base_input(
    prepared_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    prepared_sample_inputs_copy = deepcopy(prepared_inputs)
    return prepared_sample_inputs_copy


def _extract_base_generated_tokens(
    prepared_inputs: Dict[str, torch.Tensor], seq_output_name: str
) -> list[int]:
    tensor_base = prepared_inputs[seq_output_name]
    assert tensor_base.dim() == 2, tensor_base.dim()

    list_base = tensor_base.squeeze().tolist()

    return list_base


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
    current_sequence: torch.tensor,
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

    return next_token_index


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


def general_pre_process_raw_inputs(
    raw_inputs: dict[str, Any],
    experiment: "Experiment",
) -> dict[str, torch.Tensor]:
    inputs_prepared_for_memory = {}
    for name, cur_input in raw_inputs.items():
        input_object = experiment.inputs[name]

        if input_object.input_config.input_info.input_type == "sequence":
            cur_input = input_object.encode_func(cur_input)

        inputs_prepared_for_memory[name] = cur_input

    inputs_prepared = prepare_inputs_memory(
        inputs=inputs_prepared_for_memory,
        inputs_objects=experiment.inputs,
        test_mode=True,
    )

    inputs_final = impute_missing_modalities_wrapper(
        inputs_values=inputs_prepared, inputs_objects=experiment.inputs
    )

    return inputs_final


def general_pre_process_prepared_inputs(
    prepared_inputs: dict[str, Any],
    target_labels: dict[str, Any],
    sample_id: str,
    experiment: "Experiment",
    custom_hooks: "Hooks",
) -> Batch:
    """
    The custom hooks here make sure we are doing basic processing,
    i.e. not the sequence output specific things we define here in the train module.
    """

    inputs_final = prepared_inputs

    loader_batch = (inputs_final, target_labels, sample_id)

    batch_prep_hook_kwargs = {"experiment": experiment}
    state = call_hooks_stage_iterable(
        hook_iterable=custom_hooks.step_func_hooks.base_prepare_batch,
        common_kwargs={"loader_batch": loader_batch, **batch_prep_hook_kwargs},
        state=None,
    )
    batch = state["batch"]

    batch_final = Batch(inputs=batch.inputs, target_labels={}, ids=list())

    return batch_final


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


def decode_tokens(
    tokens: list[int], tokenizer: Callable, vocab: Vocab, split_on: str | None
) -> str:
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        generated_sample = tokenizer.decode(token_ids=tokens)
        return generated_sample

    tokens_decoded = vocab.lookup_tokens(indices=tokens)
    if split_on is not None:
        generated_sample = split_on.join(tokens_decoded)
    else:
        generated_sample = "".join(tokens_decoded)

    return generated_sample


def prepare_sequence_output_manual_sample_data(
    sample_inputs: Dict[str, Any], input_objects: "al_input_objects_as_dict"
) -> Dict[str, Any]:
    """
    omics: File (np.array) on disk.
    sequence: Written in yaml file.
    bytes: File (any) on disk.
    image: File (PIL readable) on disk.
    tabular: Written in yaml file.
    array: File (np.array) on disk.
    """

    prepared_inputs = {}
    for name, data in sample_inputs.items():
        input_object = input_objects[name]
        input_info = input_object.input_config.input_info
        input_type = input_info.input_type
        input_type_info = input_object.input_config.input_type_info

        match input_type:
            case "omics":
                array_raw = omics_load_wrapper(
                    data_pointer=data,
                    input_source=input_info.input_source,
                    subset_indices=input_type_info.subset_snps_file,
                    deeplake_inner_key=input_info.input_inner_key,
                )

                array_prepared = prepare_one_hot_omics_data(
                    genotype_array=array_raw,
                    na_augment_perc=input_type_info.na_augment_perc,
                    na_augment_prob=input_type_info.na_augment_prob,
                    test_mode=True,
                )
                prepared_inputs[name] = array_prepared

            case "sequence":
                split_func = get_sequence_split_function(
                    split_on=input_type_info.split_on
                )
                sequence_split = split_func(data)
                sequence_tokenized = input_object.encode_func(sequence_split)
                sequence_array = np.array(sequence_tokenized)

                prepared_sequence = prepare_sequence_data(
                    sequence_input_object=input_object,
                    cur_file_content_tokenized=sequence_array,
                    test_mode=True,
                )

                prepared_inputs[name] = prepared_sequence

            case "bytes":
                bytes_data = bytes_load_wrapper(
                    input_source=input_info.input_source,
                    data_pointer=data,
                    dtype=input_type_info.byte_encoding,
                    deeplake_inner_key=input_info.input_inner_key,
                )
                prepared_bytes_input = prepare_bytes_data(
                    bytes_input_object=input_object,
                    bytes_data=bytes_data,
                    test_mode=True,
                )

                prepared_inputs[name] = prepared_bytes_input

            case "image":
                image_data = image_load_wrapper(
                    data_pointer=data,
                    input_source=input_info.input_source,
                    deeplake_inner_key=input_info.input_inner_key,
                )
                prepared_image_data = prepare_image_data(
                    image_input_object=input_object,
                    image_data=image_data,
                    test_mode=True,
                )
                prepared_inputs[name] = prepared_image_data

            case "tabular":
                transformers = input_object.labels.label_transformers
                tabular_data = _streamline_tabular_data_for_transformers(
                    tabular_input=data, transformers=transformers
                )
                prepared_inputs[name] = tabular_data

            case "array":
                array_data = array_load_wrapper(
                    input_source=input_info.input_source,
                    data_pointer=data,
                    deeplake_inner_key=input_info.input_inner_key,
                )
                prepared_array_data = prepare_array_data(array_data=array_data)
                prepared_inputs[name] = prepared_array_data

            case _:
                raise ValueError(f"Unknown input type '{input_type}'")

    return prepared_inputs


def _streamline_tabular_data_for_transformers(
    tabular_input: dict[str, np.ndarray], transformers: al_label_transformers
) -> dict[str, np.ndarray]:
    parsed_output = {}
    for name, value in tabular_input.items():
        cur_transformer = transformers[name]
        value_np = np.array(value)
        value_streamlined = _streamline_values_for_transformers(
            transformer=cur_transformer, values=value_np
        )
        value_transformed = cur_transformer.transform(value_streamlined)
        parsed_output[name] = value_transformed
    return parsed_output


def _post_prepare_manual_inputs(
    prepared_inputs: Dict[str, Any],
    output_name: str,
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, torch.Tensor]:
    prepared_inputs_masked = {}

    for input_name, prepared_input in prepared_inputs.items():
        input_object = input_objects[input_name]
        input_info = input_object.input_config.input_info

        specials = get_special_tokens(
            tokenizer=input_object.tokenizer, vocab=input_object.vocab
        )
        pad_idx = specials.pad_idx

        if input_name == output_name:
            if input_info.input_type == "sequence":
                prepared_inputs_masked[output_name] = torch.tensor(
                    [i for i in prepared_input if i != pad_idx], dtype=torch.long
                )
            else:
                raise NotImplementedError()

        else:
            prepared_inputs_masked[input_name] = prepared_input

    prepared_inputs_batched = {
        k: v.unsqueeze(0) for k, v in prepared_inputs_masked.items()
    }
    return prepared_inputs_batched
