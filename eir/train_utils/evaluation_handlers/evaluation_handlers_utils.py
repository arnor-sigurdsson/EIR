import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import torch
import torchtext
from aislib.misc_utils import ensure_path_exists, get_logger
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import Vocab

from eir.data_load.data_preparation_modules.prepare_array import (
    array_load_wrapper,
    prepare_array_data,
    un_normalize_array,
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
from eir.data_load.datasets import al_getitem_return
from eir.data_load.label_setup import (
    al_label_transformers,
    streamline_values_for_transformers,
)
from eir.interpretation.interpret_image import un_normalize_image
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup.input_setup_modules.setup_array import (
    ArrayNormalizationStats,
    ComputedArrayInputInfo,
)
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    ImageNormalizationStats,
)
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    TokenizerProtocolPreSplit,
    TokenizerProtocolRaw,
    get_sequence_split_function,
)
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.schemas import (
    ArrayInputDataConfig,
    ByteInputDataConfig,
    ImageInputDataConfig,
    OmicsInputDataConfig,
    SequenceInputDataConfig,
    TabularInputDataConfig,
)
from eir.train_utils.utils import call_hooks_stage_iterable

if TYPE_CHECKING:
    from eir.predict import PredictExperiment, PredictHooks
    from eir.serve import ServeExperiment
    from eir.setup.input_setup import al_input_objects_as_dict
    from eir.train import Experiment
    from eir.train_utils.step_logic import Hooks

logger = get_logger(name=__name__)


def remove_special_tokens_from_string(
    string: str, special_tokens: "SpecialTokens"
) -> str:
    """
    TODO:   Deprecate in favor of `remove_special_tokens`, due to not being guaranteed
            to handle spaces / split_on correctly, only removes the tokens themselves
            and creates extra spaces.
    """
    token_names = ["mask_token", "pad_token", "eos_token", "bos_token", "unk_token"]
    for token in token_names:
        assert hasattr(special_tokens, token)
        token_value = getattr(special_tokens, token)
        string = string.replace(token_value, "")
    return string


def remove_special_tokens(
    tokens: list[int], special_tokens: "SpecialTokens"
) -> list[int]:
    token_names = ["mask_idx", "pad_idx", "eos_idx", "bos_idx", "unk_idx"]
    for token in token_names:
        assert hasattr(special_tokens, token)
        token_value = getattr(special_tokens, token)
        tokens = [i for i in tokens if i != token_value]
    return tokens


def convert_model_inputs_to_raw(
    inputs_to_model: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    input_objects: "al_input_objects_as_dict",
) -> dict[str, str | np.ndarray | dict[str, np.ndarray] | Image.Image]:
    raw_inputs = {}
    for input_name, data in inputs_to_model.items():
        input_object = input_objects[input_name]

        raw_input: str | np.ndarray | dict[str, np.ndarray] | Image.Image
        match input_object:
            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                assert isinstance(data, dict)
                raw_input = convert_tabular_input_to_raw(
                    data=data,
                    input_transformers=input_object.labels.label_transformers,
                )

            case ComputedOmicsInputInfo() | ComputedBytesInputInfo():
                assert isinstance(data, torch.Tensor)
                raw_input = data.numpy().squeeze()

            case ComputedSequenceInputInfo():
                assert isinstance(data, torch.Tensor)
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                assert input_object.tokenizer is not None
                raw_input = decode_tokens(
                    tokens=data.numpy().tolist(),
                    vocab=input_object.vocab,
                    split_on=input_type_info.split_on,
                )
                special_tokens = get_special_tokens(
                    tokenizer=input_object.tokenizer, vocab=input_object.vocab
                )
                raw_input = remove_special_tokens_from_string(
                    string=raw_input,
                    special_tokens=special_tokens,
                )

            case ComputedImageInputInfo():
                assert isinstance(data, torch.Tensor)
                raw_input = convert_image_input_to_raw(
                    data=data, normalization_stats=input_object.normalization_stats
                )

            case ComputedArrayInputInfo():
                assert isinstance(data, torch.Tensor)
                raw_input = convert_array_input_to_raw(
                    data=data, normalization_stats=input_object.normalization_stats
                )

            case _:
                raise NotImplementedError()

        raw_inputs[input_name] = raw_input

    return raw_inputs


def convert_image_input_to_raw(
    data: torch.Tensor, normalization_stats: ImageNormalizationStats
) -> Image.Image:
    data_np: np.ndarray = data.numpy()
    assert data_np.ndim == 3, "Input should be 3D"

    cur_input = un_normalize_image(
        normalized_img=data_np,
        normalization_stats=normalization_stats,
    )

    cur_input_hwc = np.moveaxis(cur_input, 0, -1)
    raw_input_uint = (cur_input_hwc * 255).astype(np.uint8)

    n_channels = raw_input_uint.shape[-1]
    mode: Optional[str]
    match n_channels:
        case 1:
            mode = "L"
            raw_input_uint = raw_input_uint.squeeze(axis=-1)
        case 3:
            mode = "RGB"
        case 4:
            mode = "RGBA"
        case _:
            mode = None

    return Image.fromarray(raw_input_uint, mode=mode)


def convert_tabular_input_to_raw(
    data: Dict[str, torch.Tensor], input_transformers: al_label_transformers
) -> Dict[str, np.ndarray]:
    all_reversed: Dict[str, np.ndarray] = {}
    for col_name, tensor_data in data.items():
        transformer = input_transformers[col_name]
        tensor_data_reshaped = tensor_data.numpy().reshape(-1, 1)
        assert tensor_data_reshaped.shape[0] > 0, "Empty tensor"

        it = transformer.inverse_transform
        if isinstance(transformer, StandardScaler):
            cur_reversed = it(tensor_data_reshaped)
        elif isinstance(transformer, LabelEncoder):
            cur_reversed = it(tensor_data_reshaped.ravel())
        else:
            raise NotImplementedError()

        cur_reversed = cur_reversed.squeeze()
        assert cur_reversed.ndim == 0

        cur_reversed = cur_reversed.item()
        all_reversed[col_name] = cur_reversed
    return all_reversed


def convert_array_input_to_raw(
    data: torch.Tensor, normalization_stats: Optional[ArrayNormalizationStats]
) -> np.ndarray:
    data_un_normalized = un_normalize_array(
        array=data, normalization_stats=normalization_stats
    )

    data_un_normalized_numpy = data_un_normalized.numpy()

    return data_un_normalized_numpy


def general_pre_process_prepared_inputs(
    prepared_inputs: dict[str, Any],
    target_labels: dict[str, Any],
    sample_ids: list[str],
    experiment: Union["Experiment", "PredictExperiment", "ServeExperiment"],
    custom_hooks: Union["Hooks", "PredictHooks"],
) -> Batch:
    """
    The custom hooks here make sure we are doing basic processing,
    i.e. not the sequence output specific things we define here in the train module.
    """

    loader_batch = (prepared_inputs, target_labels, sample_ids)

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


def get_special_tokens(
    tokenizer: TokenizerProtocolRaw | TokenizerProtocolPreSplit, vocab: Vocab
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

    for key, default in keys_and_defaults:
        cur_token = getattr(tokenizer, key, default)
        token_values[key] = cur_token

        cur_index = vocab[cur_token]
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


def decode_tokens(
    tokens: list[int],
    vocab: Vocab,
    split_on: str | None,
) -> str:
    tokens_decoded = vocab.lookup_tokens(indices=tokens)
    if split_on is not None:
        generated_sample = split_on.join(tokens_decoded)
    else:
        generated_sample = "".join(tokens_decoded)

    return generated_sample


def _streamline_tabular_data_for_transformers(
    tabular_input: dict[str, np.ndarray], transformers: al_label_transformers
) -> dict[str, torch.Tensor]:
    parsed_output = {}
    for name, value in tabular_input.items():
        cur_transformer = transformers[name]
        value_np = np.array([value])
        value_streamlined = streamline_values_for_transformers(
            transformer=cur_transformer, values=value_np
        )
        value_transformed = cur_transformer.transform(value_streamlined)
        value_tensor = torch.from_numpy(value_transformed)
        parsed_output[name] = value_tensor.squeeze(0)
    return parsed_output


def post_prepare_manual_inputs(
    prepared_inputs: Dict[str, Any],
    output_name: str,
    input_objects: "al_input_objects_as_dict",
) -> Dict[str, torch.Tensor]:
    prepared_inputs_post = {}

    for input_name, prepared_input in prepared_inputs.items():
        input_object = input_objects[input_name]
        input_info = input_object.input_config.input_info

        input_type = input_info.input_type
        if input_type != "sequence":
            prepared_inputs_post[input_name] = prepared_input
            continue

        assert isinstance(input_object, ComputedSequenceInputInfo)
        assert input_object.tokenizer is not None

        specials = get_special_tokens(
            tokenizer=input_object.tokenizer, vocab=input_object.vocab
        )
        pad_idx = specials.pad_idx

        if input_name == output_name:
            if input_info.input_type == "sequence":
                prepared_input_no_pad = [i for i in prepared_input if i != pad_idx]
                prepared_inputs_post[output_name] = torch.tensor(
                    prepared_input_no_pad,
                    dtype=torch.long,
                )
            else:
                raise NotImplementedError()

        else:
            prepared_inputs_post[input_name] = prepared_input

    return prepared_inputs_post


def _recursive_unsqueeze(
    data: dict,
    dim: int,
) -> Any:
    if isinstance(data, dict):
        return {k: _recursive_unsqueeze(v, dim=dim) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(dim=dim)
    else:
        raise NotImplementedError()


def prepare_manual_sample_data(
    sample_inputs: Dict[str, Any], input_objects: "al_input_objects_as_dict"
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    prepared_inputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
    for name, data in sample_inputs.items():
        input_object = input_objects[name]
        input_info = input_object.input_config.input_info
        input_type_info = input_object.input_config.input_type_info

        match input_object:
            case ComputedOmicsInputInfo():
                assert isinstance(input_type_info, OmicsInputDataConfig)
                array_raw = omics_load_wrapper(
                    data_pointer=data,
                    input_source=input_info.input_source,
                    subset_indices=input_object.subset_indices,
                    deeplake_inner_key=input_info.input_inner_key,
                )

                array_prepared = prepare_one_hot_omics_data(
                    genotype_array=array_raw,
                    na_augment_alpha=input_type_info.na_augment_alpha,
                    na_augment_beta=input_type_info.na_augment_beta,
                    shuffle_augment_alpha=input_type_info.shuffle_augment_alpha,
                    shuffle_augment_beta=input_type_info.shuffle_augment_beta,
                    test_mode=True,
                )
                prepared_inputs[name] = array_prepared

            case ComputedSequenceInputInfo():
                assert isinstance(input_type_info, SequenceInputDataConfig)

                sequence_split = streamline_sequence_manual_data(
                    data=data,
                    split_on=input_type_info.split_on,
                )

                sequence_tokenized = input_object.encode_func(sequence_split)
                sequence_array = np.array(sequence_tokenized)

                prepared_sequence = prepare_sequence_data(
                    sequence_input_object=input_object,
                    cur_file_content_tokenized=sequence_array,
                    test_mode=True,
                )

                prepared_inputs[name] = prepared_sequence

            case ComputedBytesInputInfo():
                assert isinstance(input_type_info, ByteInputDataConfig)
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

            case ComputedImageInputInfo():
                assert isinstance(input_type_info, ImageInputDataConfig)
                image_data = image_load_wrapper(
                    data_pointer=data,
                    image_mode=input_type_info.mode,
                    input_source=input_info.input_source,
                    deeplake_inner_key=input_info.input_inner_key,
                )
                prepared_image_data = prepare_image_data(
                    image_input_object=input_object,
                    image_data=image_data,
                    test_mode=True,
                )
                prepared_inputs[name] = prepared_image_data

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                assert isinstance(input_type_info, TabularInputDataConfig)
                transformers = input_object.labels.label_transformers
                tabular_data = _streamline_tabular_data_for_transformers(
                    tabular_input=data, transformers=transformers
                )
                prepared_inputs[name] = tabular_data

            case ComputedArrayInputInfo():
                assert isinstance(input_type_info, ArrayInputDataConfig)
                array_data = array_load_wrapper(
                    input_source=input_info.input_source,
                    data_pointer=data,
                    deeplake_inner_key=input_info.input_inner_key,
                )
                prepared_array_data = prepare_array_data(
                    array_data=array_data,
                    normalization_stats=input_object.normalization_stats,
                )
                prepared_inputs[name] = prepared_array_data

            case _:
                raise ValueError(
                    f"Unknown input type '{input_object.__class__.__name__}'"
                )

    return prepared_inputs


def serialize_raw_inputs(
    raw_inputs: dict[str, str | np.ndarray | dict[str, np.ndarray] | Image.Image],
    input_objects: "al_input_objects_as_dict",
    output_path: Path,
) -> None:
    warned_types: set[str] = set()
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


def get_dataset_loader_single_sample_generator(
    dataset: Dataset,
    infinite: bool = True,
) -> Iterator[al_getitem_return]:
    loader: Iterator[al_getitem_return]
    if infinite:
        loader = cycle(DataLoader(dataset=dataset, batch_size=1, shuffle=True))
    else:
        loader = iter(DataLoader(dataset=dataset, batch_size=1, shuffle=True))

    for input_to_model, _, cur_ids in loader:
        inputs_squeezed = _recursive_batch_dimension_squeeze(inputs=input_to_model)
        yield inputs_squeezed, {}, cur_ids


def _recursive_batch_dimension_squeeze(
    inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    We need this function as when returning from a dataloader one sample at a time,
    the single samples include a batch dimension. When concatenating these samples
    we will end up with an extra dimension causing downstream issues.
    """

    inputs_squeezed = {}
    for name, data in inputs.items():
        if isinstance(data, dict):
            inputs_squeezed[name] = _recursive_batch_dimension_squeeze(data)
        else:
            assert data.shape[0] == 1
            inputs_squeezed[name] = data.squeeze(dim=0)

    return inputs_squeezed


def extract_input_types(input_objects: "al_input_objects_as_dict") -> dict[str, str]:
    input_types: dict[str, str] = {}
    for input_name, input_object in input_objects.items():
        input_types[input_name] = input_object.input_config.input_info.input_type
    return input_types


def prepare_base_input(
    prepared_inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    prepared_sample_inputs_copy = deepcopy(prepared_inputs)
    return prepared_sample_inputs_copy


def streamline_sequence_manual_data(
    data: str, split_on: Optional[str]
) -> list[str] | str:
    """
    This is to specifically handle the case of an empty string / None being passed
    here. If e.g. we call the split_func on '', we will get [''], which will
    end up being encoded as a <unk> token. Instead, we want to return an empty
    list here. In e.g. the validation handler code, this is also set explicitly.
    """

    sequence_streamlined: list[str] | str
    if data == "" or data is None:
        sequence_streamlined = []
    else:
        split_func = get_sequence_split_function(split_on=split_on)
        split_data = split_func(data)
        sequence_streamlined = split_data

    return sequence_streamlined


def get_batch_generator(
    iterator: Iterator[Tuple[int, Tuple[Any, Any]]],
    batch_size: int,
) -> Iterator[list[Tuple[int, Tuple[Any, Any]]]]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
