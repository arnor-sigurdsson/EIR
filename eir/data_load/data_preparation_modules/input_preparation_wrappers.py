from functools import partial, update_wrapper
from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL.Image import Image

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
    sequence_load_wrapper,
)
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_array import ComputedArrayInputInfo
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.schemas import (
    ByteInputDataConfig,
    ImageInputDataConfig,
    OmicsInputDataConfig,
    SequenceInputDataConfig,
    al_input_type_info,
)


def prepare_inputs_disk(
    inputs: Dict[str, Any], inputs_objects: "al_input_objects_as_dict", test_mode: bool
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}

    for input_name, data_pointer in inputs.items():
        input_object = inputs_objects[input_name]
        input_source = input_object.input_config.input_info.input_source
        input_type_info = input_object.input_config.input_type_info
        deeplake_inner_key = input_object.input_config.input_info.input_inner_key

        drop_rate = _get_modality_drop_rate(input_type_info=input_type_info)
        should_skip = _should_skip_modality(
            modality_dropout_rate=drop_rate,
            test_mode=test_mode,
        )

        if should_skip:
            continue

        match input_object:
            case ComputedOmicsInputInfo():
                array_raw = omics_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    deeplake_inner_key=deeplake_inner_key,
                    subset_indices=input_object.subset_indices,
                )

                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, OmicsInputDataConfig)
                input_prepared = prepare_one_hot_omics_data(
                    genotype_array=array_raw,
                    na_augment_alpha=input_type_info.na_augment_alpha,
                    na_augment_beta=input_type_info.na_augment_beta,
                    shuffle_augment_alpha=input_type_info.shuffle_augment_alpha,
                    shuffle_augment_beta=input_type_info.shuffle_augment_beta,
                    test_mode=test_mode,
                )

            case ComputedSequenceInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                sequence_tokenized = sequence_load_wrapper(
                    data_pointer=data_pointer,
                    input_source=input_source,
                    deeplake_inner_key=deeplake_inner_key,
                    split_on=input_type_info.split_on,
                    encode_func=input_object.encode_func,
                )
                input_prepared = prepare_sequence_data(
                    sequence_input_object=input_object,
                    cur_file_content_tokenized=sequence_tokenized,
                    test_mode=test_mode,
                )

            case ComputedBytesInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ByteInputDataConfig)
                bytes_data = bytes_load_wrapper(
                    data_pointer=data_pointer,
                    dtype=input_type_info.byte_encoding,
                    input_source=input_source,
                    deeplake_inner_key=deeplake_inner_key,
                )
                input_prepared = prepare_bytes_data(
                    bytes_input_object=input_object,
                    bytes_data=bytes_data,
                    test_mode=test_mode,
                )

            case ComputedImageInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ImageInputDataConfig)
                image_data = image_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    image_mode=input_type_info.mode,
                    deeplake_inner_key=deeplake_inner_key,
                )
                input_prepared = prepare_image_data(
                    image_input_object=input_object,
                    image_data=image_data,
                    test_mode=test_mode,
                )

            case ComputedArrayInputInfo():
                array_data = array_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    deeplake_inner_key=deeplake_inner_key,
                )
                input_prepared = prepare_array_data(
                    array_data=array_data,
                    normalization_stats=input_object.normalization_stats,
                )

            case _:
                input_prepared = inputs[input_name]

        prepared_inputs[input_name] = input_prepared

    return prepared_inputs


def _get_modality_drop_rate(input_type_info: al_input_type_info) -> float:
    drop_rate = getattr(input_type_info, "modality_dropout_rate", 0.0)
    return drop_rate


def _should_skip_modality(modality_dropout_rate: float, test_mode: bool) -> bool:
    if test_mode or modality_dropout_rate == 0.0:
        return False

    should_skip = torch.rand(1) < modality_dropout_rate
    return bool(should_skip)


def prepare_inputs_memory(
    inputs: Dict[str, Any],
    inputs_objects: "al_input_objects_as_dict",
    test_mode: bool,
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}

    for name, data in inputs.items():
        input_object = inputs_objects[name]
        input_type_info = input_object.input_config.input_type_info

        drop_rate = _get_modality_drop_rate(input_type_info=input_type_info)
        should_skip = _should_skip_modality(
            modality_dropout_rate=drop_rate,
            test_mode=test_mode,
        )

        if should_skip:
            continue

        match input_object:
            case ComputedOmicsInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, OmicsInputDataConfig)
                input_prepared = prepare_one_hot_omics_data(
                    genotype_array=data,
                    na_augment_alpha=input_type_info.na_augment_alpha,
                    na_augment_beta=input_type_info.na_augment_beta,
                    shuffle_augment_alpha=input_type_info.shuffle_augment_alpha,
                    shuffle_augment_beta=input_type_info.shuffle_augment_beta,
                    test_mode=test_mode,
                )

            case ComputedSequenceInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                input_prepared = prepare_sequence_data(
                    sequence_input_object=input_object,
                    cur_file_content_tokenized=data,
                    test_mode=test_mode,
                )

            case ComputedBytesInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ByteInputDataConfig)
                input_prepared = prepare_bytes_data(
                    bytes_input_object=input_object,
                    bytes_data=data,
                    test_mode=test_mode,
                )

            case ComputedImageInputInfo():
                input_prepared = prepare_image_data(
                    image_input_object=input_object,
                    image_data=data,
                    test_mode=test_mode,
                )

            case ComputedArrayInputInfo():
                input_prepared = prepare_array_data(
                    array_data=data,
                    normalization_stats=input_object.normalization_stats,
                )

            case _:
                input_prepared = inputs[name]

        prepared_inputs[name] = input_prepared

    return prepared_inputs


def typed_partial_for_hook(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., np.ndarray | Image]:
    """
    Just to make mypy happy.
    """
    partial_func = partial(func, *args, **kwargs)
    return update_wrapper(partial_func, func)


def get_input_data_loading_hooks(
    inputs: al_input_objects_as_dict,
) -> dict[str, Callable[..., np.ndarray | Image]]:
    mapping = {}

    for input_name, input_object in inputs.items():
        common_kwargs = {
            "input_source": input_object.input_config.input_info.input_source,
            "deeplake_inner_key": input_object.input_config.input_info.input_inner_key,
        }

        match input_object:
            case ComputedOmicsInputInfo():
                mapping[input_name] = typed_partial_for_hook(
                    omics_load_wrapper,
                    **common_kwargs,
                    subset_indices=input_object.subset_indices,
                )

            case ComputedImageInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ImageInputDataConfig)
                mapping[input_name] = typed_partial_for_hook(
                    image_load_wrapper,
                    **common_kwargs,
                    image_mode=input_type_info.mode,
                )

            case ComputedSequenceInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                mapping[input_name] = typed_partial_for_hook(
                    sequence_load_wrapper,
                    **common_kwargs,
                    split_on=input_type_info.split_on,
                    encode_func=input_object.encode_func,
                )

            case ComputedBytesInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ByteInputDataConfig)
                mapping[input_name] = typed_partial_for_hook(
                    bytes_load_wrapper,
                    **common_kwargs,
                    dtype=input_type_info.byte_encoding,
                )

            case ComputedArrayInputInfo():
                mapping[input_name] = typed_partial_for_hook(
                    array_load_wrapper,
                    **common_kwargs,
                )

    return mapping
