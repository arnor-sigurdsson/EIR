from functools import partial
from typing import Dict, Any, Mapping, Callable

import torch

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
    sequence_load_wrapper,
    prepare_sequence_data,
)
from eir.setup.input_setup import al_input_objects_as_dict


def prepare_inputs_disk(
    inputs: Dict[str, Any], inputs_objects: "al_input_objects_as_dict", test_mode: bool
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}

    for input_name, data_pointer in inputs.items():
        input_object = inputs_objects[input_name]

        input_source = input_object.input_config.input_info.input_source
        deeplake_inner_key = input_object.input_config.input_info.input_inner_key
        input_type_info = input_object.input_config.input_type_info
        input_type = input_object.input_config.input_info.input_type

        match input_type:
            case "omics":
                array_raw = omics_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    deeplake_inner_key=deeplake_inner_key,
                    subset_indices=input_object.subset_indices,
                )
                array_prepared = prepare_one_hot_omics_data(
                    genotype_array=array_raw,
                    na_augment_perc=input_type_info.na_augment_perc,
                    na_augment_prob=input_type_info.na_augment_prob,
                    test_mode=test_mode,
                )
                prepared_inputs[input_name] = array_prepared

            case "sequence":
                sequence_tokenized = sequence_load_wrapper(
                    data_pointer=data_pointer,
                    input_source=input_source,
                    deeplake_inner_key=deeplake_inner_key,
                    split_on=input_type_info.split_on,
                    encode_func=input_object.encode_func,
                )
                prepared_sequence_inputs = prepare_sequence_data(
                    sequence_input_object=inputs_objects[input_name],
                    cur_file_content_tokenized=sequence_tokenized,
                    test_mode=test_mode,
                )
                prepared_inputs[input_name] = prepared_sequence_inputs

            case "bytes":
                bytes_data = bytes_load_wrapper(
                    data_pointer=data_pointer,
                    dtype=input_type_info.byte_encoding,
                    input_source=input_source,
                    deeplake_inner_key=deeplake_inner_key,
                )
                prepared_bytes_input = prepare_bytes_data(
                    bytes_input_object=inputs_objects[input_name],
                    bytes_data=bytes_data,
                    test_mode=test_mode,
                )
                prepared_inputs[input_name] = prepared_bytes_input

            case "image":
                image_data = image_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    deeplake_inner_key=deeplake_inner_key,
                )

                prepared_image_data = prepare_image_data(
                    image_input_object=inputs_objects[input_name],
                    image_data=image_data,
                    test_mode=test_mode,
                )
                prepared_inputs[input_name] = prepared_image_data

            case "array":
                array_data = array_load_wrapper(
                    input_source=input_source,
                    data_pointer=data_pointer,
                    deeplake_inner_key=deeplake_inner_key,
                )
                prepared_array_data = prepare_array_data(array_data=array_data)
                prepared_inputs[input_name] = prepared_array_data

            case _:
                prepared_inputs[input_name] = inputs[input_name]

    return prepared_inputs


def prepare_inputs_memory(
    inputs: Dict[str, Any], inputs_objects: "al_input_objects_as_dict", test_mode: bool
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}

    for name, data in inputs.items():
        input_object = inputs_objects[name]

        input_type_info = input_object.input_config.input_type_info
        input_type = input_object.input_config.input_info.input_type

        match input_type:
            case "omics":
                array_raw_in_memory = data
                array_prepared = prepare_one_hot_omics_data(
                    genotype_array=array_raw_in_memory,
                    na_augment_perc=input_type_info.na_augment_perc,
                    na_augment_prob=input_type_info.na_augment_prob,
                    test_mode=test_mode,
                )
                prepared_inputs[name] = array_prepared

            case "sequence":
                sequence_raw_in_memory = data
                prepared_sequence_inputs = prepare_sequence_data(
                    sequence_input_object=inputs_objects[name],
                    cur_file_content_tokenized=sequence_raw_in_memory,
                    test_mode=test_mode,
                )
                prepared_inputs[name] = prepared_sequence_inputs

            case "bytes":
                bytes_raw_in_memory = data
                prepared_bytes_input = prepare_bytes_data(
                    bytes_input_object=inputs_objects[name],
                    bytes_data=bytes_raw_in_memory,
                    test_mode=test_mode,
                )

                prepared_inputs[name] = prepared_bytes_input

            case "image":
                image_raw_in_memory = data
                prepared_image_data = prepare_image_data(
                    image_input_object=inputs_objects[name],
                    image_data=image_raw_in_memory,
                    test_mode=test_mode,
                )
                prepared_inputs[name] = prepared_image_data

            case "array":
                array_raw_in_memory = data
                prepared_inputs[name] = array_raw_in_memory

            case _:
                prepared_inputs[name] = inputs[name]

    return prepared_inputs


def get_data_loading_hooks(
    inputs: al_input_objects_as_dict,
) -> Mapping[str, Callable[..., torch.Tensor]]:
    mapping = {}

    for input_name, input_object in inputs.items():
        input_type = input_object.input_config.input_info.input_type
        input_source = input_object.input_config.input_info.input_source
        inner_key = input_object.input_config.input_info.input_inner_key

        match input_type:
            case "omics":
                mapping[input_name] = partial(
                    omics_load_wrapper,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                    subset_indices=inputs[input_name].subset_indices,
                )

            case "image":
                mapping[input_name] = partial(
                    image_load_wrapper,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                )

            case "sequence":
                mapping[input_name] = partial(
                    sequence_load_wrapper,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                    split_on=input_object.input_config.input_type_info.split_on,
                    encode_func=inputs[input_name].encode_func,
                )

            case "bytes":
                mapping[input_name] = partial(
                    bytes_load_wrapper,
                    dtype=input_object.input_config.input_type_info.byte_encoding,
                    input_source=input_source,
                )

            case "array":
                mapping[input_name] = partial(
                    array_load_wrapper,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                )

    return mapping
