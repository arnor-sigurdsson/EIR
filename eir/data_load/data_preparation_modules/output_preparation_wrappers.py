from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL.Image import Image

from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    typed_partial_for_hook,
)
from eir.data_load.data_preparation_modules.prepare_array import (
    array_load_wrapper,
    prepare_array_data,
)
from eir.data_load.data_preparation_modules.prepare_image import (
    image_load_wrapper,
    prepare_image_data,
)
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import (
    ComputedImageOutputInfo,
    ImageOutputTypeConfig,
)


def prepare_outputs_disk(
    outputs: dict[str, Any],
    output_objects: "al_output_objects_as_dict",
    test_mode: bool,
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    prepared_outputs: dict[str, dict[str, torch.Tensor | int | float]] = {}

    for output_name, output in outputs.items():
        output_object = output_objects[output_name]
        output_source = output_object.output_config.output_info.output_source
        deeplake_inner_key = output_object.output_config.output_info.output_inner_key

        output_prepared: dict[str, torch.Tensor | int | float]
        match output_object:
            case ComputedArrayOutputInfo():
                data_pointer = output[output_name]
                loaded_array = array_load_wrapper(
                    data_pointer=data_pointer,
                    input_source=output_source,
                    deeplake_inner_key=deeplake_inner_key,
                )
                array_prepared = prepare_array_data(
                    array_data=loaded_array,
                    normalization_stats=output_object.normalization_stats,
                )
                output_prepared = {output_name: array_prepared}

            case ComputedImageOutputInfo():
                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, ImageOutputTypeConfig)
                data_pointer = output[output_name]
                loaded_image = image_load_wrapper(
                    data_pointer=data_pointer,
                    input_source=output_source,
                    image_mode=output_type_info.mode,
                    deeplake_inner_key=deeplake_inner_key,
                )
                image_prepared = prepare_image_data(
                    image_input_object=output_object,
                    image_data=loaded_image,
                    test_mode=test_mode,
                )
                output_prepared = {output_name: image_prepared}

            case _:
                output_prepared = output

        prepared_outputs[output_name] = output_prepared

    return prepared_outputs


def prepare_outputs_memory(
    outputs: dict[str, Any],
    output_objects: "al_output_objects_as_dict",
    test_mode: bool,
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    prepared_outputs: dict[str, dict[str, torch.Tensor | int | float]] = {}

    for output_name, output in outputs.items():
        output_object = output_objects[output_name]

        match output_object:
            case ComputedArrayOutputInfo():
                loaded_array = output[output_name]
                array_prepared = prepare_array_data(
                    array_data=loaded_array,
                    normalization_stats=output_object.normalization_stats,
                )
                output_prepared: dict[str, torch.Tensor | int | float] = {
                    output_name: array_prepared
                }

            case ComputedImageOutputInfo():
                loaded_image = output[output_name]
                image_prepared = prepare_image_data(
                    image_input_object=output_object,
                    image_data=loaded_image,
                    test_mode=test_mode,
                )
                output_prepared = {output_name: image_prepared}

            case _:
                output_prepared = output

        prepared_outputs[output_name] = output_prepared

    return prepared_outputs


def get_output_data_loading_hooks(
    outputs: al_output_objects_as_dict,
) -> Optional[dict[str, Callable[..., np.ndarray | Image]]]:
    mapping = {}

    for output_name, output_object in outputs.items():
        output_info = output_object.output_config.output_info
        common_kwargs = {
            "input_source": output_info.output_source,
            "deeplake_inner_key": output_info.output_inner_key,
        }

        match output_object:
            case ComputedArrayOutputInfo():
                inner_function = typed_partial_for_hook(
                    array_load_wrapper,
                    **common_kwargs,
                )

                mapping[output_name] = typed_partial_for_hook(
                    _extract_nested_output_and_call,
                    output_name=output_name,
                    function=inner_function,
                )

            case ComputedImageOutputInfo():
                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, ImageOutputTypeConfig)
                inner_function = typed_partial_for_hook(
                    image_load_wrapper,
                    **common_kwargs,
                    image_mode=output_type_info.mode,
                )

                mapping[output_name] = typed_partial_for_hook(
                    _extract_nested_output_and_call,
                    output_name=output_name,
                    function=inner_function,
                )

    return mapping


def _extract_nested_output_and_call(
    output_dict: dict[str, Path | int],
    output_name: str,
    function: Callable[[Path | int], np.ndarray | Image],
) -> dict[str, np.ndarray | Image]:
    extracted_output_data_pointer = output_dict[output_name]

    loaded_output = function(extracted_output_data_pointer)

    return {output_name: loaded_output}
