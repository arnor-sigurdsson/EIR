from typing import Any, Dict, Literal, Tuple, Union

import torch

from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_array import ComputedArrayInputInfo
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.schemas import ImageInputDataConfig, TabularInputDataConfig

al_fill_values = dict[str, bool | int | float | dict[str, int | float]]


def impute_missing_modalities_wrapper(
    inputs_values: dict[str, Any], inputs_objects: "al_input_objects_as_dict"
) -> Dict[str, torch.Tensor]:
    impute_dtypes = _get_default_impute_dtypes(inputs_objects=inputs_objects)
    impute_fill_values = _get_default_impute_fill_values(inputs_objects=inputs_objects)
    inputs_imputed = impute_missing_modalities(
        inputs_values=inputs_values,
        inputs_objects=inputs_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    return inputs_imputed


def impute_missing_modalities(
    inputs_values: dict[str, Any],
    inputs_objects: al_input_objects_as_dict,
    fill_values: al_fill_values,
    dtypes: dict[str, Any],
) -> dict[str, torch.Tensor]:
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_name not in inputs_values:
            fill_value = fill_values[input_name]
            dtype = dtypes[input_name]

            shape: Tuple[int, ...]
            approach: Literal["constant", "random"]
            match input_object:
                case ComputedOmicsInputInfo():
                    assert input_type == "omics"
                    dimensions = input_object.data_dimensions
                    shape = dimensions.channels, dimensions.height, dimensions.width
                    approach = "constant"

                case ComputedSequenceInputInfo() | ComputedBytesInputInfo():
                    assert input_type in ("sequence", "bytes")
                    max_length = input_object.computed_max_length
                    shape = (max_length,)
                    approach = "constant"

                case ComputedImageInputInfo():
                    assert input_type == "image"
                    input_type_info = input_object.input_config.input_type_info
                    assert isinstance(input_type_info, ImageInputDataConfig)
                    size = input_type_info.size
                    if len(size) == 1:
                        size = [size[0], size[0]]

                    num_channels = input_object.num_channels
                    shape = (num_channels, *size)
                    approach = "random"

                case (
                    ComputedTabularInputInfo()
                    | ComputedPredictTabularInputInfo()
                    | ComputedServeTabularInputInfo()
                ):
                    assert input_type == "tabular"
                    inputs_values[input_name] = fill_value
                    continue

                case ComputedArrayInputInfo():
                    assert input_type == "array"
                    shape = input_object.data_dimensions.full_shape()
                    approach = "random"

                case _:
                    raise ValueError(f"Unrecognized input type {input_type}")

            imputed_tensor = impute_single_missing_modality(
                shape=shape,
                fill_value=fill_value,
                dtype=dtype,
                approach=approach,
            )
            inputs_values[input_name] = imputed_tensor

    return inputs_values


def impute_single_missing_modality(
    shape: Tuple[int, ...],
    fill_value: Any,
    dtype: Any,
    approach: Literal["constant", "random"],
) -> torch.Tensor:
    match approach:
        case "constant":
            imputed_tensor = torch.empty(shape, dtype=dtype).fill_(fill_value)
        case "random":
            imputed_tensor = torch.empty(shape, dtype=dtype).normal_()
        case _:
            raise ValueError(f"Unrecognized approach {approach}")

    return imputed_tensor


def _get_default_impute_fill_values(
    inputs_objects: "al_input_objects_as_dict",
) -> al_fill_values:
    fill_values: al_fill_values = {}
    for input_name, input_object in inputs_objects.items():
        match input_object:
            case ComputedOmicsInputInfo():
                fill_values[input_name] = False

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                fill_values[input_name] = _build_tabular_fill_value(
                    input_object=input_object
                )

            case (
                ComputedSequenceInputInfo()
                | ComputedBytesInputInfo()
                | ComputedImageInputInfo()
                | ComputedArrayInputInfo()
            ):
                fill_values[input_name] = 0.0
            case _:
                raise ValueError(
                    f"Unrecognized input type"
                    f" {input_object.input_config.input_info.input_type}"
                )

    return fill_values


def _build_tabular_fill_value(
    input_object: Union[
        "ComputedTabularInputInfo",
        "ComputedPredictTabularInputInfo",
        "ComputedServeTabularInputInfo",
    ]
) -> dict[str, int | float]:
    fill_value = {}
    transformers = input_object.labels.label_transformers

    input_type_info = input_object.input_config.input_type_info
    assert isinstance(input_type_info, TabularInputDataConfig)

    cat_columns = input_type_info.input_cat_columns
    for cat_column in cat_columns:
        cur_label_encoder = transformers[cat_column]
        fill_value[cat_column] = cur_label_encoder.transform(["NA"]).item()

    con_columns = input_type_info.input_con_columns
    for con_column in con_columns:
        fill_value[con_column] = 0.0

    return fill_value


def _get_default_impute_dtypes(inputs_objects: "al_input_objects_as_dict"):
    dtypes = {}
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type
        if input_type == "omics":
            dtypes[input_name] = torch.bool
        elif input_type in ("sequence", "bytes"):
            dtypes[input_name] = torch.long
        else:
            dtypes[input_name] = torch.float

    return dtypes
