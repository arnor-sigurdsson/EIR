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
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import (
    ImageInputDataConfig,
    ImageOutputTypeConfig,
    TabularInputDataConfig,
    TabularOutputTypeConfig,
)

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
        fill_value[cat_column] = cur_label_encoder.transform(["nan"]).item()

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


def impute_missing_output_modalities_wrapper(
    outputs_values: dict[str, Any], output_objects: "al_output_objects_as_dict"
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    impute_dtypes = _get_default_output_impute_dtypes(outputs_objects=output_objects)
    impute_fill_values = _get_default_output_impute_fill_values(
        outputs_objects=output_objects
    )
    outputs_imputed_modalities = impute_missing_output_modalities(
        outputs_values=outputs_values,
        outputs_objects=output_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    outputs_imputed = impute_partially_missing_output_modalities(
        outputs_values=outputs_imputed_modalities,
        output_objects=output_objects,
    )

    return outputs_imputed


def impute_partially_missing_output_modalities(
    outputs_values: dict[str, Any],
    output_objects: "al_output_objects_as_dict",
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    for output_name, output_object in output_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                cur_output_value = outputs_values[output_name]
                output_type_info = output_object.output_config.output_type_info

                assert isinstance(output_type_info, TabularOutputTypeConfig)

                cat_columns = set(output_type_info.target_cat_columns)
                con_columns = set(output_type_info.target_con_columns)
                output_columns = cat_columns.union(con_columns)

                for output_column in output_columns:
                    if output_column not in cur_output_value:
                        cur_output_value[output_column] = torch.nan

                outputs_values[output_name] = cur_output_value

    return outputs_values


def impute_missing_output_modalities(
    outputs_values: dict[str, Any],
    outputs_objects: "al_output_objects_as_dict",
    fill_values: al_fill_values,
    dtypes: dict[str, Any],
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    """
    Note that ultimately we never want to use these values for anything e.g. in the
    loss calculation, but rather just skip them completely. However, we do need
    the shapes to match for collation purposes.

    Note we skip imputing sequence outputs as they are handled on the fly
    separately based on the input.
    """
    for output_name, output_object in outputs_objects.items():
        output_type = output_object.output_config.output_info.output_type
        if output_name not in outputs_values:
            fill_value = fill_values[output_name]
            dtype = dtypes[output_name]

            shape: Tuple[int, ...]
            approach: Literal["constant", "random"]
            match output_object:
                case ComputedSequenceOutputInfo():
                    continue

                case ComputedArrayOutputInfo():
                    assert output_type == "array"
                    shape = output_object.data_dimensions.full_shape()
                    approach = "random"

                case ComputedTabularOutputInfo():
                    assert output_type == "tabular"
                    outputs_values[output_name] = fill_value
                    continue

                case ComputedImageOutputInfo():
                    assert output_type == "image"
                    output_type_info = output_object.output_config.output_type_info
                    assert isinstance(output_type_info, ImageOutputTypeConfig)
                    size = output_type_info.size
                    if len(size) == 1:
                        size = [size[0], size[0]]

                    num_channels = output_object.num_channels
                    shape = (num_channels, *size)
                    approach = "random"

                case _:
                    raise ValueError(f"Unrecognized output type {output_type}")

            imputed_tensor = impute_single_missing_modality(
                shape=shape,
                fill_value=fill_value,
                dtype=dtype,
                approach=approach,
            )
            outputs_values[output_name] = {output_name: imputed_tensor}

    return outputs_values


def _get_default_output_impute_dtypes(
    outputs_objects: al_output_objects_as_dict,
) -> dict[str, Any]:
    dtypes = {}
    for output_name, output_object in outputs_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                dtypes[output_name] = torch.float
            case ComputedSequenceOutputInfo():
                dtypes[output_name] = torch.long
            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                dtypes[output_name] = torch.float
            case _:
                raise ValueError(
                    f"Unrecognized output type"
                    f" {output_object.output_config.output_info.output_type}"
                )

    return dtypes


def _get_default_output_impute_fill_values(
    outputs_objects: al_output_objects_as_dict,
) -> al_fill_values:
    fill_values: al_fill_values = {}
    for output_name, output_object in outputs_objects.items():
        match output_object:
            case ComputedTabularOutputInfo():
                fill_values[output_name] = _build_tabular_output_fill_value(
                    output_object=output_object
                )
            case ComputedSequenceOutputInfo():
                fill_values[output_name] = 0
            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                fill_values[output_name] = torch.nan
            case _:
                raise ValueError(
                    f"Unrecognized output type"
                    f" {output_object.output_config.output_info.output_type}"
                )

    return fill_values


def _build_tabular_output_fill_value(
    output_object: Union["ComputedTabularOutputInfo",]
) -> dict[str, int | float]:
    fill_value: dict[str, int | float] = {}

    output_type_info = output_object.output_config.output_type_info
    assert isinstance(output_type_info, TabularOutputTypeConfig)

    cat_columns = output_type_info.target_cat_columns
    for cat_column in cat_columns:
        fill_value[cat_column] = torch.nan

    con_columns = output_type_info.target_con_columns
    for con_column in con_columns:
        fill_value[con_column] = torch.nan

    return fill_value
