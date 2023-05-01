from typing import Dict, Any, Tuple

import torch
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_tabular import TabularInputInfo


def impute_missing_modalities_wrapper(
    inputs_values: Dict[str, Any], inputs_objects: "al_input_objects_as_dict"
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
    inputs_values: Dict[str, Any],
    inputs_objects: "al_input_objects_as_dict",
    fill_values: Dict[str, Any],
    dtypes: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_name not in inputs_values:
            fill_value = fill_values[input_name]
            dtype = dtypes[input_name]

            if input_type == "omics":
                dimensions = input_object.data_dimensions
                shape = dimensions.channels, dimensions.height, dimensions.width

                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "sequence":
                max_length = input_object.computed_max_length
                shape = (max_length,)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "bytes":
                max_length = input_object.input_config.input_type_info.max_length
                shape = (max_length,)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "image":
                size = input_object.input_config.input_type_info.size
                if len(size) == 1:
                    size = [size[0], size[0]]

                num_channels = input_object.num_channels
                shape = (num_channels, *size)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "tabular":
                inputs_values[input_name] = fill_value

            elif input_type == "array":
                shape = input_object.data_dimensions.full_shape()
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

    return inputs_values


def impute_single_missing_modality(
    shape: Tuple[int, ...], fill_value: Any, dtype: Any
) -> torch.Tensor:
    imputed_tensor = torch.empty(shape, dtype=dtype).fill_(fill_value)
    return imputed_tensor


def _get_default_impute_fill_values(inputs_objects: "al_input_objects_as_dict"):
    fill_values = {}
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_type == "omics":
            fill_values[input_name] = False
        elif input_type == "tabular":
            fill_values[input_name] = _build_tabular_fill_value(
                input_object=input_object
            )
        else:
            fill_values[input_name] = 0

    return fill_values


def _build_tabular_fill_value(input_object: "TabularInputInfo"):
    fill_value = {}
    transformers = input_object.labels.label_transformers

    cat_columns = input_object.input_config.input_type_info.input_cat_columns
    for cat_column in cat_columns:
        cur_label_encoder = transformers[cat_column]
        fill_value[cat_column] = cur_label_encoder.transform(["NA"]).item()

    con_columns = input_object.input_config.input_type_info.input_con_columns
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
