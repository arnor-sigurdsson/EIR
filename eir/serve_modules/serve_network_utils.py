import base64
from io import BytesIO
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import torch
from aislib.misc_utils import get_logger
from PIL import Image
from sklearn.preprocessing import StandardScaler

from eir.data_load.label_setup import (
    al_label_transformers,
    al_label_transformers_object,
)
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
from eir.setup.schemas import SequenceInputDataConfig
from eir.train_utils.evaluation_handlers.evaluation_handlers_utils import (
    streamline_sequence_manual_data,
)

logger = get_logger(name=__name__, tqdm_compatible=True)


def prepare_request_input_data(
    request_data: Dict[str, Any],
    input_objects: al_input_objects_as_dict,
) -> Dict[str, Any]:
    inputs_prepared: dict[str, np.ndarray | torch.Tensor | list[str] | str | dict] = {}

    for name, serialized_data in request_data.items():
        input_object = input_objects[name]
        input_type = input_object.input_config.input_info.input_type
        input_type_info = input_object.input_config.input_type_info

        match input_object:
            case ComputedOmicsInputInfo():
                assert input_type == "omics"
                shape = input_object.data_dimensions.full_shape()[1:]
                array_np = _deserialize_array(
                    array_str=serialized_data,
                    dtype=np.bool_,
                    shape=shape,
                )
                assert len(array_np.shape) == 2
                array_raw = torch.from_numpy(array_np)

                inputs_prepared[name] = array_raw

            case ComputedSequenceInputInfo():
                assert input_type == "sequence"
                assert isinstance(input_type_info, SequenceInputDataConfig)

                sequence_streamlined = streamline_sequence_manual_data(
                    data=serialized_data,
                    split_on=input_type_info.split_on,
                )

                inputs_prepared[name] = sequence_streamlined

            case ComputedBytesInputInfo():
                assert input_type == "bytes"
                array_np = _deserialize_array(
                    array_str=serialized_data,
                    dtype=np.uint8,
                    shape=(-1,),
                )
                array_raw = torch.from_numpy(array_np).to(dtype=torch.int64)
                inputs_prepared[name] = array_raw

            case ComputedImageInputInfo():
                assert input_type == "image"
                image_data = _deserialize_image(image_str=serialized_data)
                inputs_prepared[name] = image_data

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                assert input_type == "tabular"
                transformers = input_object.labels.label_transformers
                tabular_data = _streamline_tabular_request_data(
                    tabular_input=serialized_data, transformers=transformers
                )
                inputs_prepared[name] = tabular_data

            case ComputedArrayInputInfo():
                assert input_type == "array"
                array_np = _deserialize_array(
                    array_str=serialized_data,
                    dtype=input_object.dtype,
                    shape=input_object.data_dimensions.full_shape(),
                )
                inputs_prepared[name] = array_np

            case _:
                raise ValueError(f"Unknown input type '{input_type}'")

    return inputs_prepared


def _streamline_tabular_request_data(
    tabular_input: Dict, transformers: al_label_transformers
) -> Dict:
    parsed_output = {}
    for name, value in tabular_input.items():
        cur_transformer = transformers[name]

        value_transformed = _parse_transformer_output(
            transformer=cur_transformer, value=value
        )

        parsed_output[name] = value_transformed

    return parsed_output


def _parse_transformer_output(
    transformer: al_label_transformers_object, value: float | int
) -> Union[float, int]:
    value_parsed: list
    if isinstance(transformer, StandardScaler):
        value_parsed = [[value]]
    else:
        value_parsed = [value]

    value_transformed = transformer.transform(value_parsed)
    if isinstance(transformer, StandardScaler):
        value_transformed = value_transformed[0][0]
    else:
        value_transformed = value_transformed[0]

    assert not isinstance(value_transformed, list)

    return value_transformed


def _deserialize_array(
    array_str: str, dtype: npt.DTypeLike, shape: tuple
) -> np.ndarray:
    array_bytes = base64.b64decode(array_str)
    return np.frombuffer(array_bytes, dtype=dtype).reshape(shape).copy()


def _deserialize_image(image_str: str) -> Image.Image:
    """
    Note we convert to RGB to be compatible with the default_loader.
    """
    image_data = base64.b64decode(image_str)
    image = Image.open(BytesIO(image_data))
    return image.convert("RGB")
