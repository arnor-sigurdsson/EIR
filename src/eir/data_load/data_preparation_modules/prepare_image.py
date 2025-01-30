from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL.Image import Image, fromarray

from eir.data_load.data_preparation_modules.common import _load_deeplake_sample
from eir.data_load.data_source_modules import deeplake_ops
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    default_image_loader,
)
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo


def prepare_image_data(
    image_input_object: Union["ComputedImageInputInfo", "ComputedImageOutputInfo"],
    image_data: Image,
    test_mode: bool,
) -> torch.Tensor:
    """
    The transforms take care of converting the image object to a copied tensor.
    """

    image_data_clone = image_data.copy()

    if test_mode:
        image_prepared = image_input_object.base_transforms(img=image_data_clone)
    else:
        image_prepared = image_input_object.all_transforms(img=image_data_clone)

    return image_prepared


def image_load_wrapper(
    data_pointer: Path | int,
    input_source: str,
    image_mode: str | None,
    deeplake_inner_key: str | None = None,
) -> Image:
    """
    Squeeze there since deeplake seems to support only 3D images. Therefore,
    when saving we generally add a channel dimension, e.g. (16, 16) -> (16, 16, 1).
    """
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        assert isinstance(data_pointer, int)
        array = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        assert isinstance(array, np.ndarray)

        if len(array.shape) == 3 and array.shape[2] == 1:
            array = array.squeeze(axis=-1)

        pil_image = fromarray(obj=array)
    else:
        assert isinstance(data_pointer, str | Path)
        pil_image = default_image_loader(path=str(data_pointer))

    if image_mode is not None:
        pil_image = pil_image.convert(mode=image_mode)

    return pil_image
