from pathlib import Path
from typing import Optional, Union

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
    data_pointer: Union[Path, int],
    input_source: str,
    image_mode: Optional[str],
    deeplake_inner_key: Optional[str] = None,
) -> Image:
    """
    Squeeze there since loading a saved 2D image, e.g. saved as (16, 16),
    in deeplake will be loaded as (16, 16, 1).
    """
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        assert isinstance(data_pointer, int)
        image_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        array = np.uint8(image_data * 255).squeeze(axis=-1)  # type: ignore
        pil_image = fromarray(obj=array)
    else:
        assert isinstance(data_pointer, Path)
        pil_image = default_image_loader(path=str(data_pointer))

    if image_mode is not None:
        pil_image = pil_image.convert(mode=image_mode)

    return pil_image
