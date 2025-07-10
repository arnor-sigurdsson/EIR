from pathlib import Path
from typing import Union

import torch
from PIL.Image import Image

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
    image_mode: str | None,
) -> Image:
    assert isinstance(data_pointer, str | Path)
    pil_image = default_image_loader(path=str(data_pointer))

    if image_mode is not None:
        pil_image = pil_image.convert(mode=image_mode)

    return pil_image
