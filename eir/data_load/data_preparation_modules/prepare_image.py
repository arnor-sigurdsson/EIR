from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL.Image import Image, fromarray
from torchvision.datasets.folder import default_loader

from eir.data_load.data_preparation_modules.common import _load_deeplake_sample
from eir.data_load.data_source_modules import deeplake_ops
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo


def prepare_image_data(
    image_input_object: "ComputedImageInputInfo", image_data: Image, test_mode: bool
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
    deeplake_inner_key: Optional[str] = None,
) -> Image:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        assert isinstance(data_pointer, int)
        image_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        pil_image = fromarray(obj=np.uint8(image_data * 255))
    else:
        assert isinstance(data_pointer, Path)
        pil_image = default_loader(path=str(data_pointer))

    return pil_image
