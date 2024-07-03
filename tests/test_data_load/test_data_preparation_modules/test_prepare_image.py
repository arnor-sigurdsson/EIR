from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from eir.data_load.data_preparation_modules import prepare_image
from eir.setup.input_setup_modules.setup_image import (
    ImageNormalizationStats,
    get_image_transforms,
)


@pytest.mark.parametrize("resize_approach", ["resize", "randomcrop", "centercrop"])
def test_prepare_image_data(resize_approach):
    normalization_stats = ImageNormalizationStats(means=[0], stds=[0.1])
    base_transforms, all_transforms = get_image_transforms(
        target_size=(32, 32),
        normalization_stats=normalization_stats,
        auto_augment=False,
        resize_approach=resize_approach,
    )

    input_config_mock = MagicMock()
    input_config_mock.base_transforms = base_transforms
    input_config_mock.all_transforms = all_transforms

    arr = np.random.rand(64, 64)
    image_data = Image.fromarray(np.uint8(arr * 255))
    arr_pil = np.array(image_data)

    prepared_tensor = prepare_image.prepare_image_data(
        image_input_object=input_config_mock, image_data=image_data, test_mode=False
    )

    assert (arr != image_data).any()
    assert arr_pil.shape == (64, 64)
    assert arr.shape == (64, 64)

    assert prepared_tensor.shape == (1, 32, 32)
