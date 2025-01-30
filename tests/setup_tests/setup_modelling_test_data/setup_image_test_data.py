import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tests.setup_tests.setup_modelling_test_data.setup_targets_test_data import (
    get_current_test_label_values,
    get_test_label_file_fieldnames,
    set_up_label_file_writing,
    set_up_label_line_dict,
)
from tests.setup_tests.setup_modelling_test_data.setup_test_data_utils import (
    set_up_test_data_root_outpath,
)

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_data import TestDataConfig


def create_test_image_data(
    test_data_config: "TestDataConfig", image_output_folder: Path
) -> Path:
    c = test_data_config

    fieldnames = get_test_label_file_fieldnames()
    label_file_handle, label_file_writer = set_up_label_file_writing(
        base_path=c.scoped_tmp_path, fieldnames=fieldnames, extra_name="_image"
    )

    image_output_folder = set_up_test_data_root_outpath(base_folder=image_output_folder)

    for cls, _snp_row_idx in c.target_classes.items():
        for sample_idx in range(c.n_per_class):
            sample_outpath = image_output_folder / f"{sample_idx}_{cls}.png"

            cur_image_object = create_test_image(
                size=16,
                patch_size=4,
                target_class=cls,
            )
            cur_image_object.image.save(fp=sample_outpath)

            label_line_base = set_up_label_line_dict(
                sample_name=sample_outpath.stem, fieldnames=fieldnames
            )

            label_line_dict = get_current_test_label_values(
                values_dict=label_line_base,
                num_active_elements_in_sample=cur_image_object.active_pixels_in_sample,
                cur_class=cls,
            )
            label_file_writer.writerow(label_line_dict)

    label_file_handle.close()

    return image_output_folder


@dataclass
class ImageTestSample:
    image: Image
    target_class: str
    active_pixels_in_sample: int


def create_test_image(
    size: int,
    target_class: str,
    patch_size: int,
    min_active_patch_pixels: int = 4,
    max_active_patch_pixels: int = 16,
) -> ImageTestSample:
    sample_range = list(range(min_active_patch_pixels, max_active_patch_pixels))
    num_active_in_sample_patch = random.choice(sample_range)
    active_patch_indices = random.sample(
        population=sample_range, k=num_active_in_sample_patch - min_active_patch_pixels
    )

    patch_base = np.zeros(patch_size**2)
    for index in active_patch_indices:
        patch_base[index] = 255
    patch_base = patch_base.reshape(patch_size, patch_size).astype(dtype=np.uint8)

    image_base = np.zeros((size, size), dtype=np.uint8)
    if target_class == "Africa":
        image_base[0:4, 0:4] = patch_base
    elif target_class == "Asia":
        image_base[0:4, 12:] = patch_base
    elif target_class == "Europe":
        image_base[12:, 12:] = patch_base
    else:
        raise ValueError(f"Unknown target class '{target_class}'")

    img = Image.fromarray(image_base, mode="L")

    image_sample = ImageTestSample(
        image=img,
        target_class=target_class,
        active_pixels_in_sample=num_active_in_sample_patch,
    )

    return image_sample
