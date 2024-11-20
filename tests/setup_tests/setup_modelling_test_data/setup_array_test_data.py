from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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


def create_test_array_data_and_labels(
    test_data_config: "TestDataConfig", array_output_folder: Path
) -> Path:
    c = test_data_config

    fieldnames = get_test_label_file_fieldnames()
    label_file_handle, label_file_writer = set_up_label_file_writing(
        base_path=c.scoped_tmp_path, fieldnames=fieldnames, extra_name="_arrays"
    )

    array_output_folder = set_up_test_data_root_outpath(base_folder=array_output_folder)

    for cls, cls_integer in c.target_classes.items():
        for sample_idx in range(c.n_per_class):
            sample_output_path = array_output_folder / f"{sample_idx}_{cls}"

            num_active_elements_in_sample = _create_and_save_test_array(
                test_data_config=c,
                sample_output_path=sample_output_path,
                class_integer=cls_integer,
            )

            label_line_base = set_up_label_line_dict(
                sample_name=sample_output_path.stem,
                fieldnames=fieldnames,
            )

            label_line_dict = get_current_test_label_values(
                values_dict=label_line_base,
                num_active_elements_in_sample=len(num_active_elements_in_sample),
                cur_class=cls,
            )
            label_file_writer.writerow(label_line_dict)

    label_file_handle.close()

    return array_output_folder


def _create_and_save_test_array(
    test_data_config: "TestDataConfig",
    sample_output_path: Path,
    class_integer: int,
):
    c = test_data_config
    dims = c.extras["array_dims"]

    base_array, elements_active = _set_up_base_test_array(
        dims=dims,
        class_integer=class_integer,
        nan_probability=0.05,
        nan_array_probability=0.1,
    )

    np.save(str(sample_output_path), base_array)

    return elements_active


def _set_up_base_test_array(
    dims: int,
    class_integer: int,
    nan_probability: float = 0.0,
    nan_array_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.array(list(range(0, 100, 10)))

    lower_bound, upper_bound = 4, 11

    np.random.shuffle(candidates)
    num_elements_this_sample = np.random.randint(lower_bound, upper_bound)
    elements_active = np.array(sorted(candidates[:num_elements_this_sample]))

    base_array = np.zeros(shape=100)
    base_array[elements_active] = float(class_integer)

    if dims == 2:
        base_array = np.tile(base_array, reps=(4, 1))
    elif dims == 3:
        base_array = np.tile(base_array, reps=(2, 4, 1))

    if nan_probability > 0 and np.random.random() < nan_array_probability:
        mask = np.random.random(base_array.shape) < nan_probability
        base_array[mask] = np.nan

    return base_array, elements_active
