import csv
from pathlib import Path
from typing import List

import numpy as np


def set_up_label_file_writing(
    base_path: Path, fieldnames: List[str], extra_name: str = ""
):
    label_file = str(base_path / f"labels{extra_name}.csv")

    label_file_handle = open(str(label_file), "w")

    writer = csv.DictWriter(f=label_file_handle, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    return label_file_handle, writer


def set_up_label_line_dict(sample_name: str, fieldnames: List[str]):
    label_line_dict = {k: None for k in fieldnames}
    assert "ID" in label_line_dict.keys()
    label_line_dict["ID"] = sample_name
    return label_line_dict


def get_current_test_label_values(
    values_dict: dict, num_active_elements_in_sample: int, cur_class: str
) -> dict:
    class_base_heights = {"Asia": 120, "Europe": 140, "Africa": 160}
    cur_base_height = class_base_heights[cur_class]

    added_height = 5 * num_active_elements_in_sample
    noise = np.random.randn()

    height_value = cur_base_height + added_height + noise
    values_dict["Height"] = height_value
    values_dict["ExtraTarget"] = height_value - 50
    values_dict["SparseHeight"] = height_value

    values_dict["Origin"] = cur_class
    values_dict["OriginExtraCol"] = cur_class
    values_dict["SparseOrigin"] = cur_class

    return values_dict


def get_test_label_file_fieldnames() -> list[str]:
    fieldnames = [
        "ID",
        "Origin",
        "Height",
        "OriginExtraCol",
        "ExtraTarget",
        "SparseHeight",
        "SparseOrigin",
    ]

    return fieldnames
