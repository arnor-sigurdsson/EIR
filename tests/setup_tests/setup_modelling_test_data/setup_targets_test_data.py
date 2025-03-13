import csv
from pathlib import Path

import numpy as np


def set_up_label_file_writing(
    base_path: Path, fieldnames: list[str], extra_name: str = ""
):
    label_file = str(base_path / f"labels{extra_name}.csv")

    label_file_handle = open(str(label_file), "w")

    writer = csv.DictWriter(f=label_file_handle, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    return label_file_handle, writer


def set_up_label_line_dict(sample_name: str, fieldnames: list[str]):
    label_line_dict = dict.fromkeys(fieldnames)
    assert "ID" in label_line_dict
    label_line_dict["ID"] = sample_name
    return label_line_dict


def get_current_test_label_values(
    values_dict: dict,
    num_active_elements_in_sample: int,
    cur_class: str,
) -> dict:
    class_base_heights = {"Asia": 120, "Europe": 130, "Africa": 140}
    class_multiplier = {"Asia": 3.5, "Europe": 5.0, "Africa": 10.0}
    cur_base_height = class_base_heights[cur_class]
    cur_class_multiplier = class_multiplier[cur_class]

    added_height = cur_class_multiplier * num_active_elements_in_sample
    noise = np.random.randn()

    height_value = cur_base_height + added_height + noise

    values_dict["Height"] = height_value
    values_dict["ExtraTarget"] = height_value - 50
    values_dict["SparseHeight"] = height_value

    values_dict["Origin"] = cur_class
    values_dict["OriginExtraCol"] = cur_class
    values_dict["SparseOrigin"] = cur_class

    if cur_class == "Asia":
        values_dict["BinaryOrigin"] = 0
    else:
        values_dict["BinaryOrigin"] = 1

    base_time = 1000 if cur_class == "Asia" else 700

    height_effect = -1.5 * (height_value - 140)

    time_noise = np.random.normal(0, 10)

    time = max(50.0, base_time + height_effect + time_noise)
    values_dict["Time"] = time

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
        "BinaryOrigin",
        "Time",
    ]

    return fieldnames
