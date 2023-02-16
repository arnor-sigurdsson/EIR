import csv
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Sequence, Iterable, Union, Dict, Callable, Optional

import numpy as np
import pandas as pd


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
    values_dict, num_active_elements_in_sample: int, cur_class: str
):
    class_base_heights = {"Asia": 120, "Europe": 140, "Africa": 160}
    cur_base_height = class_base_heights[cur_class]

    added_height = 5 * num_active_elements_in_sample
    noise = np.random.randn()

    height_value = cur_base_height + added_height + noise
    values_dict["Height"] = height_value
    values_dict["ExtraTarget"] = height_value - 50

    values_dict["Origin"] = cur_class
    values_dict["OriginExtraCol"] = cur_class

    return values_dict


def set_up_test_data_root_outpath(base_folder: Path) -> Path:
    if not base_folder.exists():
        base_folder.mkdir()

    return base_folder


def common_split_test_data_wrapper(
    test_folder: Path,
    name: str,
    post_split_callables: Optional[Dict[str, Callable]] = None,
) -> None:
    train_ids = None
    test_ids = None
    train_labels_path = test_folder / "labels_train.csv"
    test_labels_path = test_folder / "labels_test.csv"

    data_folder = test_folder / name
    if (data_folder / "train_set").exists() or (data_folder / "test_set").exists():
        assert (data_folder / "train_set").exists()
        assert (data_folder / "test_set").exists()
        return

    if test_labels_path.exists() or train_labels_path.exists():
        assert train_labels_path.exists()
        assert test_labels_path.exists()

        train_ids = list(pd.read_csv(train_labels_path)["ID"].values)
        test_ids = list(pd.read_csv(test_labels_path)["ID"].values)

    train_files, test_files = split_test_file_folder(
        test_folder=data_folder, train_ids=train_ids, test_ids=test_ids
    )
    train_ids = get_ids_from_paths(paths=train_files)
    test_ids = get_ids_from_paths(paths=test_files)

    if not test_labels_path.exists():
        split_label_file(
            label_file_path=test_folder / "labels.csv",
            train_ids=train_ids,
            test_ids=test_ids,
        )

    if post_split_callables is not None and name in post_split_callables:
        cur_func = post_split_callables[name]
        cur_func(test_root_folder=test_folder, train_ids=train_ids, test_ids=test_ids)


def split_test_file_folder(
    test_folder: Path,
    train_ids: Union[Sequence[str], None] = None,
    test_ids: Union[Sequence[str], None] = None,
) -> Tuple[Sequence[Path], Sequence[Path]]:
    all_arrays = [i for i in test_folder.iterdir()]
    shuffle(all_arrays)

    train_array_test_set_folder = test_folder / "train_set"
    train_array_test_set_folder.mkdir()
    test_array_test_set_folder = test_folder / "test_set"
    test_array_test_set_folder.mkdir()

    if test_ids:
        test_arrays_test_set = [i for i in all_arrays if i.stem in test_ids]
    else:
        test_arrays_test_set = all_arrays[:200]
    for array_file in test_arrays_test_set:
        array_file.replace(test_array_test_set_folder / array_file.name)

    if train_ids:
        train_arrays_test_set = [i for i in all_arrays if i.stem in train_ids]
    else:
        train_arrays_test_set = all_arrays[200:]

    for array_file in train_arrays_test_set:
        array_file.replace(train_array_test_set_folder / array_file.name)

    return train_arrays_test_set, test_arrays_test_set


def split_label_file(
    label_file_path: Path, train_ids: Sequence[str], test_ids: Sequence[str]
) -> Tuple[Path, Path]:
    df = pd.read_csv(label_file_path).set_index("ID")
    df_train, df_test = split_df_by_index(df=df, train_ids=train_ids, test_ids=test_ids)
    label_file_path.unlink()

    train_outpath = label_file_path.parent / "labels_train.csv"
    test_outpath = label_file_path.parent / "labels_test.csv"

    df_train.to_csv(path_or_buf=str(train_outpath))
    df_test.to_csv(path_or_buf=str(test_outpath))

    return train_outpath, test_outpath


def get_ids_from_paths(paths: Iterable[Path]) -> Sequence[str]:
    return [i.stem for i in paths]


def split_df_by_index(
    df: pd.DataFrame, train_ids: Sequence[str], test_ids: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df.index.isin(train_ids)]
    df_test = df[df.index.isin(test_ids)]

    return df_train, df_test
