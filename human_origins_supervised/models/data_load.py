from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.functional import pad
from torch.utils.data import Dataset

from human_origins_supervised.label_loading_ops import COLUMN_OPS

logger = get_logger(__name__)


def get_meta_from_label_file(
    label_fpath: Path, label_column: str, ids_to_keep: Union[List[str], None] = None
) -> pd.DataFrame:
    """
    We want to set up a dataframe containing the labels for the current modelling task
    we are dealing with. We also don't want to create a new label file for each run
    rather just refer to a "master" file for consistency.

    Hence this function just reads in the label file, and only returns rows from the
    master df which match `ids_to_keep`.

    :param label_fpath: Path to the label .csv file.
    :param label_column: Which column to grab (currently only supports one).
    :param ids_to_keep: Which IDs to grab from the label file.
    :return: Dataframe with ID as index and only the IDs listed in `ids_to_filter`.
    """
    df = pd.read_csv(label_fpath, usecols=["ID", label_column], dtype={"ID": str})
    df = df.set_index("ID")

    if ids_to_keep:
        df = df[df.index.isin(ids_to_keep)]

    return df


def parse_label_df(
    df, column_ops: Dict[str, List[Tuple[Callable, Dict]]]
) -> pd.DataFrame:
    """
    We want to be able to dynamically apply various operations to different columns
    in the label file (e.g. different operations for creating obesity labels or parsing
    country of origin).

    :param df: Dataframe to perform processing on.
    :param column_ops: A dictionarity of colum names, where each value is a list
    of tuples, where each tuple is a callable as the first element and the callable's
    arguments as the second element.
    :return: Parsed dataframe.
    """

    for column_name, ops_funcs in column_ops.items():
        if column_name in df.columns:
            for func, args_dict in ops_funcs:
                logger.debug("Applying func %s to column in pre-processing.", func)
                df = func(df=df, column_name=column_name, **args_dict)

    return df


@dataclass
class Sample:
    sample_id: str
    array: torch.Tensor
    label: Union[Dict[str, str], float, None]


def set_up_dataset_labels(
    cl_args: Namespace, all_ids: List[str], train_ids: List[str], valid_ids: List[str]
) -> Tuple[Dict[str, str], Dict[str, str]]:

    df_labels = get_meta_from_label_file(
        cl_args.label_file, cl_args.label_column, all_ids
    )
    df_labels = parse_label_df(df_labels, COLUMN_OPS)

    df_labels_train = df_labels.loc[train_ids]
    df_labels_valid = df_labels.loc[valid_ids]

    if cl_args.model_task == "reg":
        reg_col = cl_args.label_column
        logger.debug("Applying standard scaling to column %s.", reg_col)

        scaler = StandardScaler()
        scaler.fit(df_labels_train[reg_col])

        df_labels_train[reg_col] = scaler.transform(df_labels_train[reg_col])
        df_labels_valid[reg_col] = scaler.transform(df_labels_valid[reg_col])

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")
    return train_labels_dict, valid_labels_dict


def set_up_datasets(
    cl_args: Namespace, with_labels: bool = True, valid_fraction=0.1
) -> Tuple[
    Union["MemoryArrayDataset", "DiskArrayDataset"],
    Union["MemoryArrayDataset", "DiskArrayDataset"],
]:

    dataset = DiskArrayDataset
    if cl_args.memory_dataset:
        dataset = MemoryArrayDataset

    all_ids = [i.stem for i in Path(cl_args.data_folder).iterdir()]
    train_ids, valid_ids = train_test_split(all_ids, test_size=valid_fraction)

    train_labels = None
    valid_labels = None

    if with_labels:
        train_labels, valid_labels = set_up_dataset_labels(
            cl_args=cl_args, all_ids=all_ids, train_ids=train_ids, valid_ids=valid_ids
        )

    dataset_class_common_args = {
        "data_folder": cl_args.data_folder,
        "model_task": cl_args.model_task,
        "target_width": cl_args.target_width,
        "label_column": cl_args.label_column,
        "data_type": cl_args.data_type,
    }
    train_dataset = dataset(
        **dataset_class_common_args, ids=train_ids, labels_dict=train_labels
    )
    valid_dataset = dataset(
        **dataset_class_common_args,
        ids=valid_ids,
        labels_dict=valid_labels,
        label_encoder=train_dataset.label_encoder,
    )

    assert len(train_dataset) > len(valid_dataset)
    assert set(valid_dataset.ids).isdisjoint(train_dataset.ids)

    return train_dataset, valid_dataset


class ArrayDatasetBase(Dataset):
    def __init__(
        self,
        data_folder: Path,
        model_task: str,
        ids=List[str],
        label_column: str = None,
        labels_dict: Dict[str, str] = None,
        label_encoder=None,
        target_height: int = 4,
        target_width: int = None,
        data_type: str = "packbits",
    ):
        super().__init__()

        self.data_folder = data_folder
        self.model_task = model_task
        self.ids = ids
        self.target_height = target_height
        self.target_width = target_width
        self.data_type = data_type

        self.samples: Union[List[Sample], None] = None

        self.label_column = label_column
        self.labels_dict = labels_dict if labels_dict else {}
        self.labels_unique = None
        self.num_classes = None
        self.label_encoder = label_encoder

    def get_samples(self, array_hook: Callable = lambda x: x):
        files = {i.stem: i for i in Path(self.data_folder).iterdir()}
        samples = []

        for sample_id in self.ids:
            cur_sample = Sample(
                sample_id=sample_id,
                array=array_hook(files.get(sample_id)),
                label=self.labels_dict.get(sample_id, None),
            )
            samples.append(cur_sample)

        return samples

    @property
    def data_width(self):
        raise NotImplementedError

    def init_label_attributes(self):
        if not self.label_column:
            raise ValueError("Please specify label column name.")

        non_labelled = tuple(i for i in self.samples if not i.label)
        if non_labelled:
            raise ValueError(
                f"Expected all observations to have a label associated "
                f"with them, but got {non_labelled}."
            )

        if self.model_task == "cls":
            self.labels_unique = sorted(
                np.unique([i.label[self.label_column] for i in self.samples])
            )
            self.num_classes = len(self.labels_unique)

            if not self.label_encoder:
                self.label_encoder = LabelEncoder().fit(self.labels_unique)

                le_it = self.label_encoder.inverse_transform
                assert le_it([0]) == self.labels_unique[0]

        elif self.model_task == "reg":
            self.num_classes = 1


class MemoryArrayDataset(ArrayDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = self.get_samples(array_hook=self.mem_sample_loader)

        if self.labels_dict:
            self.init_label_attributes()

    def mem_sample_loader(self, sample_fpath):
        """
        A small hook to actually load the arrays into `self.samples` instead of just
        pointing to filenames.
        """
        array = np.load(sample_fpath)
        if self.data_type == "packbits":
            array = np.unpackbits(array).reshape(self.target_height, -1)
        elif self.data_type != "uint8":
            raise ValueError

        array = array.astype(np.uint8)
        return torch.from_numpy(array).unsqueeze(0)

    @property
    def data_width(self):
        data = self.samples[0].array
        return data.shape[1]

    def __getitem__(self, index: int):
        sample = self.samples[index]

        array = sample.array
        label = (
            self.label_encoder.transform([sample.label[self.label_column]])
            if self.labels_dict
            else []
        ).squeeze()
        sample_id = sample.sample_id

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        return array, label, sample_id

    def __len__(self):
        return len(self.samples)


class DiskArrayDataset(ArrayDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = self.get_samples()

        if self.labels_dict:
            self.init_label_attributes()

    @property
    def data_width(self):
        data = np.load(self.samples[0].array)
        if self.data_type == "packbits":
            data = np.unpackbits(data).reshape(self.target_height, -1)
        return data.shape[1]

    def __getitem__(self, index):
        sample = self.samples[index]

        array = np.load(sample.array)
        label = (
            self.label_encoder.transform([sample.label[self.label_column]])
            if self.labels_dict
            else []
        ).squeeze()
        sample_id = sample.sample_id

        if self.data_type == "packbits":
            array = np.unpackbits(array).reshape(self.target_height, -1)

        array = torch.from_numpy(array).unsqueeze(0)

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        return array, label, sample_id

    def __len__(self):
        return len(self.samples)
