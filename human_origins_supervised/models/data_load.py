from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from aislib import data_load as data_utils
from aislib.misc_utils import get_logger
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import pad
from torch.utils.data import Dataset

from human_origins_supervised.label_loading_ops import COLUMN_OPS
from human_origins_supervised.label_loading_ops.common_ops import (
    filter_ids_from_array_and_id_lists,
)

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
) -> Dict[str, Dict[str, str]]:
    """
    We want to be able to dynamically apply various operations to different columns
    in the label file (e.g. different operations for creating obesity labels or parsing
    country of origin).

    :param df: Dataframe to perform processing on.
    :param column_ops:
    :return:
    """

    for column_name, ops_funcs in column_ops.items():
        if column_name in df.columns:
            for func, args_dict in ops_funcs:
                logger.debug("Applying func %s to column in pre-processing.", func)
                df = func(df=df, column_name=column_name, **args_dict)

    df_as_dict = df.to_dict("index")

    return df_as_dict


class CustomArrayDataset(Dataset):
    """
    TODO: Figure out how to subclass from this in a more sane way.
    """

    def __init__(
        self,
        data_folder,
        model_task,
        label_fpath=None,
        label_column=None,
        target_height=4,
        target_width=None,
        data_type="packbits",
        with_labels=True,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.label_fpath = label_fpath
        self.label_column = label_column
        self.target_height = target_height
        self.target_width = target_width
        self.data_type = data_type
        self.with_labels = with_labels
        self.model_task = model_task

        self.ids = None
        self.arrays = None

        self.labels_dict = None
        self.labels = None
        self.labels_unique = None
        self.num_classes = None
        self.label_encoder = None
        self.labels_numerical = None

    @property
    def data_width(self):
        raise NotImplementedError

    def init_label_attributes(self):
        if not self.ids:
            raise NotImplementedError("ids variable must be defined.")

        df_labels = get_meta_from_label_file(
            self.label_fpath, self.label_column, self.ids
        )

        self.labels_dict = parse_label_df(df_labels, COLUMN_OPS)
        self.ids, self.arrays = filter_ids_from_array_and_id_lists(
            list(self.labels_dict.keys()), self.ids, self.arrays
        )
        assert len(self.labels_dict) == len(self.ids) == len(self.arrays)

        self.labels = [self.labels_dict[i][self.label_column] for i in self.ids]

        if self.model_task == "cls":
            self.labels_unique = sorted(np.unique(self.labels))
            self.num_classes = len(np.unique(self.labels))
            self.label_encoder = LabelEncoder().fit(self.labels_unique)
            self.labels_numerical = self.label_encoder.transform(self.labels)

            le_it = self.label_encoder.inverse_transform
            assert le_it([0]) == self.labels_unique[0]

        elif self.model_task == "reg":
            self.labels_numerical = [float(i) for i in self.labels]
            self.num_classes = 1


class MemoryArrayDataset(CustomArrayDataset):

    """
    TODO: Update data_utils.load_np_arrays_from_folder to return ids.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.data_type == "uint8":
            files = [i for i in Path(self.data_folder).iterdir()]
            self.arrays = np.array([np.load(i).astype(np.uint8) for i in files])
            self.ids = [i.stem for i in files]
        else:
            loader = data_utils.load_np_packbits_from_folder
            self.arrays, self.ids = loader(
                self.data_folder, self.target_height, np.uint8, True
            )

        self.arrays = torch.from_numpy(self.arrays)

        if self.with_labels:
            self.init_label_attributes()

    @property
    def data_width(self):
        data = self.arrays[0]
        return data.shape[1]

    def __getitem__(self, index: int):
        data = self.arrays[index].unsqueeze(0)
        label = self.labels_numerical[index] if self.with_labels else []
        id_ = self.ids[index]

        if self.target_width:
            right_padding = self.target_width - data.shape[2]
            data = pad(data, [0, right_padding])

        return data, label, id_

    def __len__(self):
        return self.arrays.shape[0]


class DiskArrayDataset(CustomArrayDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.arrays = [i for i in Path(self.data_folder).iterdir()]
        self.ids = [i.stem for i in self.arrays]

        if self.with_labels:
            self.init_label_attributes()

    @property
    def data_width(self):
        data = np.load(self.arrays[0])
        if self.data_type == "packbits":
            data = np.unpackbits(data).reshape(self.target_height, -1)
        return data.shape[1]

    def __getitem__(self, index):
        data = np.load(self.arrays[index])
        label = self.labels_numerical[index] if self.with_labels else []
        id_ = self.ids[index]

        if self.data_type == "packbits":
            data = np.unpackbits(data).reshape(self.target_height, -1)

        data = torch.from_numpy(data).unsqueeze(0)

        if self.target_width:
            right_padding = self.target_width - data.shape[2]
            data = pad(data, [0, right_padding])

        return data, label, id_

    def __len__(self):
        return len(self.arrays)
