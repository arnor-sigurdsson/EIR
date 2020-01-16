from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import joblib
import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.functional import pad
from torch.utils.data import Dataset

from human_origins_supervised.data_load.label_setup import (
    set_up_train_and_valid_labels,
    al_label_dict,
    get_transformer_path,
    get_target_transformer,
)
from .data_loading_funcs import make_random_snps_missing

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryArrayDataset", "DiskArrayDataset"]


def construct_dataset_init_params_from_cl_args(cl_args):
    """
    Shared between here and predict.py.
    """
    dataset_class_common_args = {
        "data_folder": cl_args.data_folder,
        "model_task": cl_args.model_task,
        "target_width": cl_args.target_width,
        "target_column": cl_args.target_column,
    }

    return dataset_class_common_args


def save_target_transformer(
    run_folder: Path,
    target_column: str,
    target_transformer: Union[StandardScaler, LabelEncoder],
) -> Path:
    """
    :param run_folder: Current run folder, used to anchor saving of transformer.
    :param target_column: The target column passed in for the current run.
    :param target_transformer: The transformer object to save.
    :return: Output path of where the target transformer was saved.
    """
    target_transformer_outpath = get_transformer_path(
        run_folder, target_column, "target_transformer"
    )
    ensure_path_exists(target_transformer_outpath)
    joblib.dump(target_transformer, target_transformer_outpath)

    return target_transformer_outpath


def set_up_datasets(cl_args: Namespace) -> Tuple[al_datasets, al_datasets]:
    """
    This function is only ever called if we have labels.
    """
    train_labels, valid_labels = set_up_train_and_valid_labels(cl_args)
    dataset_class_common_args = construct_dataset_init_params_from_cl_args(cl_args)

    dataset_class = MemoryArrayDataset if cl_args.memory_dataset else DiskArrayDataset
    train_dataset = dataset_class(
        **dataset_class_common_args,
        labels_dict=train_labels,
        na_augment=cl_args.na_augment,
    )
    valid_dataset = dataset_class(
        **dataset_class_common_args,
        labels_dict=valid_labels,
        target_transformer=train_dataset.target_transformer,
    )

    run_folder = Path("./runs", cl_args.run_name)
    save_target_transformer(
        run_folder, cl_args.target_column, train_dataset.target_transformer
    )

    assert len(train_dataset) > len(valid_dataset)
    assert set(valid_dataset.labels_dict.keys()).isdisjoint(
        train_dataset.labels_dict.keys()
    )

    return train_dataset, valid_dataset


@dataclass
class Sample:
    sample_id: str
    array: torch.Tensor
    labels: Union[Dict[str, str], float, None]


class ArrayDatasetBase(Dataset):
    def __init__(
        self,
        data_folder: Path,
        model_task: str,
        target_column: str = None,
        labels_dict: al_label_dict = None,
        target_transformer: Union[LabelEncoder, StandardScaler] = None,
        target_height: int = 4,
        target_width: int = None,
        na_augment: float = 0.0,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.model_task = model_task
        self.target_height = target_height
        self.target_width = target_width

        self.samples: Union[List[Sample], None] = None

        self.target_column = target_column
        self.labels_dict = labels_dict if labels_dict else {}
        self.labels_unique = None
        self.num_classes = None
        self.target_transformer = target_transformer

        self.na_augment = na_augment

    def parse_label(
        self, sample_label_dict: Dict[str, Union[str, float]]
    ) -> Union[List[None], np.ndarray, float]:
        """
        TODO:   Check if there is a significant performance hit doing the transform on
                the fly here versus doing one pass and loading.
        """

        if not sample_label_dict:
            return []

        label_value = sample_label_dict[self.target_column]
        tt_t = self.target_transformer.transform

        # StandardScaler() takes [[arr]] whereas LabelEncoder() takes [arr]
        label_value = [label_value] if self.model_task == "reg" else label_value
        label_value_trns = tt_t([label_value]).squeeze()

        return label_value_trns

    def get_samples(self, array_hook: Callable = lambda x: x):
        files = {i.stem: i for i in Path(self.data_folder).iterdir()}

        # When training or evaluating on test, labels_dict maps to train/val/test IDs.
        # When predicting on unknown test, then we don't have a labels dict.
        # We don't want to use `files` variable for train/val, as the self.samples
        # would have all obs. in both train/val, which is probably not a good idea.
        sample_id_iter = self.labels_dict if self.labels_dict else files
        samples = []

        for sample_id in sample_id_iter:
            cur_sample = Sample(
                sample_id=sample_id,
                array=array_hook(files.get(sample_id)),
                labels=self.labels_dict.get(sample_id, None),
            )
            samples.append(cur_sample)

        return samples

    @property
    def data_width(self):
        raise NotImplementedError

    def init_label_attributes(self):
        if not self.target_column:
            raise ValueError("Please specify label column name.")

        non_labelled = tuple(i for i in self.samples if not i.labels)
        if non_labelled:
            raise ValueError(
                f"Expected all observations to have a label associated "
                f"with them, but got {non_labelled}."
            )
        all_sample_target_labels = np.array(
            [i.labels[self.target_column] for i in self.samples]
        )

        if self.model_task == "cls":
            self.labels_unique = sorted(np.unique(all_sample_target_labels))
            self.num_classes = len(self.labels_unique)

        elif self.model_task == "reg":
            self.num_classes = 1
            # StandardScaler() expects 2D array, LabelEncoder() a 1D array
            all_sample_target_labels = all_sample_target_labels.reshape(-1, 1)

        if not self.target_transformer:
            target_transformer = get_target_transformer(self.model_task)
            target_transformer.fit(all_sample_target_labels)
            self.target_transformer = target_transformer


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

        array = array.astype(np.uint8)
        return torch.from_numpy(array).unsqueeze(0)

    @property
    def data_width(self):
        data = self.samples[0].array
        return data.shape[1]

    def __getitem__(self, index: int):
        sample = self.samples[index]

        array = sample.array
        label = self.parse_label(sample.labels)
        sample_id = sample.sample_id

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        if self.na_augment:
            array = make_random_snps_missing(array, self.na_augment)

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
        return data.shape[1]

    def __getitem__(self, index):
        sample = self.samples[index]

        array = np.load(sample.array)
        label = self.parse_label(sample.labels)
        sample_id = sample.sample_id

        array = torch.from_numpy(array).unsqueeze(0)
        if self.na_augment:
            array = make_random_snps_missing(array, self.na_augment)

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        return array, label, sample_id

    def __len__(self):
        return len(self.samples)
