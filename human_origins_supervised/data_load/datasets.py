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
)
from .data_loading_funcs import make_random_snps_missing

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryArrayDataset", "DiskArrayDataset"]
al_target_columns = Dict[str, List[str]]
al_target_transformers = Union[StandardScaler, LabelEncoder]
al_sample_label_dict = Dict[str, Union[str, float]]
al_label_value = Union[List[None], np.ndarray, float]


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
        target_transformers=train_dataset.target_transformers,
    )

    run_folder = Path("./runs", cl_args.run_name)
    target_transformers = train_dataset.target_transformers
    for (transformer_name, target_transformer) in target_transformers.items():
        save_target_transformer(
            run_folder=run_folder,
            transformer_name=transformer_name,
            target_transformer=target_transformer,
        )

    assert len(train_dataset) > len(valid_dataset)
    assert set(valid_dataset.labels_dict.keys()).isdisjoint(
        train_dataset.labels_dict.keys()
    )

    return train_dataset, valid_dataset


def construct_dataset_init_params_from_cl_args(cl_args):
    """
    Shared between here and predict.py.
    """
    target_columns = merge_target_columns(
        target_con_columns=cl_args.target_con_columns,
        target_cat_columns=cl_args.target_cat_columns,
    )
    dataset_class_common_args = {
        "data_folder": cl_args.data_folder,
        "target_width": cl_args.target_width,
        "target_columns": target_columns,
    }

    return dataset_class_common_args


def merge_target_columns(
    target_con_columns: List[str], target_cat_columns: List[str]
) -> Dict[str, List[str]]:

    if len(target_con_columns + target_cat_columns) == 0:
        raise ValueError("Expected at least 1 target column")

    all_target_columns = {"con": [], "cat": []}
    [all_target_columns["con"].append(i) for i in target_con_columns]
    [all_target_columns["cat"].append(i) for i in target_cat_columns]

    assert len(all_target_columns) > 0

    return all_target_columns


def save_target_transformer(
    run_folder: Path, transformer_name: str, target_transformer: al_target_transformers
) -> Path:
    """
    :param run_folder: Current run folder, used to anchor saving of transformer.
    :param transformer_name: The target column passed in for the current run.
    :param target_transformer: The transformer object to save.
    :return: Output path of where the target transformer was saved.
    """
    target_transformer_outpath = get_transformer_path(
        run_path=run_folder,
        transformer_name=transformer_name,
        suffix="target_transformer",
    )
    ensure_path_exists(target_transformer_outpath)
    joblib.dump(target_transformer, target_transformer_outpath)

    return target_transformer_outpath


@dataclass
class Sample:
    sample_id: str
    array: Union[str, torch.Tensor]
    labels: Union[Dict[str, str], float, None]


class ArrayDatasetBase(Dataset):
    def __init__(
        self,
        data_folder: Path,
        target_columns: al_target_columns,
        labels_dict: al_label_dict = None,
        target_transformers: Dict[str, al_target_transformers] = None,
        target_height: int = 4,
        target_width: int = None,
        na_augment: float = 0.0,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.target_height = target_height
        self.target_width = target_width

        self.samples: Union[List[Sample], None] = None

        self.target_columns = target_columns
        self.labels_dict = labels_dict if labels_dict else {}
        self.target_transformers = target_transformers
        self.num_classes = None

        self.na_augment = na_augment

    def init_label_attributes(self):
        if not self.target_columns:
            raise ValueError("Please specify label column name.")

        if not self.target_transformers:
            self.target_transformers = set_up_all_target_transformers(
                self.labels_dict, self.target_columns
            )

        # TODO: Rename to be more descriptive, dict now instead of int.
        self.num_classes = set_up_num_classes(self.target_transformers)

    def set_up_samples(self, array_hook: Callable = lambda x: x) -> List[Sample]:
        """
        If we have initialized a labels_dict variable, we use that to iterate over IDs.
        This is for (a) training, (b) evaluating or (c) testing on test set for
        generalization error.

        When predicting on unknown set (i.e. no labels), then we don't have a
        labels dict, hence we refer to `files directly`.

        We don't want to use `files` variable for train/val, as the self.samples
        would have all obs. in both train/val, which is probably not a good idea as
        it might pose data leakage risks.
        """
        files = {i.stem: i for i in Path(self.data_folder).iterdir()}

        sample_id_iter = self.labels_dict if self.labels_dict else files
        samples = []

        for sample_id in sample_id_iter:
            raw_sample_labels = self.labels_dict.get(sample_id, None)
            parsed_sample_labels = transform_sample_labels(
                target_transformers=self.target_transformers,
                sample_label_dict=raw_sample_labels,
            )

            cur_sample = Sample(
                sample_id=sample_id,
                array=array_hook(files.get(sample_id)),
                labels=parsed_sample_labels,
            )
            samples.append(cur_sample)

        return samples

    @property
    def data_width(self):
        raise NotImplementedError

    def check_non_labelled(self):
        non_labelled = tuple(i for i in self.samples if not i.labels)
        if non_labelled:
            raise ValueError(
                f"Expected all observations to have a label associated "
                f"with them, but got {non_labelled}."
            )


def set_up_all_target_transformers(
    labels_dict: al_label_dict, target_columns: al_target_columns
) -> Dict[str, al_target_transformers]:

    target_transformers = {}
    for column_type in target_columns:
        logger.debug(
            "Fitting transformers on %s target columns %s", column_type, target_columns
        )

        target_columns_of_cur_type = target_columns[column_type]

        for cur_target_column in target_columns_of_cur_type:
            cur_target_transformer = fit_transformer_on_target_column(
                labels_dict=labels_dict,
                target_column=cur_target_column,
                column_type=column_type,
            )
            target_transformers[cur_target_column] = cur_target_transformer

    return target_transformers


def fit_transformer_on_target_column(
    labels_dict: al_label_dict, target_column, column_type: str
) -> al_target_transformers:
    """
    TODO: Maybe make this more efficient by just getting unique values in target values?
    """

    transformer = get_target_transformer(column_type)

    target_values = list((i[target_column] for i in labels_dict.values()))
    transformer.fit(target_values)

    return transformer


def transform_sample_labels(
    target_transformers: Dict[str, al_target_transformers],
    sample_label_dict: al_sample_label_dict,
):

    transformed_labels = {}
    for label_column, label_value in sample_label_dict.items():
        transformer = target_transformers[label_column]
        cur_label_parsed = transform_label_value(transformer, label_value)
        transformed_labels[label_column] = cur_label_parsed.item()

    return transformed_labels


def transform_label_value(transformer, label_value):
    tt_t = transformer.transform
    if isinstance(transformer, StandardScaler):
        label_value = [label_value]

    label_value_transformed = tt_t([label_value]).squeeze()

    return label_value_transformed


def set_up_num_classes(
    target_transformers: Dict[str, al_target_transformers]
) -> Dict[str, int]:

    num_classes_dict = {}
    for target_column, transformer in target_transformers.items():
        if isinstance(transformer, StandardScaler):
            num_classes = 1
        else:
            num_classes = len(transformer.classes_)
        num_classes_dict[target_column] = num_classes

    return num_classes_dict


class MemoryArrayDataset(ArrayDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples(array_hook=self.mem_sample_loader)
        self.check_non_labelled()

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
        labels = sample.labels
        sample_id = sample.sample_id

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        if self.na_augment:
            array = make_random_snps_missing(array, self.na_augment)

        return array, labels, sample_id

    def __len__(self):
        return len(self.samples)


class DiskArrayDataset(ArrayDatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples()
        self.check_non_labelled()

    @property
    def data_width(self):
        data = np.load(self.samples[0].array)
        return data.shape[1]

    def __getitem__(self, index):
        sample = self.samples[index]

        array = np.load(sample.array)
        labels = sample.labels
        sample_id = sample.sample_id

        array = torch.from_numpy(array).unsqueeze(0)
        if self.na_augment:
            array = make_random_snps_missing(array, self.na_augment)

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        return array, labels, sample_id

    def __len__(self):
        return len(self.samples)


def get_target_transformer(column_type):
    if column_type == "con":
        return StandardScaler()
    elif column_type == "cat":
        return LabelEncoder()

    raise ValueError()
