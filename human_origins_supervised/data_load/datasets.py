from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import joblib
import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import pad
from torch.utils.data import Dataset

from human_origins_supervised.data_load.label_setup import (
    set_up_train_and_valid_labels,
    al_label_dict,
    al_target_columns,
    al_label_transformers,
    get_transformer_path,
)
from human_origins_supervised.data_load.data_loading_funcs import (
    make_random_snps_missing,
)
from human_origins_supervised.train_utils.utils import get_run_folder
from human_origins_supervised.data_load.label_setup import (
    set_up_label_transformers,
    _streamline_values_for_transformers,
)

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryArrayDataset", "DiskArrayDataset"]
al_sample_label_dict = Dict[str, Union[str, float]]
al_label_value = Union[str, float, int]
al_num_classes = Dict[str, int]


def set_up_datasets(cl_args: Namespace) -> Tuple[al_datasets, al_datasets]:
    """
    This function is only ever called if we have labels.
    """
    train_labels, valid_labels = set_up_train_and_valid_labels(cl_args)

    dataset_class_common_args = _construct_common_dataset_init_params(
        cl_args=cl_args, train_labels=train_labels
    )

    dataset_class = MemoryArrayDataset if cl_args.memory_dataset else DiskArrayDataset
    train_dataset = dataset_class(
        **dataset_class_common_args,
        labels_dict=train_labels,
        na_augment=cl_args.na_augment,
    )

    valid_dataset = dataset_class(**dataset_class_common_args, labels_dict=valid_labels)

    all_transformers = {
        **train_dataset.target_transformers,
        **train_dataset.extra_con_transformers,
    }
    _save_transformer_set(transformers=all_transformers, run_name=cl_args.run_name)

    _check_valid_and_train_datasets(
        train_dataset=train_dataset, valid_dataset=valid_dataset
    )

    return train_dataset, valid_dataset


def _construct_common_dataset_init_params(
    cl_args: Namespace, train_labels: al_label_dict
) -> Dict:
    """
    We do not use extra embed columns here because they do not have a transformer
    associated with them.
    """
    target_columns = merge_target_columns(
        target_con_columns=cl_args.target_con_columns,
        target_cat_columns=cl_args.target_cat_columns,
    )

    target_transformers_fit_on_train = set_up_label_transformers(
        labels_dict=train_labels, label_columns=target_columns
    )

    contn_columns_dict = {"extra_con": cl_args.contn_columns}
    extra_con_transformers_fit_on_train = set_up_label_transformers(
        labels_dict=train_labels, label_columns=contn_columns_dict
    )

    dataset_class_common_args = {
        "data_folder": cl_args.data_folder,
        "target_width": cl_args.target_width,
        "target_columns": target_columns,
        "target_transformers": target_transformers_fit_on_train,
        "extra_con_transformers": extra_con_transformers_fit_on_train,
    }

    return dataset_class_common_args


def merge_target_columns(
    target_con_columns: List[str], target_cat_columns: List[str]
) -> al_target_columns:

    if len(target_con_columns + target_cat_columns) == 0:
        raise ValueError("Expected at least 1 label column")

    all_target_columns = {"con": [], "cat": []}

    [all_target_columns["con"].append(i) for i in target_con_columns]
    [all_target_columns["cat"].append(i) for i in target_cat_columns]

    assert len(all_target_columns) > 0

    return all_target_columns


def _save_transformer_set(
    transformers: Dict[str, al_label_transformers], run_name: str
) -> None:
    run_folder = get_run_folder(run_name)

    for (transformer_name, transformer_object) in transformers.items():
        save_label_transformer(
            run_folder=run_folder,
            transformer_name=transformer_name,
            target_transformer_object=transformer_object,
        )


def save_label_transformer(
    run_folder: Path,
    transformer_name: str,
    target_transformer_object: al_label_transformers,
) -> Path:
    target_transformer_outpath = get_transformer_path(
        run_path=run_folder, transformer_name=transformer_name
    )
    ensure_path_exists(target_transformer_outpath)
    joblib.dump(value=target_transformer_object, filename=target_transformer_outpath)

    return target_transformer_outpath


def _check_valid_and_train_datasets(
    train_dataset: al_datasets, valid_dataset: al_datasets
) -> None:
    assert len(train_dataset) > len(valid_dataset)
    assert set(valid_dataset.labels_dict.keys()).isdisjoint(
        train_dataset.labels_dict.keys()
    )


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
        target_transformers: Dict[str, al_label_transformers] = None,
        extra_con_transformers: Dict[str, StandardScaler] = None,
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
        self.extra_con_transformers = extra_con_transformers
        self.num_classes = None

        self.na_augment = na_augment

    def init_label_attributes(self):
        if not self.target_columns:
            raise ValueError("Please specify label column name.")

        # TODO: Rename to be more descriptive, dict now instead of int.
        self.num_classes = _set_up_num_classes(self.target_transformers)

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
            parsed_sample_labels = _transform_labels_in_sample(
                target_transformers=self.target_transformers,
                extra_con_transformers=self.extra_con_transformers,
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


def _transform_labels_in_sample(
    target_transformers: Dict[str, al_label_transformers],
    sample_label_dict: al_sample_label_dict,
    extra_con_transformers: Union[Dict[str, StandardScaler], None] = None,
):
    """
    We transform the target and extra continuous labels only because for:

        - extra embed columns: We use the set up embedding dictionary.
    """

    transformed_labels = {}

    if not extra_con_transformers:
        extra_con_transformers = {}

    merged_transformers = {**target_transformers, **extra_con_transformers}

    for label_column, label_value in sample_label_dict.items():

        if label_column in merged_transformers.keys():

            transformer = merged_transformers[label_column]
            cur_label_parsed = _transform_single_label_value(
                transformer=transformer, label_value=label_value
            )
            transformed_labels[label_column] = cur_label_parsed.item()

    return transformed_labels


def _transform_single_label_value(transformer, label_value: Union[str, float, int]):
    tt_t = transformer.transform
    label_value = np.array([label_value])

    label_value_streamlined = _streamline_values_for_transformers(
        transformer=transformer, values=label_value
    )

    label_value_transformed = tt_t(label_value_streamlined).squeeze()

    return label_value_transformed


def _set_up_num_classes(
    target_transformers: Dict[str, al_label_transformers]
) -> al_num_classes:

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

    # Note that dataloaders automatically convert arrays to tensors here, for labels
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

    # Note that dataloaders automatically convert arrays to tensors here, for labels
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
