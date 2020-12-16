from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable, Any

import numpy as np
import torch
from aislib.misc_utils import get_logger
from torch.utils.data import Dataset
from tqdm import tqdm

from snp_pred.data_load.data_augmentation import make_random_omics_columns_missing
from snp_pred.data_load.label_setup import (
    al_label_dict,
    al_target_columns,
    get_array_path_iterator,
    Labels,
)
from snp_pred.data_load.label_setup import merge_target_columns

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryDataset", "DiskDataset"]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_labels_transformed_all = Dict[str, Union[int, str, float]]
al_sample_label_dict_target = Dict[str, Union[int, float]]
al_sample_label_dict_extra = Dict[str, Union[str, float]]
al_all_labels = Dict[
    str, Union[al_sample_label_dict_target, al_sample_label_dict_extra]
]
al_inputs = Union[Dict[str, torch.Tensor], Dict[str, Any]]
al_getitem_return = Tuple[Dict[str, torch.Tensor], al_sample_label_dict_target, str]


def set_up_datasets(
    cl_args: Namespace,
    target_labels: Labels,
    tabular_inputs_labels: Union[Labels, None] = None,
) -> Tuple[al_datasets, al_datasets]:

    dataset_class_common_args = _construct_common_dataset_init_params(cl_args=cl_args)

    dataset_class = MemoryDataset if cl_args.memory_dataset else DiskDataset
    train_dataset = dataset_class(
        **dataset_class_common_args,
        target_labels_dict=target_labels.train_labels,
        tabular_inputs_labels_dict=tabular_inputs_labels.train_labels,
        na_augment=(cl_args.na_augment_perc, cl_args.na_augment_prob),
    )

    valid_dataset = dataset_class(
        **dataset_class_common_args,
        target_labels_dict=target_labels.valid_labels,
        tabular_inputs_labels_dict=tabular_inputs_labels.valid_labels,
    )

    _check_valid_and_train_datasets(
        train_dataset=train_dataset, valid_dataset=valid_dataset
    )

    return train_dataset, valid_dataset


def _construct_common_dataset_init_params(cl_args: Namespace) -> Dict:
    target_columns = merge_target_columns(
        target_con_columns=cl_args.target_con_columns,
        target_cat_columns=cl_args.target_cat_columns,
    )

    dataset_class_common_args = {
        "target_columns": target_columns,
        "data_source": cl_args.data_source,
    }

    return dataset_class_common_args


def _check_valid_and_train_datasets(
    train_dataset: al_datasets, valid_dataset: al_datasets
) -> None:
    assert len(train_dataset) > len(valid_dataset)
    assert set(valid_dataset.target_labels_dict.keys()).isdisjoint(
        train_dataset.target_labels_dict.keys()
    )


@dataclass
class Sample:
    """
    array: can be path to array or the loaded array itself
    """

    sample_id: str
    inputs: Dict[str, Any]
    target_labels: al_sample_label_dict_target


# TODO: Update to work with data_sources, Dict[str, Path], use prefixes, e.g. omics_
#       We can also add tabular_inputs_labels_dict in there, with tabular_ prefix
#       Then we have assumptions, e.g. omics is always on disk, tabular is always a dict

# TODO: Add in support for missing input modalities
class DatasetBase(Dataset):
    def __init__(
        self,
        data_source: Path,
        target_columns: al_target_columns,
        target_labels_dict: al_label_dict = None,
        tabular_inputs_labels_dict: al_label_dict = None,
        na_augment: Tuple[float] = (0.0, 0.0),
    ):
        super().__init__()

        self.data_source = data_source

        self.samples: Union[List[Sample], None] = None

        self.target_columns = target_columns
        self.target_labels_dict = target_labels_dict if target_labels_dict else {}
        self.extra_tabular_labels_dict = (
            tabular_inputs_labels_dict if tabular_inputs_labels_dict else {}
        )

        self.na_augment = na_augment

    def init_label_attributes(self):
        if not self.target_columns:
            raise ValueError("Please specify label column name.")

    def set_up_samples(self, array_hook: Callable = lambda x: x) -> List[Sample]:
        """
        If we have initialized a labels_dict variable, we use that to iterate over IDs.
        This is for (a) training, (b) evaluating or (c) testing on test set for
        generalization error.

        When predicting on unknown set (i.e. no labels), then we don't have a
        labels dict, hence we refer to `files directly`. The reason why we use an
        empty dictionary by default is that the default collate function will raise
        an error if we return None.

        We don't want to use `files` variable for train/val, as the self.samples
        would have all obs. in both train/val, which is probably not a good idea as
        it might pose data leakage risks.
        """

        logger.debug("Setting up samples in current dataset.")
        path_iterator = get_array_path_iterator(data_source=Path(self.data_source))
        files = {i.stem: i for i in path_iterator}

        sample_id_iter = self.target_labels_dict if self.target_labels_dict else files
        samples = []

        for sample_id in tqdm(sample_id_iter, desc="Progress"):
            sample_target_labels = self.target_labels_dict.get(sample_id, {})

            sample_array_input = array_hook(files.get(sample_id))

            sample_inputs = {
                "omics_cl_args": sample_array_input,
            }
            if self.extra_tabular_labels_dict:
                sample_tabular_inputs = self.extra_tabular_labels_dict.get(
                    sample_id, {}
                )
                sample_inputs["tabular_cl_args"] = sample_tabular_inputs

            cur_sample = Sample(
                sample_id=sample_id,
                inputs=sample_inputs,
                target_labels=sample_target_labels,
            )
            samples.append(cur_sample)

        return samples

    @property
    def data_width(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def check_non_labelled(self):
        if not self.target_labels_dict:
            return

        non_labelled = tuple(i for i in self.samples if not i.target_labels)
        if non_labelled:
            raise ValueError(
                f"Expected all observations to have a label associated "
                f"with them, but got {non_labelled}."
            )


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples(array_hook=self._mem_sample_loader)
        self.check_non_labelled()

    @staticmethod
    def _mem_sample_loader(sample_fpath: Union[str, Path]) -> torch.ByteTensor:
        """
        A small hook to actually load the arrays into `self.samples` instead of just
        pointing to filenames.
        """
        array = np.load(sample_fpath)

        array = array.astype(np.uint8)
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    @property
    def data_width(self):
        data = self.samples[0].inputs["omics_cl_args"]
        return data.shape[1]

    # Note that dataloaders automatically convert arrays to tensors here, for labels
    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        inputs_prepared = copy(sample.inputs)

        genotype_array_raw = sample.inputs["omics_cl_args"]
        genotype_array_prepared = prepare_one_hot_omics_data(
            genotype_array=genotype_array_raw,
            na_augment_perc=self.na_augment[0],
            na_augment_prob=self.na_augment[1],
        )
        inputs_prepared["omics_cl_args"] = genotype_array_prepared

        target_labels = sample.target_labels
        sample_id = sample.sample_id

        return inputs_prepared, target_labels, sample_id

    def __len__(self):
        return len(self.samples)


def prepare_one_hot_omics_data(
    genotype_array: torch.Tensor,
    na_augment_perc: float,
    na_augment_prob: float,
) -> torch.BoolTensor:
    array_bool = genotype_array.to(dtype=torch.bool)

    if na_augment_perc > 0 and na_augment_prob > 0:
        array_bool = make_random_omics_columns_missing(
            omics_array=array_bool,
            percentage=na_augment_perc,
            probability=na_augment_prob,
        )

    return array_bool


class DiskDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples()
        self.check_non_labelled()

    @property
    def data_width(self):
        data = np.load(self.samples[0].inputs["omics_cl_args"])
        return data.shape[1]

    # Note that dataloaders automatically convert arrays to tensors here, for labels
    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        inputs_prepared = copy(sample.inputs)

        # TODO: Refactor np.load --> torch and reuse in memory dataset
        genotype_array_raw = np.load(sample.inputs["omics_cl_args"])
        genotype_array_raw = torch.from_numpy(genotype_array_raw).unsqueeze(0)

        genotype_array_prepared = prepare_one_hot_omics_data(
            genotype_array=genotype_array_raw,
            na_augment_perc=self.na_augment[0],
            na_augment_prob=self.na_augment[1],
        )
        inputs_prepared["omics_cl_args"] = genotype_array_prepared

        target_labels = sample.target_labels
        sample_id = sample.sample_id

        return inputs_prepared, target_labels, sample_id

    def __len__(self):
        return len(self.samples)


def _load_one_hot_array():
    pass
