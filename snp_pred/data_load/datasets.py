from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple, Callable

import joblib
import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.preprocessing import StandardScaler
from snp_pred.data_load.data_augmentation import make_random_snps_missing
from snp_pred.data_load.label_setup import (
    set_up_label_transformers,
    _streamline_values_for_transformers,
)
from snp_pred.data_load.label_setup import (
    set_up_train_and_valid_labels,
    al_label_dict,
    al_label_values_raw,
    al_sample_labels_raw,
    al_target_columns,
    al_label_transformers_object,
    al_label_transformers,
    get_transformer_path,
    get_array_path_iterator,
)
from snp_pred.train_utils.utils import get_run_folder
from torch.nn.functional import pad
from torch.utils.data import Dataset
from tqdm import tqdm

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryArrayDataset", "DiskArrayDataset"]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_labels_transformed_all = Dict[str, Union[int, str, float]]
al_sample_label_dict_target = Dict[str, Union[int, float]]
al_sample_label_dict_extra = Dict[str, Union[str, float]]
al_all_labels = Dict[
    str, Union[al_sample_label_dict_target, al_sample_label_dict_extra]
]
al_getitem_return = Tuple[torch.Tensor, al_all_labels, str]

al_num_classes = Dict[str, int]


def set_up_datasets(
    cl_args: Namespace, custom_label_ops: Union[None, Callable]
) -> Tuple[al_datasets, al_datasets]:
    """
    This function is only ever called if we have labels.
    """
    train_labels, valid_labels = set_up_train_and_valid_labels(
        cl_args=cl_args, custom_label_ops=custom_label_ops
    )

    dataset_class_common_args = _construct_common_dataset_init_params(
        cl_args=cl_args, train_labels=train_labels
    )

    dataset_class = MemoryArrayDataset if cl_args.memory_dataset else DiskArrayDataset
    train_dataset = dataset_class(
        **dataset_class_common_args,
        labels_dict=train_labels,
        na_augment=(cl_args.na_augment_perc, cl_args.na_augment_prob),
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

    contn_columns_dict = {"extra_con": cl_args.extra_con_columns}
    extra_con_transformers_fit_on_train = set_up_label_transformers(
        labels_dict=train_labels, label_columns=contn_columns_dict
    )

    dataset_class_common_args = {
        "data_source": cl_args.data_source,
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


def _save_transformer_set(transformers: al_label_transformers, run_name: str) -> None:
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
    target_transformer_object: al_label_transformers_object,
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
    """
    array: can be path to array or the loaded array itself
    """

    sample_id: str
    array: Union[str, torch.Tensor]
    labels: al_all_labels


class ArrayDatasetBase(Dataset):
    def __init__(
        self,
        data_source: Path,
        target_columns: al_target_columns,
        labels_dict: al_label_dict = None,
        target_transformers: al_label_transformers = None,
        extra_con_transformers: Dict[str, StandardScaler] = None,
        target_height: int = 4,
        target_width: int = None,
        na_augment: Tuple[float] = (0.0, 0.0),
    ):
        super().__init__()

        self.data_source = data_source
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

        logger.debug("Setting up samples in current dataset.")
        path_iterator = get_array_path_iterator(data_source=Path(self.data_source))
        files = {i.stem: i for i in path_iterator}

        sample_id_iter = self.labels_dict if self.labels_dict else files
        samples = []

        for sample_id in tqdm(sample_id_iter, desc="Progress"):
            raw_sample_labels = self.labels_dict.get(sample_id, None)
            parsed_sample_labels = _transform_labels_in_sample(
                target_transformers=self.target_transformers,
                extra_con_transformers=self.extra_con_transformers,
                sample_labels_raw_dict=raw_sample_labels,
            )

            sample_labels = _split_labels_into_target_and_extra(
                sample_labels=parsed_sample_labels, target_columns=self.target_columns
            )

            cur_sample = Sample(
                sample_id=sample_id,
                array=array_hook(files.get(sample_id)),
                labels=sample_labels,
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
    target_transformers: al_label_transformers,
    sample_labels_raw_dict: al_sample_labels_raw,
    extra_con_transformers: Union[Dict[str, StandardScaler], None] = None,
) -> al_sample_labels_transformed_all:
    """
    We transform the target and extra continuous labels only because for:

        - extra embed columns: We use the set up embedding dictionary attached to the
          model itself. Since the embeddings are trainable.
    """

    transformed_labels = {}

    if not extra_con_transformers:
        extra_con_transformers = {}

    merged_transformers = {**target_transformers, **extra_con_transformers}

    for label_column, label_value in sample_labels_raw_dict.items():

        if label_column in merged_transformers.keys():

            transformer = merged_transformers[label_column]
            cur_label_parsed = _transform_single_label_value(
                transformer=transformer, label_value=label_value
            )
            transformed_labels[label_column] = cur_label_parsed.item()

        else:
            transformed_labels[label_column] = label_value

    return transformed_labels


def _transform_single_label_value(
    transformer, label_value: al_label_values_raw
) -> np.ndarray:
    tt_t = transformer.transform
    label_value = np.array([label_value])

    label_value_streamlined = _streamline_values_for_transformers(
        transformer=transformer, values=label_value
    )

    label_value_transformed = tt_t(label_value_streamlined).squeeze()

    return label_value_transformed


def _split_labels_into_target_and_extra(
    sample_labels: al_sample_labels_transformed_all, target_columns: al_target_columns
) -> al_all_labels:

    target_columns_flat = target_columns["con"] + target_columns["cat"]
    target_labels = {k: v for k, v in sample_labels.items() if k in target_columns_flat}
    extra_labels = {
        k: v for k, v in sample_labels.items() if k not in target_columns_flat
    }

    split_labels_dict = {"target_labels": target_labels, "extra_labels": extra_labels}

    return split_labels_dict


def _set_up_num_classes(target_transformers: al_label_transformers) -> al_num_classes:

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
        data = self.samples[0].array
        return data.shape[1]

    # Note that dataloaders automatically convert arrays to tensors here, for labels
    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        array = sample.array.to(dtype=torch.bool)
        labels = sample.labels
        sample_id = sample.sample_id

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        if self.na_augment[0]:
            array = make_random_snps_missing(
                array=array,
                percentage=self.na_augment[0],
                probability=self.na_augment[1],
            )
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
    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        array = np.load(sample.array)
        labels = sample.labels
        sample_id = sample.sample_id

        array = torch.from_numpy(array).unsqueeze(0).to(dtype=torch.bool)
        if self.na_augment[0]:
            array = make_random_snps_missing(
                array=array,
                percentage=self.na_augment[0],
                probability=self.na_augment[1],
            )

        if self.target_width:
            right_padding = self.target_width - array.shape[2]
            array = pad(array, [0, right_padding])

        return array, labels, sample_id

    def __len__(self):
        return len(self.samples)
