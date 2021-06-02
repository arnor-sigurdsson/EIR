from argparse import Namespace
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    List,
    Dict,
    Union,
    Tuple,
    Callable,
    Iterable,
    Sequence,
    Generator,
    DefaultDict,
    Any,
    TYPE_CHECKING,
)

import numpy as np
import torch
from aislib.misc_utils import get_logger
from torch.utils.data import Dataset
from tqdm import tqdm

from eir.data_load.data_augmentation import make_random_omics_columns_missing
from eir.data_load.label_setup import (
    al_label_dict,
    al_target_columns,
    get_array_path_iterator,
    Labels,
)
from eir.data_load.label_setup import merge_target_columns

if TYPE_CHECKING:
    from eir.train import DataDimensions

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


def set_up_datasets_from_cl_args(
    cl_args: Namespace,
    target_labels: Labels,
    data_dimensions: Dict[str, "DataDimensions"],
    tabular_inputs_labels: Union[Labels, None] = None,
) -> Tuple[al_datasets, al_datasets]:

    dataset_class = MemoryDataset if cl_args.memory_dataset else DiskDataset

    train_kwargs = construct_default_dataset_kwargs_from_cl_args(
        cl_args=cl_args,
        target_labels_dict=target_labels.train_labels,
        data_dimensions=data_dimensions,
        tabular_labels_dict=tabular_inputs_labels.train_labels,
        na_augment=True,
    )

    valid_kwargs = construct_default_dataset_kwargs_from_cl_args(
        cl_args=cl_args,
        target_labels_dict=target_labels.valid_labels,
        data_dimensions=data_dimensions,
        tabular_labels_dict=tabular_inputs_labels.valid_labels,
        na_augment=False,
    )

    train_dataset = dataset_class(**train_kwargs)
    valid_dataset = dataset_class(**valid_kwargs)

    _check_valid_and_train_datasets(
        train_dataset=train_dataset, valid_dataset=valid_dataset
    )

    return train_dataset, valid_dataset


def construct_default_dataset_kwargs_from_cl_args(
    cl_args: Namespace,
    target_labels_dict: Union[None, al_label_dict],
    data_dimensions: Dict[str, "DataDimensions"],
    tabular_labels_dict: Union[None, al_label_dict],
    na_augment: bool,
) -> Dict[str, Any]:

    target_columns = merge_target_columns(
        target_con_columns=cl_args.target_con_columns,
        target_cat_columns=cl_args.target_cat_columns,
    )

    data_sources = gather_all_data_sources(
        cl_args=cl_args, tabular_labels_dict=tabular_labels_dict
    )

    dataset_kwargs = {
        "target_columns": target_columns,
        "data_sources": data_sources,
        "data_dimensions": data_dimensions,
        "target_labels_dict": target_labels_dict,
    }

    if na_augment:
        dataset_kwargs["na_augment"] = (
            cl_args.na_augment_perc,
            cl_args.na_augment_prob,
        )

    return dataset_kwargs


def gather_all_data_sources(
    cl_args: Namespace, tabular_labels_dict: Union[Labels, None]
) -> Dict[str, str]:
    all_sources = {}

    if cl_args.omics_sources and cl_args.omics_names:
        for source_on_disk, name in zip(cl_args.omics_sources, cl_args.omics_names):
            all_sources[name] = source_on_disk

    if tabular_labels_dict:
        all_sources["tabular_cl_args"] = tabular_labels_dict

    return all_sources


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


class DatasetBase(Dataset):
    def __init__(
        self,
        data_sources: Dict[str, Any],
        data_dimensions: Dict,
        target_columns: al_target_columns,
        target_labels_dict: al_label_dict = None,
        na_augment: Tuple[float] = (0.0, 0.0),
    ):
        super().__init__()

        self.data_sources = data_sources
        self.data_dimensions = data_dimensions

        self.samples: Union[List[Sample], None] = None

        self.target_columns = target_columns
        self.target_labels_dict = target_labels_dict if target_labels_dict else {}

        self.na_augment = na_augment

    def init_label_attributes(self):
        if not self.target_columns:
            raise ValueError("Please specify label column name.")

    def set_up_samples(self, omics_hook: Callable = lambda x: x) -> List[Sample]:
        """
        We do an extra filtering step at the end to account for the situation where
        we have a target label file with more samples than there are any inputs
        available for. This is quite likely if we have e.g. pre-split data into
        train/val and test folders.
        """

        def _default_factory() -> Sample:
            return Sample(sample_id="", inputs={}, target_labels={})

        samples = defaultdict(_default_factory)

        ids_to_keep = None
        if self.target_labels_dict:
            samples = _add_target_labels_to_samples(
                target_labels_dict=self.target_labels_dict, samples=samples
            )
            ids_to_keep = self.target_labels_dict.keys()

        for source_name, source_data in self.data_sources.items():

            if source_name.startswith("omics_"):

                samples = _add_file_data_to_samples(
                    source_data=source_data,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    file_loading_hook=omics_hook,
                    source_name=source_name,
                )

            elif source_name.startswith("tabular_"):
                samples = _add_tabular_data_to_samples(
                    tabular_dict=source_data,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    source_name=source_name,
                )

        if self.target_labels_dict:
            num_samples_raw = len(samples)
            samples = list(i for i in samples.values() if i.inputs and i.target_labels)
            num_missing = num_samples_raw - len(samples)
            logger.debug(
                "Filtered out %d samples that had no inputs or no target labels.",
                num_missing,
            )

        return samples

    def __getitem__(self, index: int):
        raise NotImplementedError

    def check_samples(self):

        no_ids, no_inputs, no_target_labels = [], [], []

        for s in self.samples:
            if not s.sample_id:
                no_ids.append(s)

            if not s.inputs:
                no_inputs.append(s)

            if self.target_labels_dict:
                if not s.target_labels:
                    no_target_labels.append(s)

        if no_ids:
            raise ValueError(
                f"Expected all observations to have a sample ID associated "
                f"with them, but got {no_ids}."
            )

        if no_inputs:
            raise ValueError(
                f"Expected all observations to have an input associated "
                f"with them, but got {no_inputs}."
            )

        if self.target_labels_dict:
            if no_target_labels:
                raise ValueError(
                    f"Expected all observations to have a label associated "
                    f"with them, but got {no_target_labels}."
                )


def _add_target_labels_to_samples(
    target_labels_dict: al_label_dict, samples: DefaultDict[str, Sample]
) -> DefaultDict[str, Sample]:
    target_label_iterator = tqdm(target_labels_dict.items(), desc="Target Labels")

    for sample_id, sample_target_labels in target_label_iterator:
        _add_id_to_samples(samples=samples, sample_id=sample_id)
        samples[sample_id].target_labels = sample_target_labels

    return samples


def _add_file_data_to_samples(
    source_data: str,
    samples: DefaultDict[str, Sample],
    ids_to_keep: Union[None, Sequence[str]],
    file_loading_hook: Callable,
    source_name: str = "File Data",
) -> DefaultDict[str, Sample]:

    file_data_iterator = get_file_sample_id_iterator_basic(
        data_source=source_data, ids_to_keep=ids_to_keep
    )
    omics_file_iterator_tqdm = tqdm(file_data_iterator, desc=source_name)

    for sample_id, file in omics_file_iterator_tqdm:

        sample_data = file_loading_hook(file)

        samples = _add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[source_name] = sample_data

    return samples


def _add_tabular_data_to_samples(
    tabular_dict: al_label_dict,
    samples: DefaultDict[str, Sample],
    ids_to_keep: Union[None, Sequence[str]],
    source_name: str = "Tabular Data",
) -> DefaultDict[str, Sample]:

    tabular_iterator = tqdm(tabular_dict.items(), desc=source_name)

    for sample_id, tabular_inputs in tabular_iterator:

        if sample_id not in ids_to_keep:
            continue

        samples = _add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[source_name] = tabular_inputs

    return samples


def _add_id_to_samples(
    samples: DefaultDict[str, Sample], sample_id: str
) -> DefaultDict[str, Sample]:
    """
    This kind of weird function is used because in some cases, we cannot expect the
    target labels to have added samples, because we could be predicting on completely
    unknown samples without any target label data.

    Hence, we might have sparse modular data available for the samples, e.g. only omics
    for some samples, but only tabular data for others. So we want to ensure that the
    data is filled in.
    """
    if not samples[sample_id].sample_id:
        samples[sample_id].sample_id = sample_id
    else:
        assert samples[sample_id].sample_id == sample_id

    return samples


def get_file_sample_id_iterator_basic(
    data_source: str,
    ids_to_keep: Union[None, Sequence[str]],
) -> Generator[Tuple[Any, str], None, None]:

    base_file_iterator = get_array_path_iterator(
        data_source=Path(data_source), validate=False
    )

    for file in base_file_iterator:
        sample_id = file.stem

        if ids_to_keep:
            if sample_id in ids_to_keep:
                yield sample_id, file
        else:
            yield sample_id, file


def get_file_sample_id_iterator(
    data_source: str, ids_to_keep: Union[None, Sequence[str]]
) -> Generator[Tuple[Any, str], None, None]:
    def _id_from_filename(file: Path) -> str:
        return file.stem

    def _filter_ids_callable(item, sample_id):
        if sample_id in ids_to_keep:
            return True
        return False

    base_file_iterator = get_array_path_iterator(
        data_source=Path(data_source), validate=False
    )

    sample_id_and_file_iterator = _get_sample_id_data_iterator(
        base_iterator=base_file_iterator, id_callable=_id_from_filename
    )

    if ids_to_keep:
        final_iterator = _get_filter_iterator(
            base_iterator=sample_id_and_file_iterator,
            filter_callable=_filter_ids_callable,
        )
    else:
        final_iterator = sample_id_and_file_iterator

    yield from final_iterator


def _get_sample_id_data_iterator(
    base_iterator, id_callable: Callable
) -> Generator[Tuple[Any, str], None, None]:
    for item in base_iterator:
        sample_id = id_callable(item)
        yield item, sample_id


def _get_filter_iterator(base_iterator, filter_callable) -> Generator[Any, None, None]:
    for item in base_iterator:
        if filter_callable(*item):
            yield item


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples(omics_hook=self._mem_sample_loader)
        self.check_samples()

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

    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        inputs_prepared = copy(sample.inputs)
        inputs_prepared = prepare_inputs_memory(
            inputs=inputs_prepared,
            na_augment_perc=self.na_augment[0],
            na_augment_prob=self.na_augment[1],
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs=inputs_prepared, data_dimensions=self.data_dimensions
        )

        target_labels = sample.target_labels
        sample_id = sample.sample_id

        return inputs_final, target_labels, sample_id

    def __len__(self):
        return len(self.samples)


def _get_default_impute_fill_values(data_sources: Iterable[str]):
    fill_values = {}
    for source in data_sources:
        if source.startswith("omics_"):
            fill_values[source] = False
        else:
            fill_values[source] = -1

    return fill_values


def _get_default_impute_dtypes(data_sources: Iterable[str]):
    dtypes = {}
    for source in data_sources:
        if source.startswith("omics_"):
            dtypes[source] = torch.bool
        else:
            dtypes[source] = torch.float

    return dtypes


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
        self.check_samples()

    # Note that dataloaders automatically convert arrays to tensors here, for labels
    def __getitem__(self, index: int) -> al_getitem_return:
        sample = self.samples[index]

        inputs_prepared = copy(sample.inputs)
        inputs_prepared = prepare_inputs_disk(
            inputs=inputs_prepared,
            na_augment_perc=self.na_augment[0],
            na_augment_prob=self.na_augment[1],
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs=inputs_prepared, data_dimensions=self.data_dimensions
        )

        target_labels = sample.target_labels
        sample_id = sample.sample_id

        return inputs_final, target_labels, sample_id

    def __len__(self):
        return len(self.samples)


def prepare_inputs_disk(
    inputs: Dict[str, Any], na_augment_perc: float, na_augment_prob: float
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}
    for name, data in inputs.items():

        if name.startswith("omics_"):
            array_raw = _load_one_hot_array(path=data)
            array_prepared = prepare_one_hot_omics_data(
                genotype_array=array_raw,
                na_augment_perc=na_augment_perc,
                na_augment_prob=na_augment_prob,
            )
            prepared_inputs[name] = array_prepared

        else:
            prepared_inputs[name] = inputs[name]

    return prepared_inputs


def prepare_inputs_memory(
    inputs: Dict[str, Any], na_augment_perc: float, na_augment_prob: float
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}
    for name, data in inputs.items():

        if name.startswith("omics_"):
            array_raw = data
            array_prepared = prepare_one_hot_omics_data(
                genotype_array=array_raw,
                na_augment_perc=na_augment_perc,
                na_augment_prob=na_augment_prob,
            )
            prepared_inputs[name] = array_prepared

        else:
            prepared_inputs[name] = inputs[name]

    return prepared_inputs


def impute_missing_modalities_wrapper(
    inputs: Dict[str, Any], data_dimensions: Dict[str, "DataDimensions"]
):
    impute_dtypes = _get_default_impute_dtypes(data_sources=data_dimensions.keys())
    impute_fill_values = _get_default_impute_fill_values(
        data_sources=data_dimensions.keys()
    )
    inputs_imputed = impute_missing_modalities(
        inputs=inputs,
        data_dimensions=data_dimensions,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    return inputs_imputed


def impute_missing_modalities(
    inputs: Dict[str, Any],
    data_dimensions: Dict[str, "DataDimensions"],
    fill_values: Dict[str, Any],
    dtypes: Dict[str, Any],
) -> Dict[str, torch.Tensor]:

    for name, dimensions in data_dimensions.items():
        if name not in inputs:
            fill_value = fill_values[name]
            dtype = dtypes[name]
            shape = dimensions.channels, dimensions.height, dimensions.width

            imputed_tensor = impute_single_missing_modality(
                shape=shape, fill_value=fill_value, dtype=dtype
            )

            inputs[name] = imputed_tensor

    return inputs


def impute_single_missing_modality(shape: Tuple[int, ...], fill_value: Any, dtype: Any):
    imputed_tensor = torch.empty(shape, dtype=dtype).fill_(fill_value)
    return imputed_tensor


def _load_one_hot_array(path: Path) -> torch.Tensor:
    genotype_array_raw = np.load(str(path))
    genotype_array_raw = torch.from_numpy(genotype_array_raw).unsqueeze(0)

    return genotype_array_raw
