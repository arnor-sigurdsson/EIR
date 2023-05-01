import reprlib
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
    Sequence,
    Optional,
    Set,
    Iterable,
    Mapping,
    DefaultDict,
    Any,
    TYPE_CHECKING,
)

import torch
from aislib.misc_utils import get_logger
from torch.utils.data import Dataset
from tqdm import tqdm

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.preparation_wrappers import (
    prepare_inputs_disk,
    prepare_inputs_memory,
    get_data_loading_hooks,
)
from eir.data_load.data_preparation_modules.prepare_tabular import (
    add_tabular_data_to_samples,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.data_load.data_source_modules.common_utils import add_id_to_samples
from eir.data_load.data_source_modules.local_ops import (
    get_file_sample_id_iterator_basic,
    add_sequence_data_from_csv_to_samples,
)
from eir.data_load.label_setup import al_label_dict, al_target_label_dict
from eir.setup import config

if TYPE_CHECKING:
    from eir.setup.input_setup import (
        al_input_objects_as_dict,
    )
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.target_setup.target_label_setup import MergedTargetLabels

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryDataset", "DiskDataset"]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_label_dict_target = Dict[str, Dict[str, Union[int, float]]]
al_inputs = Union[Dict[str, torch.Tensor], Dict[str, Any]]
al_getitem_return = Tuple[Dict[str, torch.Tensor], al_sample_label_dict_target, str]


def set_up_datasets_from_configs(
    configs: config.Configs,
    target_labels: "MergedTargetLabels",
    inputs_as_dict: "al_input_objects_as_dict",
    outputs_as_dict: "al_output_objects_as_dict",
    train_ids_to_keep: Optional[Sequence[str]] = None,
    valid_ids_to_keep: Optional[Sequence[str]] = None,
) -> Tuple[al_datasets, al_datasets]:
    dataset_class = (
        MemoryDataset if configs.global_config.memory_dataset else DiskDataset
    )

    train_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels.train_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=False,
        ids_to_keep=train_ids_to_keep,
    )

    valid_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels.valid_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=True,
        ids_to_keep=valid_ids_to_keep,
    )

    train_dataset = dataset_class(**train_kwargs)
    valid_dataset = dataset_class(**valid_kwargs)

    _check_valid_and_train_datasets(
        train_dataset=train_dataset, valid_dataset=valid_dataset
    )

    return train_dataset, valid_dataset


def construct_default_dataset_kwargs_from_cl_args(
    target_labels_dict: Union[None, al_target_label_dict],
    inputs: "al_input_objects_as_dict",
    outputs: "al_output_objects_as_dict",
    test_mode: bool,
    ids_to_keep: Union[None, Sequence[str]] = None,
) -> Dict[str, Any]:
    ids_to_keep = set(ids_to_keep) if ids_to_keep is not None else None

    dataset_kwargs = {
        "inputs": inputs,
        "outputs": outputs,
        "target_labels_dict": target_labels_dict,
        "test_mode": test_mode,
        "ids_to_keep": ids_to_keep,
    }

    return dataset_kwargs


def _check_valid_and_train_datasets(
    train_dataset: al_datasets, valid_dataset: al_datasets
) -> None:
    if len(train_dataset) < len(valid_dataset):
        logger.warning(
            "Size of training dataset (size: %d) is smaller than validation dataset ("
            "size: %d). Generally it is the opposite, but if this intended please"
            "ignore this message.",
            len(train_dataset),
            len(valid_dataset),
        )

    assert set(valid_dataset.target_labels_dict.keys()).isdisjoint(
        train_dataset.target_labels_dict.keys()
    )


@dataclass
class Sample:
    sample_id: str
    inputs: Dict[str, Any]
    target_labels: al_sample_label_dict_target


class DatasetBase(Dataset):
    def __init__(
        self,
        inputs: "al_input_objects_as_dict",
        outputs: "al_output_objects_as_dict",
        test_mode: bool,
        target_labels_dict: al_target_label_dict = None,
        ids_to_keep: Optional[Set[str]] = None,
    ):
        super().__init__()

        self.samples: Union[List[Sample], None] = None

        self.inputs = inputs
        self.outputs = outputs
        self.test_mode = test_mode
        self.target_labels_dict = target_labels_dict if target_labels_dict else {}
        self.ids_to_keep = set(ids_to_keep) if ids_to_keep else None

    def init_label_attributes(self):
        if not self.outputs:
            raise ValueError("Please specify label column name.")

    def set_up_samples(
        self, data_loading_hooks: Mapping[str, Callable] = None
    ) -> List[Sample]:
        """
        We do an extra filtering step at the end to account for the situation where
        we have a target label file with more samples than there are any inputs
        available for. This is quite likely if we have e.g. pre-split data into
        train/val and test folders.
        """

        mode_str = "evaluation/test" if self.test_mode else "train"
        logger.debug("Setting up dataset in %s mode.", mode_str)

        def _identity(sample_data: Any) -> Any:
            return sample_data

        if data_loading_hooks is None:
            data_loading_hooks = defaultdict(lambda: _identity)

        def _default_sample_factory() -> Sample:
            return Sample(sample_id="", inputs={}, target_labels={})

        samples = defaultdict(_default_sample_factory)

        ids_to_keep = self.ids_to_keep
        if self.target_labels_dict:
            samples = _add_target_labels_to_samples(
                target_labels_dict=self.target_labels_dict, samples=samples
            )
            if ids_to_keep:
                ids_to_keep = set(
                    i for i in self.target_labels_dict.keys() if i in ids_to_keep
                )
            else:
                ids_to_keep = set(self.target_labels_dict.keys())

        for input_name, input_object in self.inputs.items():
            input_info = input_object.input_config.input_info
            input_source = input_info.input_source
            input_type = input_info.input_type
            input_inner_key = input_info.input_inner_key

            is_tabular = input_type == "tabular"
            is_sequence = input_type in ("sequence", "bytes")

            if is_tabular:
                samples = add_tabular_data_to_samples(
                    tabular_dict=input_object.labels.all_labels,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    source_name=input_name,
                )
            elif is_sequence and Path(input_source).suffix == ".csv":
                samples = add_sequence_data_from_csv_to_samples(
                    input_source=input_source,
                    input_name=input_name,
                    samples=samples,
                    encode_func=input_object.encode_func,
                    split_on=input_object.input_config.input_type_info.split_on,
                    ids_to_keep=ids_to_keep,
                )
            else:
                samples = _add_data_to_samples_wrapper(
                    input_source=input_source,
                    input_name=input_name,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    data_loading_hook=data_loading_hooks[input_name],
                    deeplake_input_inner_key=input_inner_key,
                )

        num_samples_raw = len(samples)
        if self.target_labels_dict:
            samples = list(i for i in samples.values() if i.inputs and i.target_labels)
            num_missing = num_samples_raw - len(samples)
            logger.info(
                "Filtered out %d samples that had no inputs or no target labels.",
                num_missing,
            )
        else:
            samples = list(i for i in samples.values() if i.inputs)
            num_missing = num_samples_raw - len(samples)
            logger.info(
                "Filtered out %d samples that had no inputs.",
                num_missing,
            )

        _log_missing_samples_between_modalities(
            samples=samples, input_keys=self.inputs.keys()
        )
        return samples

    def __getitem__(self, index: int):
        raise NotImplementedError()

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

        if self.samples is None or len(self.samples) == 0:
            raise ValueError(
                f"Expected to have at least one sample, but got {self.samples} instead."
                f" Possibly there is a mismatch between input IDs and target IDs."
            )

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


def _log_missing_samples_between_modalities(
    samples: Sequence[Sample], input_keys: Iterable[str]
) -> None:
    missing_counts = {k: 0 for k in input_keys}
    missing_ids = {k: [] for k in input_keys}

    for sample in samples:
        for key in input_keys:
            if key not in sample.inputs:
                missing_counts[key] += 1
                missing_ids[key].append(sample.sample_id)

    no_samples = len(samples)
    message = (
        f"Using total of {no_samples} samples with following counts per "
        f"modality (note missing tabular modalities have been imputed already):\n"
    )

    for key in input_keys:
        cur_missing = missing_counts[key]
        cur_present = no_samples - cur_missing
        cur_missing_ids = reprlib.repr(missing_ids[key])
        cur_str = (
            f"Available {key}: {cur_present} "
            f"(missing: {cur_missing}, missing IDs: {cur_missing_ids})"
        )

        cur_str += "\n"
        message += cur_str

    logger.info(message.rstrip())


def _add_target_labels_to_samples(
    target_labels_dict: al_label_dict, samples: DefaultDict[str, Sample]
) -> DefaultDict[str, Sample]:
    target_label_iterator = tqdm(target_labels_dict.items(), desc="Target Labels")

    for sample_id, sample_target_labels in target_label_iterator:
        add_id_to_samples(samples=samples, sample_id=sample_id)
        samples[sample_id].target_labels = sample_target_labels

    return samples


def _add_data_to_samples_wrapper(
    input_source: str,
    input_name: str,
    samples: DefaultDict[str, Sample],
    ids_to_keep: Union[None, Set[str]],
    data_loading_hook: Callable,
    deeplake_input_inner_key: Optional[str] = None,
) -> DefaultDict[str, Sample]:
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        samples = deeplake_ops.add_deeplake_data_to_samples(
            input_source=input_source,
            input_name=input_name,
            samples=samples,
            ids_to_keep=ids_to_keep,
            deeplake_input_inner_key=deeplake_input_inner_key,
            data_loading_hook=data_loading_hook,
        )

    else:
        samples = _add_file_data_to_samples(
            input_source=input_source,
            samples=samples,
            ids_to_keep=ids_to_keep,
            data_loading_hook=data_loading_hook,
            input_name=input_name,
        )

    return samples


def _add_file_data_to_samples(
    input_source: str,
    input_name: str,
    samples: DefaultDict[str, Sample],
    ids_to_keep: Union[None, Set[str]],
    data_loading_hook: Callable,
) -> DefaultDict[str, Sample]:
    file_data_iterator = get_file_sample_id_iterator_basic(
        data_source=input_source, ids_to_keep=ids_to_keep
    )
    file_iterator_tqdm = tqdm(file_data_iterator, desc=input_name)

    for sample_id, file in file_iterator_tqdm:
        sample_data = data_loading_hook(file)

        samples = add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[input_name] = sample_data

    return samples


class DiskDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        self.samples = self.set_up_samples()
        self.check_samples()

    def __getitem__(self, index: int) -> al_getitem_return:
        """
        NB: Dataloaders automatically convert arrays to tensors here when returning.
        """
        sample = self.samples[index]

        inputs = copy(sample.inputs)
        inputs_prepared = prepare_inputs_disk(
            inputs=inputs, inputs_objects=self.inputs, test_mode=self.test_mode
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs_values=inputs_prepared, inputs_objects=self.inputs
        )

        target_labels = sample.target_labels
        sample_id = sample.sample_id
        return inputs_final, target_labels, sample_id

    def __len__(self):
        return len(self.samples)


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        data_loading_hooks = get_data_loading_hooks(inputs=self.inputs)
        self.samples = self.set_up_samples(data_loading_hooks=data_loading_hooks)
        self.check_samples()

    def __getitem__(self, index: int) -> al_getitem_return:
        """
        NB: Dataloaders automatically convert arrays to tensors here when returning.
        """
        sample = self.samples[index]

        inputs = copy(sample.inputs)
        inputs_prepared = prepare_inputs_memory(
            inputs=inputs, inputs_objects=self.inputs, test_mode=self.test_mode
        )

        inputs_final = impute_missing_modalities_wrapper(
            inputs_values=inputs_prepared, inputs_objects=self.inputs
        )

        target_labels = sample.target_labels
        sample_id = sample.sample_id

        return inputs_final, target_labels, sample_id

    def __len__(self):
        return len(self.samples)
