import reprlib
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
    impute_missing_output_modalities_wrapper,
)
from eir.data_load.data_preparation_modules.input_preparation_wrappers import (
    get_input_data_loading_hooks,
    prepare_inputs_disk,
    prepare_inputs_memory,
)
from eir.data_load.data_preparation_modules.output_preparation_wrappers import (
    get_output_data_loading_hooks,
    prepare_outputs_disk,
    prepare_outputs_memory,
)
from eir.data_load.data_preparation_modules.prepare_tabular import (
    add_tabular_data_to_samples,
)
from eir.data_load.data_source_modules import deeplake_ops
from eir.data_load.data_source_modules.common_utils import add_id_to_samples
from eir.data_load.data_source_modules.local_ops import (
    add_sequence_data_from_csv_to_samples,
    get_file_sample_id_iterator_basic,
)
from eir.data_load.label_setup import al_target_label_dict
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.setup import config
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.schemas import SequenceInputDataConfig
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.target_setup.target_label_setup import (
        MergedTargetLabels,
        MissingTargetsInfo,
    )

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type Aliases
al_datasets = Union["MemoryDataset", "DiskDataset"]
al_dataset_types = Type[al_datasets]
# embeddings --> remain str, cat targets --> int, con extra/target --> float
al_sample_label_dict_target = Dict[str, Dict[str, Union[int, float, torch.Tensor]]]
al_inputs = Union[Dict[str, torch.Tensor], Dict[str, Any]]
al_getitem_return = tuple[dict[str, torch.Tensor], al_sample_label_dict_target, str]


def set_up_datasets_from_configs(
    configs: config.Configs,
    target_labels: "MergedTargetLabels",
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    train_ids_to_keep: Optional[Sequence[str]] = None,
    valid_ids_to_keep: Optional[Sequence[str]] = None,
) -> Tuple[al_datasets, al_datasets]:
    dataset_class: al_dataset_types = (
        MemoryDataset if configs.global_config.memory_dataset else DiskDataset
    )

    train_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels.train_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=False,
        ids_to_keep=train_ids_to_keep,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    valid_kwargs = construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels.valid_labels,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        test_mode=True,
        ids_to_keep=valid_ids_to_keep,
        missing_ids_per_output=target_labels.missing_ids_per_output,
    )

    train_dataset: al_datasets = dataset_class(**train_kwargs)
    valid_dataset: al_datasets = dataset_class(**valid_kwargs)

    _check_valid_and_train_datasets(
        train_dataset=train_dataset, valid_dataset=valid_dataset
    )

    return train_dataset, valid_dataset


def construct_default_dataset_kwargs_from_cl_args(
    target_labels_dict: Union[None, al_target_label_dict],
    inputs: al_input_objects_as_dict,
    outputs: "al_output_objects_as_dict",
    test_mode: bool,
    missing_ids_per_output: "MissingTargetsInfo",
    ids_to_keep: Union[None, Sequence[str]] = None,
) -> Dict[str, Any]:
    ids_to_keep_set = set(ids_to_keep) if ids_to_keep is not None else None

    dataset_kwargs = {
        "inputs": inputs,
        "outputs": outputs,
        "target_labels_dict": target_labels_dict,
        "test_mode": test_mode,
        "missing_ids_per_output": missing_ids_per_output,
        "ids_to_keep": ids_to_keep_set,
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
        inputs: al_input_objects_as_dict,
        outputs: "al_output_objects_as_dict",
        test_mode: bool,
        missing_ids_per_output: "MissingTargetsInfo",
        target_labels_dict: Optional[al_target_label_dict] = None,
        ids_to_keep: Optional[Set[str]] = None,
    ):
        super().__init__()

        self.samples: List[Sample] = []

        self.inputs = inputs
        self.outputs = outputs
        self.test_mode = test_mode
        self.target_labels_dict = target_labels_dict if target_labels_dict else {}
        self.missing_ids_per_output = missing_ids_per_output
        self.ids_to_keep = set(ids_to_keep) if ids_to_keep else None

    def init_label_attributes(self):
        if not self.outputs:
            raise ValueError("Please specify label column name.")

    def set_up_samples(
        self,
        input_data_loading_hooks: Optional[Mapping[str, Callable]] = None,
        output_data_loading_hooks: Optional[Mapping[str, Callable]] = None,
    ) -> List[Sample]:
        """
        We do an extra filtering step at the end to account for the situation where
        we have a target label file with more samples than there are any inputs
        available for. This is quite likely if we have e.g. pre-split data into
        train/val and test folders.
        """

        def _identity(sample_data: Any) -> Any:
            return sample_data

        def _default_sample_factory() -> Sample:
            return Sample(sample_id="", inputs={}, target_labels={})

        mode_str = "evaluation/test" if self.test_mode else "train"
        logger.debug("Setting up dataset in %s mode.", mode_str)

        if not output_data_loading_hooks:
            output_data_loading_hooks = defaultdict(lambda: _identity)

        samples: DefaultDict[str, Sample] = defaultdict(_default_sample_factory)

        if self.target_labels_dict:
            samples = _add_target_labels_to_samples(
                target_labels_dict=self.target_labels_dict,
                samples=samples,
                output_data_loading_hooks=output_data_loading_hooks,
            )

        if not input_data_loading_hooks:
            input_data_loading_hooks = defaultdict(lambda: _identity)

        ids_to_keep = initialize_ids_to_keep(
            target_labels_dict=self.target_labels_dict,
            ids_to_keep=self.ids_to_keep,
        )
        samples = add_data_to_samples(
            inputs=self.inputs,
            samples=samples,
            ids_to_keep=ids_to_keep,
            data_loading_hooks=input_data_loading_hooks,
        )

        samples_list = filter_samples(
            samples=samples, target_labels_dict=self.target_labels_dict
        )

        _log_missing_samples_between_modalities(
            samples=samples_list, input_keys=self.inputs.keys()
        )
        return samples_list

    def __len__(self):
        return len(self.samples)

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


def initialize_ids_to_keep(
    target_labels_dict: Dict, ids_to_keep: Optional[set]
) -> Optional[set]:
    if target_labels_dict:
        if ids_to_keep:
            ids_to_keep = set(i for i in target_labels_dict.keys() if i in ids_to_keep)
        else:
            ids_to_keep = set(target_labels_dict.keys())
    return ids_to_keep


def add_data_to_samples(
    inputs: al_input_objects_as_dict,
    samples: DefaultDict[str, "Sample"],
    ids_to_keep: Optional[set],
    data_loading_hooks: Mapping[str, Callable],
) -> DefaultDict[str, "Sample"]:
    for input_name, input_object in inputs.items():
        input_info = input_object.input_config.input_info
        input_source = input_info.input_source
        input_inner_key = input_info.input_inner_key

        match input_object:
            case ComputedTabularInputInfo() | ComputedPredictTabularInputInfo():
                samples = add_tabular_data_to_samples(
                    df_tabular=input_object.labels.all_labels,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    source_name=input_name,
                )
            case ComputedSequenceInputInfo() if Path(input_source).suffix == ".csv":
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, SequenceInputDataConfig)
                samples = add_sequence_data_from_csv_to_samples(
                    input_source=input_source,
                    input_name=input_name,
                    samples=samples,
                    encode_func=input_object.encode_func,
                    split_on=input_type_info.split_on,
                    ids_to_keep=ids_to_keep,
                )
            case _:
                samples = _add_data_to_samples_wrapper(
                    input_source=input_source,
                    input_name=input_name,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    data_loading_hook=data_loading_hooks[input_name],
                    deeplake_input_inner_key=input_inner_key,
                )
    return samples


def filter_samples(
    samples: DefaultDict[str, "Sample"], target_labels_dict: Optional[Dict]
) -> List[Sample]:
    num_samples_raw = len(samples)
    if target_labels_dict:
        samples_list = list(i for i in samples.values() if i.inputs and i.target_labels)
        num_missing = num_samples_raw - len(samples_list)
        logger.info(
            "Filtered out %d samples that had no inputs or no target labels.",
            num_missing,
        )
    else:
        samples_list = list(i for i in samples.values() if i.inputs)
        num_missing = num_samples_raw - len(samples_list)
        logger.info(
            "Filtered out %d samples that had no inputs.",
            num_missing,
        )
    return samples_list


def _log_missing_samples_between_modalities(
    samples: Sequence[Sample], input_keys: Iterable[str]
) -> None:
    missing_counts = {k: 0 for k in input_keys}
    missing_ids: dict[str, list[str]] = {k: [] for k in input_keys}
    any_missing = False

    for sample in samples:
        for key in input_keys:
            if key not in sample.inputs:
                missing_counts[key] += 1
                missing_ids[key].append(sample.sample_id)
                any_missing = True

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

    if any_missing:
        warning_message = (
            "There were missing inputs in samples for some modalities, "
            "Please review the info log above for detailed counts and IDs. "
            "If this is expected, ignore this message. If not, possible "
            "causes are (a) different inputs having different sample IDs "
            "present, (b) sample IDs between inputs / targets are not matching, "
            "or (c) something else."
        )
        logger.warning(warning_message)


def _add_target_labels_to_samples(
    target_labels_dict: al_target_label_dict,
    samples: DefaultDict[str, Sample],
    output_data_loading_hooks: Mapping[str, Callable],
) -> DefaultDict[str, Sample]:
    target_label_iterator = tqdm(target_labels_dict.items(), desc="Target Labels")

    for sample_id, sample_target_labels_dict in target_label_iterator:
        add_id_to_samples(samples=samples, sample_id=sample_id)

        target_labels_loaded = {}
        for output_name, output_target_labels in sample_target_labels_dict.items():
            cur_hook = output_data_loading_hooks[output_name]
            cur_target_labels = cur_hook(output_target_labels)
            target_labels_loaded[output_name] = cur_target_labels

        samples[sample_id].target_labels = target_labels_loaded

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
        assert deeplake_input_inner_key is not None
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

        self.samples: list[Sample] = self.set_up_samples()
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
            inputs_values=inputs_prepared,
            inputs_objects=self.inputs,
        )

        target_labels = sample.target_labels

        targets_prepared = prepare_outputs_disk(
            outputs=target_labels,
            output_objects=self.outputs,
            test_mode=self.test_mode,
        )

        targets_final = impute_missing_output_modalities_wrapper(
            outputs_values=targets_prepared,
            output_objects=self.outputs,
        )

        sample_id = sample.sample_id
        return inputs_final, targets_final, sample_id

    def __len__(self):
        return len(self.samples)


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        input_data_loading_hooks = get_input_data_loading_hooks(inputs=self.inputs)
        output_data_loading_hooks = get_output_data_loading_hooks(outputs=self.outputs)

        self.samples: list[Sample] = self.set_up_samples(
            input_data_loading_hooks=input_data_loading_hooks,
            output_data_loading_hooks=output_data_loading_hooks,
        )
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
            inputs_values=inputs_prepared,
            inputs_objects=self.inputs,
        )

        target_labels = sample.target_labels

        targets_prepared = prepare_outputs_memory(
            outputs=target_labels,
            output_objects=self.outputs,
            test_mode=self.test_mode,
        )

        targets_final = impute_missing_output_modalities_wrapper(
            outputs_values=targets_prepared,
            output_objects=self.outputs,
        )

        sample_id = sample.sample_id

        return inputs_final, targets_final, sample_id

    def __len__(self):
        return len(self.samples)
