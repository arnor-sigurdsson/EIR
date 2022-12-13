import reprlib
from collections import defaultdict
from copy import copy
import warnings
from dataclasses import dataclass
from functools import partial
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
    Literal,
    TYPE_CHECKING,
)

import numpy as np
import torch
from PIL.Image import Image, fromarray
from aislib.misc_utils import get_logger
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from eir.data_load.data_augmentation import make_random_omics_columns_missing
from eir.data_load.data_source_modules import deeplake_ops
from eir.data_load.data_source_modules.common_utils import add_id_to_samples
from eir.data_load.data_source_modules.local_ops import (
    get_file_sample_id_iterator_basic,
    add_sequence_data_from_csv_to_samples,
)
from eir.data_load.label_setup import (
    al_label_dict,
)
from eir.setup import config
from eir.setup.input_setup import get_sequence_split_function

if TYPE_CHECKING:
    from eir.setup.input_setup import (
        al_input_objects_as_dict,
        SequenceInputInfo,
        TabularInputInfo,
        BytesInputInfo,
        ImageInputInfo,
    )
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.train import MergedTargetLabels

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
    target_labels_dict: Union[None, al_label_dict],
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
        target_labels_dict: al_label_dict = None,
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

            if input_type == "omics":

                samples = _add_data_to_samples_wrapper(
                    input_source=input_source,
                    input_name=input_name,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    data_loading_hook=data_loading_hooks[input_name],
                    deeplake_input_inner_key=input_inner_key,
                )

            elif input_type == "image":

                samples = _add_data_to_samples_wrapper(
                    input_source=input_source,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    data_loading_hook=data_loading_hooks[input_name],
                    input_name=input_name,
                    deeplake_input_inner_key=input_inner_key,
                )

            elif input_type == "tabular":
                samples = _add_tabular_data_to_samples(
                    tabular_dict=input_object.labels.all_labels,
                    samples=samples,
                    ids_to_keep=ids_to_keep,
                    source_name=input_name,
                )

            elif input_type in ("sequence", "bytes"):

                if Path(input_source).is_dir():
                    samples = _add_data_to_samples_wrapper(
                        input_source=input_source,
                        input_name=input_name,
                        samples=samples,
                        ids_to_keep=ids_to_keep,
                        data_loading_hook=data_loading_hooks[input_name],
                        deeplake_input_inner_key=input_inner_key,
                    )

                elif Path(input_source).suffix == ".csv":
                    samples = add_sequence_data_from_csv_to_samples(
                        input_object=input_source,
                        samples=samples,
                        encode_func=input_object.encode_func,
                        split_on=input_object.input_config.input_type_info.split_on,
                        ids_to_keep=ids_to_keep,
                        source_name=input_name,
                    )

        num_samples_raw = len(samples)
        if self.target_labels_dict:
            samples = list(i for i in samples.values() if i.inputs and i.target_labels)
            num_missing = num_samples_raw - len(samples)
            logger.debug(
                "Filtered out %d samples that had no inputs or no target labels.",
                num_missing,
            )
        else:
            samples = list(i for i in samples.values() if i.inputs)
            num_missing = num_samples_raw - len(samples)
            logger.debug(
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

    logger.debug(message.rstrip())


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


def _add_tabular_data_to_samples(
    tabular_dict: al_label_dict,
    samples: DefaultDict[str, Sample],
    ids_to_keep: Union[None, Sequence[str]],
    source_name: str = "Tabular Data",
) -> DefaultDict[str, Sample]:
    def _get_tabular_iterator(ids_to_keep_: Union[None, set]):

        for sample_id_, tabular_inputs_ in tabular_dict.items():
            if ids_to_keep_ and sample_id_ not in ids_to_keep_:
                continue

            yield sample_id_, tabular_inputs_

    known_length = None if not ids_to_keep else len(ids_to_keep)
    tabular_iterator = tqdm(
        _get_tabular_iterator(ids_to_keep_=ids_to_keep),
        desc=source_name,
        total=known_length,
    )

    for sample_id, tabular_inputs in tabular_iterator:
        samples = add_id_to_samples(samples=samples, sample_id=sample_id)

        samples[sample_id].inputs[source_name] = tabular_inputs

    return samples


class MemoryDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.target_labels_dict:
            self.init_label_attributes()

        data_loading_hooks = self.data_loading_hooks()
        self.samples = self.set_up_samples(data_loading_hooks=data_loading_hooks)
        self.check_samples()

    def data_loading_hooks(
        self,
    ) -> Mapping[str, Callable[..., torch.Tensor]]:
        mapping = {}

        for input_name, input_object in self.inputs.items():
            input_type = input_object.input_config.input_info.input_type
            input_source = input_object.input_config.input_info.input_source

            if input_type == "omics":
                inner_key = input_object.input_config.input_info.input_inner_key
                mapping[input_name] = partial(
                    _omics_load_wrapper,
                    subset_indices=self.inputs[input_name].subset_indices,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                )

            elif input_type == "image":
                inner_key = input_object.input_config.input_info.input_inner_key
                mapping[input_name] = partial(
                    _image_load_wrapper,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                )

            elif input_type == "sequence":
                inner_key = input_object.input_config.input_info.input_inner_key
                mapping[input_name] = partial(
                    _sequence_load_wrapper,
                    split_on=input_object.input_config.input_type_info.split_on,
                    encode_func=self.inputs[input_name].encode_func,
                    input_source=input_source,
                    deeplake_inner_key=inner_key,
                )
            elif input_type == "bytes":
                mapping[input_name] = partial(
                    _bytes_load_wrapper,
                    dtype=input_object.input_config.input_type_info.byte_encoding,
                    input_source=input_source,
                )

        return mapping

    def __getitem__(self, index: int) -> al_getitem_return:
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


def _sequence_load_wrapper(
    data_pointer: Union[Path, int, np.ndarray],
    input_source: str,
    split_on: str,
    encode_func: Callable[[Sequence[str]], List[int]],
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:
    """
    In the case of .csv input sources, we have already loaded and tokenized the data.
    """

    split_func = get_sequence_split_function(split_on=split_on)
    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        text_as_np_array = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        content = text_as_np_array[0]
    elif input_source.endswith(".csv"):
        return data_pointer
    else:
        content = load_sequence_from_disk(sequence_file_path=data_pointer)

    file_content_split = split_func(content)
    file_content_encoded = encode_func(file_content_split)
    sequence_tokenized = np.array(file_content_encoded)

    return sequence_tokenized


def _get_default_impute_fill_values(inputs_objects: "al_input_objects_as_dict"):
    fill_values = {}
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_type == "omics":
            fill_values[input_name] = False
        elif input_type == "tabular":
            fill_values[input_name] = _build_tabular_fill_value(
                input_object=input_object
            )
        else:
            fill_values[input_name] = 0

    return fill_values


def _build_tabular_fill_value(input_object: "TabularInputInfo"):
    fill_value = {}
    transformers = input_object.labels.label_transformers

    cat_columns = input_object.input_config.input_type_info.input_cat_columns
    for cat_column in cat_columns:
        cur_label_encoder = transformers[cat_column]
        fill_value[cat_column] = cur_label_encoder.transform(["NA"]).item()

    con_columns = input_object.input_config.input_type_info.input_con_columns
    for con_column in con_columns:
        fill_value[con_column] = 0.0

    return fill_value


def _get_default_impute_dtypes(inputs_objects: "al_input_objects_as_dict"):
    dtypes = {}
    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type
        if input_type == "omics":
            dtypes[input_name] = torch.bool
        elif input_type in ("sequence", "bytes"):
            dtypes[input_name] = torch.long
        else:
            dtypes[input_name] = torch.float

    return dtypes


def prepare_one_hot_omics_data(
    genotype_array: np.ndarray,
    na_augment_perc: float,
    na_augment_prob: float,
    test_mode: bool,
) -> torch.BoolTensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """

    tensor_bool = torch.BoolTensor(genotype_array).unsqueeze(0).detach().clone()

    if not test_mode and na_augment_perc > 0 and na_augment_prob > 0:
        tensor_bool = make_random_omics_columns_missing(
            omics_array=tensor_bool,
            percentage=na_augment_perc,
            probability=na_augment_prob,
        )

    return tensor_bool


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


def prepare_inputs_disk(
    inputs: Dict[str, Any], inputs_objects: "al_input_objects_as_dict", test_mode: bool
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}
    for input_name, data_pointer in inputs.items():

        input_object = inputs_objects[input_name]

        input_source = input_object.input_config.input_info.input_source
        deeplake_inner_key = input_object.input_config.input_info.input_inner_key
        input_type_info = input_object.input_config.input_type_info
        input_type = input_object.input_config.input_info.input_type

        if input_type == "omics":
            array_raw = _omics_load_wrapper(
                input_source=input_source,
                data_pointer=data_pointer,
                deeplake_inner_key=deeplake_inner_key,
                subset_indices=input_object.subset_indices,
            )
            array_prepared = prepare_one_hot_omics_data(
                genotype_array=array_raw,
                na_augment_perc=input_type_info.na_augment_perc,
                na_augment_prob=input_type_info.na_augment_prob,
                test_mode=test_mode,
            )
            prepared_inputs[input_name] = array_prepared

        elif input_type == "sequence":

            sequence_tokenized = _sequence_load_wrapper(
                data_pointer=data_pointer,
                input_source=input_source,
                deeplake_inner_key=deeplake_inner_key,
                split_on=input_type_info.split_on,
                encode_func=input_object.encode_func,
            )
            prepared_sequence_inputs = prepare_sequence_data(
                sequence_input_object=inputs_objects[input_name],
                cur_file_content_tokenized=sequence_tokenized,
                test_mode=test_mode,
            )
            prepared_inputs[input_name] = prepared_sequence_inputs

        elif input_type == "bytes":

            bytes_data = _bytes_load_wrapper(
                data_pointer=data_pointer,
                dtype=input_type_info.byte_encoding,
                input_source=input_source,
                deeplake_inner_key=deeplake_inner_key,
            )
            prepared_bytes_input = prepare_bytes_data(
                bytes_input_object=inputs_objects[input_name],
                bytes_data=bytes_data,
                test_mode=test_mode,
            )
            prepared_inputs[input_name] = prepared_bytes_input

        elif input_type == "image":
            image_data = _image_load_wrapper(
                input_source=input_source,
                data_pointer=data_pointer,
                deeplake_inner_key=deeplake_inner_key,
            )

            prepared_image_data = prepare_image_data(
                image_input_object=inputs_objects[input_name],
                image_data=image_data,
                test_mode=test_mode,
            )
            prepared_inputs[input_name] = prepared_image_data

        else:
            prepared_inputs[input_name] = inputs[input_name]

    return prepared_inputs


def prepare_image_data(
    image_input_object: "ImageInputInfo", image_data: Image, test_mode: bool
) -> torch.Tensor:

    """
    The transforms take care of converting the image object to a copied tensor.
    """

    image_data_clone = image_data.copy()

    if test_mode:
        image_prepared = image_input_object.base_transforms(img=image_data_clone)
    else:
        image_prepared = image_input_object.all_transforms(img=image_data_clone)

    return image_prepared


def prepare_bytes_data(
    bytes_input_object: "BytesInputInfo", bytes_data: np.ndarray, test_mode: bool
) -> torch.Tensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """
    bio = bytes_input_object

    sampling_strat = bio.input_config.input_type_info.sampling_strategy_if_longer
    if test_mode:
        sampling_strat = "from_start"

    bytes_tensor = torch.LongTensor(bytes_data).detach().clone()

    padding_value = bio.vocab.get("<pad>", 0)
    cur_bytes_padded = process_tensor_to_length(
        tensor=bytes_tensor,
        max_length=bio.input_config.input_type_info.max_length,
        sampling_strategy_if_longer=sampling_strat,
        padding_value=padding_value,
    )

    return cur_bytes_padded


def prepare_sequence_data(
    sequence_input_object: "SequenceInputInfo",
    cur_file_content_tokenized: np.ndarray,
    test_mode: bool,
) -> torch.Tensor:
    """
    We use clone here to copy the original data, vs. using from_numpy
    which shares memory, causing us to modify the original data.
    """

    sio = sequence_input_object

    cur_tokens_as_tensor = torch.LongTensor(cur_file_content_tokenized).detach().clone()

    sampling_strategy = sio.input_config.input_type_info.sampling_strategy_if_longer
    if test_mode:
        sampling_strategy = "from_start"

    padding_token = getattr(sio.tokenizer, "pad_token", "<pad>")
    padding_value = sio.encode_func([padding_token])[0]
    cur_tokens_padded = process_tensor_to_length(
        tensor=cur_tokens_as_tensor,
        max_length=sio.computed_max_length,
        sampling_strategy_if_longer=sampling_strategy,
        padding_value=padding_value,
    )

    return cur_tokens_padded


def process_tensor_to_length(
    tensor: torch.Tensor,
    max_length: int,
    sampling_strategy_if_longer: Literal["from_start", "uniform"],
    padding_value: int = 0,
) -> torch.Tensor:
    tensor_length = len(tensor)

    if tensor_length > max_length:

        if sampling_strategy_if_longer == "from_start":
            truncated_tensor = tensor[:max_length]
            return truncated_tensor

        if sampling_strategy_if_longer == "uniform":
            uniformly_sampled_tensor = _sample_sequence_uniform(
                tensor=tensor, tensor_length=tensor_length, max_length=max_length
            )
            return uniformly_sampled_tensor

    right_padding = max_length - tensor_length
    padded_tensor = pad(input=tensor, pad=[0, right_padding], value=padding_value)

    return padded_tensor


def _sample_sequence_uniform(
    tensor: torch.Tensor, tensor_length: int, max_length: int
) -> torch.Tensor:
    random_index_start = torch.randperm(max(1, tensor_length - max_length))[0]
    random_index_end = random_index_start + max_length
    return tensor[random_index_start:random_index_end]


def _bytes_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    dtype: str,
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:

    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        bytes_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        ).astype(dtype=dtype)
    else:
        bytes_data = np.fromfile(file=data_pointer, dtype=dtype)

    return bytes_data


def _image_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    deeplake_inner_key: Optional[str] = None,
) -> Image:

    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        image_data = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
        pil_image = fromarray(obj=np.uint8(image_data * 255))
    else:
        pil_image = default_loader(path=str(data_pointer))

    return pil_image


def load_sequence_from_disk(sequence_file_path: Path) -> str:
    with open(sequence_file_path, "r") as infile:
        return infile.read().strip()


def prepare_inputs_memory(
    inputs: Dict[str, Any], inputs_objects: "al_input_objects_as_dict", test_mode: bool
) -> Dict[str, torch.Tensor]:
    prepared_inputs = {}

    for name, data in inputs.items():

        input_object = inputs_objects[name]

        input_type_info = input_object.input_config.input_type_info
        input_type = input_object.input_config.input_info.input_type

        if input_type == "omics":
            array_raw_in_memory = data
            array_prepared = prepare_one_hot_omics_data(
                genotype_array=array_raw_in_memory,
                na_augment_perc=input_type_info.na_augment_perc,
                na_augment_prob=input_type_info.na_augment_prob,
                test_mode=test_mode,
            )
            prepared_inputs[name] = array_prepared

        elif input_type == "sequence":
            sequence_raw_in_memory = data
            prepared_sequence_inputs = prepare_sequence_data(
                sequence_input_object=inputs_objects[name],
                cur_file_content_tokenized=sequence_raw_in_memory,
                test_mode=test_mode,
            )
            prepared_inputs[name] = prepared_sequence_inputs

        elif input_type == "bytes":
            bytes_raw_in_memory = data
            prepared_bytes_input = prepare_bytes_data(
                bytes_input_object=inputs_objects[name],
                bytes_data=bytes_raw_in_memory,
                test_mode=test_mode,
            )

            prepared_inputs[name] = prepared_bytes_input

        elif input_type == "image":
            image_raw_in_memory = data
            prepared_image_data = prepare_image_data(
                image_input_object=inputs_objects[name],
                image_data=image_raw_in_memory,
                test_mode=test_mode,
            )
            prepared_inputs[name] = prepared_image_data

        else:
            prepared_inputs[name] = inputs[name]

    return prepared_inputs


def impute_missing_modalities_wrapper(
    inputs_values: Dict[str, Any], inputs_objects: "al_input_objects_as_dict"
) -> Dict[str, torch.Tensor]:
    impute_dtypes = _get_default_impute_dtypes(inputs_objects=inputs_objects)
    impute_fill_values = _get_default_impute_fill_values(inputs_objects=inputs_objects)
    inputs_imputed = impute_missing_modalities(
        inputs_values=inputs_values,
        inputs_objects=inputs_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    return inputs_imputed


def impute_missing_modalities(
    inputs_values: Dict[str, Any],
    inputs_objects: "al_input_objects_as_dict",
    fill_values: Dict[str, Any],
    dtypes: Dict[str, Any],
) -> Dict[str, torch.Tensor]:

    for input_name, input_object in inputs_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_name not in inputs_values:
            fill_value = fill_values[input_name]
            dtype = dtypes[input_name]

            if input_type == "omics":
                dimensions = input_object.data_dimensions
                shape = dimensions.channels, dimensions.height, dimensions.width

                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "sequence":
                max_length = input_object.computed_max_length
                shape = (max_length,)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "bytes":
                max_length = input_object.input_config.input_type_info.max_length
                shape = (max_length,)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "image":
                size = input_object.input_config.input_type_info.size
                if len(size) == 1:
                    size = [size[0], size[0]]

                num_channels = input_object.num_channels
                shape = (num_channels, *size)
                imputed_tensor = impute_single_missing_modality(
                    shape=shape, fill_value=fill_value, dtype=dtype
                )
                inputs_values[input_name] = imputed_tensor

            elif input_type == "tabular":
                inputs_values[input_name] = fill_value

    return inputs_values


def impute_single_missing_modality(
    shape: Tuple[int, ...], fill_value: Any, dtype: Any
) -> torch.Tensor:
    imputed_tensor = torch.empty(shape, dtype=dtype).fill_(fill_value)
    return imputed_tensor


def _omics_load_wrapper(
    data_pointer: Union[Path, int],
    input_source: str,
    subset_indices: Union[Sequence[int], None],
    deeplake_inner_key: Optional[str] = None,
) -> np.ndarray:

    if deeplake_ops.is_deeplake_dataset(data_source=input_source):
        assert deeplake_inner_key is not None
        genotype_array_raw = _load_deeplake_sample(
            data_pointer=data_pointer,
            input_source=input_source,
            inner_key=deeplake_inner_key,
        )
    else:
        genotype_array_raw = np.load(str(data_pointer))

    if subset_indices is not None:
        genotype_array_raw = genotype_array_raw[:, subset_indices]

    genotype_array_raw_bool = genotype_array_raw.astype(bool)

    return genotype_array_raw_bool


def _load_deeplake_sample(
    data_pointer: int, input_source: str, inner_key: str
) -> np.ndarray:
    """
    Deeplake warns about indexing directly into a DS, vs. random access. For now we'll
    use this random access pattern here as we have to be able to connect to other
    data sources (which might be outside deeplake).
    """
    assert inner_key is not None
    deeplake_ds = deeplake_ops.load_deeplake_dataset(data_source=input_source)
    deeplake_ds_index = data_pointer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_data = deeplake_ds[deeplake_ds_index][inner_key].numpy()

    return sample_data
