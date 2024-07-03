from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from eir.data_load.data_loading_funcs import get_weighted_random_sampler
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.train_utils.distributed import in_distributed_env

if TYPE_CHECKING:
    from eir.data_load.datasets import DatasetBase
    from eir.setup.output_setup import al_output_objects_as_dict


def get_output_info_generator(
    outputs_as_dict: "al_output_objects_as_dict",
) -> Generator[Tuple[str, str, str], None, None]:
    for output_name, output_object in outputs_as_dict.items():
        match output_object:
            case ComputedTabularOutputInfo():
                target_columns = output_object.target_columns
                for column_type, list_of_cols_of_this_type in target_columns.items():
                    for cur_column in list_of_cols_of_this_type:
                        yield output_name, column_type, cur_column
            case (
                ComputedSequenceOutputInfo()
                | ComputedArrayOutputInfo()
                | ComputedImageOutputInfo()
            ):
                yield output_name, "general", output_name
            case _:
                raise TypeError(f"Unknown output object: {output_object}")


@dataclass(frozen=True)
class Batch:
    inputs: Dict[str, torch.Tensor]
    target_labels: Dict[str, Dict[str, torch.Tensor]]
    ids: List[str]


@overload
def get_train_sampler(
    columns_to_sample: None, train_dataset: "DatasetBase"
) -> None: ...


@overload
def get_train_sampler(
    columns_to_sample: Sequence[str], train_dataset: "DatasetBase"
) -> Union[WeightedRandomSampler, DistributedSampler]: ...


def get_train_sampler(columns_to_sample, train_dataset):
    """
    TODO:   Refactor, remove dependency on train_dataset and use instead
            Iterable[Samples], and outputs directly.
    """
    in_distributed_run = in_distributed_env()

    if columns_to_sample is None:
        if in_distributed_run:
            return DistributedSampler(dataset=train_dataset)
        else:
            return None

    if in_distributed_run:
        raise NotImplementedError(
            "Weighted sampling not yet implemented for distributed training."
        )

    loaded_target_columns = _gather_all_loaded_columns(outputs=train_dataset.outputs)

    is_sample_column_loaded = False
    is_sample_all_cols = False

    if columns_to_sample == ["all"]:
        is_sample_all_cols = True
    else:
        parsed_columns_to_sample = set(i.split(".", 1)[1] for i in columns_to_sample)
        is_sample_column_loaded = parsed_columns_to_sample.issubset(
            set(loaded_target_columns)
        )

    if not is_sample_column_loaded and not is_sample_all_cols:
        raise ValueError(
            "Weighted sampling from non-loaded columns not supported yet "
            f"(could not find {columns_to_sample})."
        )

    if is_sample_all_cols:
        columns_to_sample = []
        for output_name, output_object in train_dataset.outputs.items():
            if output_object.output_config.output_info.output_type != "tabular":
                continue

            cat_columns_with_output_prefix = [
                output_name + "." + i for i in output_object.target_columns["cat"]
            ]
            columns_to_sample += cat_columns_with_output_prefix

    train_sampler = get_weighted_random_sampler(
        samples=train_dataset.samples, columns_to_sample=columns_to_sample
    )
    return train_sampler


def _gather_all_loaded_columns(outputs: "al_output_objects_as_dict") -> Sequence[str]:
    loaded_cat_columns: list[str] = []
    loaded_con_columns: list[str] = []

    for output_name, output_object in outputs.items():
        match output_object:
            case ComputedTabularOutputInfo():
                cur_target_columns = output_object.target_columns
                loaded_cat_columns += cur_target_columns.get("cat", [])
                loaded_con_columns += cur_target_columns.get("con", [])
            case ComputedSequenceOutputInfo():
                pass
            case _:
                raise TypeError(f"Unknown output object: {output_object}")

    loaded_target_columns = loaded_cat_columns + loaded_con_columns

    return loaded_target_columns
