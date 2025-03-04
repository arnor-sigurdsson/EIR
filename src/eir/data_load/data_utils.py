from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
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
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schema_modules.output_schemas_survival import SurvivalOutputTypeConfig
from eir.train_utils.distributed import in_distributed_env
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.datasets import (
        al_local_datasets,
        al_sample_label_dict_target,
    )
    from eir.setup.output_setup import al_output_objects_as_dict

logger = get_logger(name=__name__)


def get_output_info_generator(
    outputs_as_dict: "al_output_objects_as_dict",
) -> Generator[tuple[str, str, str]]:
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
            case ComputedSurvivalOutputInfo():
                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, SurvivalOutputTypeConfig)
                event_name = output_type_info.event_column
                yield output_name, "survival", event_name
            case _:
                raise TypeError(f"Unknown output object: {output_object}")


@dataclass(frozen=True)
class Batch:
    inputs: dict[str, torch.Tensor]
    target_labels: dict[str, dict[str, torch.Tensor]]
    ids: list[str]


@overload
def get_finite_train_sampler(
    columns_to_sample: None, train_dataset: "al_local_datasets"
) -> None: ...


@overload
def get_finite_train_sampler(
    columns_to_sample: Sequence[str],
    train_dataset: "al_local_datasets",
) -> WeightedRandomSampler | DistributedSampler: ...


def get_finite_train_sampler(columns_to_sample, train_dataset):
    in_distributed_run = in_distributed_env()

    if columns_to_sample is None:
        return None

    if in_distributed_run:
        raise NotImplementedError(
            "Weighted sampling not yet implemented for distributed training."
        )

    logger.debug("Weighted sampling enabled, setting up.")

    loaded_target_columns = _gather_all_loaded_columns(outputs=train_dataset.outputs)

    is_sample_column_loaded = False
    is_sample_all_cols = False

    if columns_to_sample == ["all"]:
        is_sample_all_cols = True
    else:
        parsed_columns_to_sample = {i.split("__", 1)[1] for i in columns_to_sample}
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
                output_name + "__" + i for i in output_object.target_columns["cat"]
            ]
            columns_to_sample += cat_columns_with_output_prefix

    train_sampler = get_weighted_random_sampler(
        target_storage=train_dataset.target_labels_storage,
        columns_to_sample=columns_to_sample,
    )
    return train_sampler


def _gather_all_loaded_columns(outputs: "al_output_objects_as_dict") -> Sequence[str]:
    loaded_cat_columns: list[str] = []
    loaded_con_columns: list[str] = []

    for _output_name, output_object in outputs.items():
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


@dataclass
class Sample:
    sample_id: str
    inputs: dict[str, Any]
    target_labels: "al_sample_label_dict_target"


def consistent_nan_collate(batch):
    """
    Sometimes, if we have a mixed batch with NaN and float32 values, it can
    happen that the first element is NaN. Then, PyTorch default_collate uses
    that to determine the dtype, and the full thing will be cast to float64.
    Generally, this is OK, but e.g. on MPS devices, this will raise an error
    as float64 is not supported on MPS.

    Hence, we enforce a float64 -> float32 conversion.
    """

    result = torch.utils.data.default_collate(batch)

    def ensure_float32(obj):
        if isinstance(obj, torch.Tensor) and obj.dtype == torch.float64:
            return obj.to(torch.float32)
        elif isinstance(obj, dict):
            return {k: ensure_float32(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return type(obj)(ensure_float32(x) for x in obj)
        return obj

    final = ensure_float32(result)

    return final
