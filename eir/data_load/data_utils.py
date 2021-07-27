from dataclasses import dataclass
from typing import Tuple, Dict, List, overload, TYPE_CHECKING

import torch
from torch.utils.data import WeightedRandomSampler

from eir.data_load.data_loading_funcs import get_weighted_random_sampler
from eir.data_load.label_setup import al_target_columns

if TYPE_CHECKING:
    from eir.data_load.datasets import DatasetBase


def get_target_columns_generator(target_columns: al_target_columns) -> Tuple[str, str]:
    for column_type, list_of_cols_of_this_type in target_columns.items():
        for cur_column in list_of_cols_of_this_type:
            yield column_type, cur_column


@dataclass(frozen=True)
class Batch:
    inputs: Dict[str, torch.Tensor]
    target_labels: Dict[str, torch.Tensor]
    ids: List[str]


@overload
def get_train_sampler(columns_to_sample: None, train_dataset: "DatasetBase") -> None:
    ...


@overload
def get_train_sampler(
    columns_to_sample: List[str], train_dataset: "DatasetBase"
) -> WeightedRandomSampler:
    ...


def get_train_sampler(columns_to_sample, train_dataset):
    """
    TODO:   Refactor, remove dependency on train_dataset and use instead
            Iterable[Samples], and target_columns directly.
    """
    if columns_to_sample is None:
        return None

    loaded_target_columns = (
        train_dataset.target_columns["con"] + train_dataset.target_columns["cat"]
    )

    is_sample_column_loaded = set(columns_to_sample).issubset(
        set(loaded_target_columns)
    )
    is_sample_all_cols = columns_to_sample == ["all"]

    if not is_sample_column_loaded and not is_sample_all_cols:
        raise ValueError(
            "Weighted sampling from non-loaded columns not supported yet "
            f"(could not find {columns_to_sample})."
        )

    if is_sample_all_cols:
        columns_to_sample = train_dataset.target_columns["cat"]

    train_sampler = get_weighted_random_sampler(
        samples=train_dataset.samples, target_columns=columns_to_sample
    )
    return train_sampler
