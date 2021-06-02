from dataclasses import dataclass
from typing import Tuple, Dict, List

import torch

from eir.data_load.label_setup import al_target_columns


def get_target_columns_generator(target_columns: al_target_columns) -> Tuple[str, str]:
    for column_type, list_of_cols_of_this_type in target_columns.items():
        for cur_column in list_of_cols_of_this_type:
            yield column_type, cur_column


@dataclass(frozen=True)
class Batch:
    inputs: Dict[str, torch.Tensor]
    target_labels: Dict[str, torch.Tensor]
    ids: List[str]
