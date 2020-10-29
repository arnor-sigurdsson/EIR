from dataclasses import dataclass
from typing import Tuple, Dict, Union, List

import torch

from snp_pred.data_load.label_setup import al_target_columns


def get_target_columns_generator(target_columns: al_target_columns) -> Tuple[str, str]:
    for column_type, list_of_cols_of_this_type in target_columns.items():
        for cur_column in list_of_cols_of_this_type:
            yield column_type, cur_column


@dataclass(frozen=True)
class Batch:
    inputs: torch.Tensor
    target_labels: Dict[str, torch.Tensor]
    extra_inputs: Union[torch.Tensor, None]
    ids: List[str]
