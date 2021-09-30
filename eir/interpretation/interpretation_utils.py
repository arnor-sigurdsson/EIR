from typing import Union

import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_target_class_name(
    sample_label: torch.Tensor,
    target_transformer: Union[StandardScaler, LabelEncoder],
    column_type: str,
    target_column_name: str,
):
    if column_type == "con":
        return target_column_name

    tt_it = target_transformer.inverse_transform
    cur_trn_label = tt_it([sample_label.item()])[0]

    return cur_trn_label
