from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, Dict, TYPE_CHECKING

from eir.data_load.label_setup import (
    Labels,
    set_up_train_and_valid_tabular_data,
    TabularFileInfo,
)
from eir.models.tabular.tabular import get_unique_values_from_transformers
from eir.setup import schemas

if TYPE_CHECKING:
    from eir.train_utils.step_logic import Hooks


@dataclass
class ComputedTabularInputInfo:
    labels: Labels
    input_config: schemas.InputConfig
    total_num_features: int


def set_up_tabular_input_for_training(
    input_config: schemas.InputConfig,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> ComputedTabularInputInfo:
    tabular_file_info = get_tabular_input_file_info(
        input_source=input_config.input_info.input_source,
        tabular_data_type_config=input_config.input_type_info,
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    tabular_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_file_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
        include_missing=True,
    )

    total_num_features = get_tabular_num_features(
        label_transformers=tabular_labels.label_transformers,
        cat_columns=input_config.input_type_info.input_cat_columns,
        con_columns=input_config.input_type_info.input_con_columns,
    )
    tabular_input_info = ComputedTabularInputInfo(
        labels=tabular_labels,
        input_config=input_config,
        total_num_features=total_num_features,
    )

    return tabular_input_info


def get_tabular_input_file_info(
    input_source: str,
    tabular_data_type_config: schemas.TabularInputDataConfig,
) -> TabularFileInfo:
    table_info = TabularFileInfo(
        file_path=Path(input_source),
        con_columns=tabular_data_type_config.input_con_columns,
        cat_columns=tabular_data_type_config.input_cat_columns,
        parsing_chunk_size=tabular_data_type_config.label_parsing_chunk_size,
    )

    return table_info


def get_tabular_num_features(
    label_transformers: Dict, cat_columns: Sequence[str], con_columns: Sequence[str]
) -> int:
    unique_cat_values = get_unique_values_from_transformers(
        transformers=label_transformers,
        keys_to_use=cat_columns,
    )
    cat_num_features = sum(
        len(unique_values) for unique_values in unique_cat_values.values()
    )
    total_num_features = cat_num_features + len(con_columns)

    return total_num_features
