from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence, Union

from eir.data_load.label_setup import (
    Labels,
    TabularFileInfo,
    save_transformer_set,
    set_up_train_and_valid_tabular_data,
)
from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    load_transformers,
)
from eir.models.input.tabular.tabular import get_unique_values_from_transformers
from eir.setup import schemas

if TYPE_CHECKING:
    from eir.setup.input_setup import al_input_objects_as_dict
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
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.TabularInputDataConfig)

    tabular_file_info = get_tabular_input_file_info(
        input_source=input_config.input_info.input_source,
        tabular_data_type_config=input_type_info,
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    tabular_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_file_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
        impute_missing=True,
    )

    total_num_features = get_tabular_num_features(
        label_transformers=tabular_labels.label_transformers,
        cat_columns=list(input_type_info.input_cat_columns),
        con_columns=list(input_type_info.input_con_columns),
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
    label_transformers: Dict, cat_columns: list[str], con_columns: list[str]
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


def set_up_tabular_input_from_pretrained(
    input_config: schemas.InputConfig,
    custom_input_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> "ComputedTabularInputInfo":
    tabular_input_object = set_up_tabular_input_for_training(
        input_config=input_config, train_ids=train_ids, valid_ids=valid_ids, hooks=hooks
    )

    pretrained_config = input_config.pretrained_config
    assert pretrained_config is not None
    pretrained_run_folder = get_run_folder_from_model_path(
        model_path=pretrained_config.model_path
    )

    loaded_transformers = load_transformers(
        run_folder=pretrained_run_folder, transformers_to_load=None
    )
    loaded_transformers_input = loaded_transformers[custom_input_name]

    tabular_input_object.labels.label_transformers = loaded_transformers_input

    return tabular_input_object


def serialize_all_input_transformers(
    inputs_dict: "al_input_objects_as_dict", run_folder: Path
):
    for input_name, input_object in inputs_dict.items():
        match input_object:
            case ComputedTabularInputInfo(labels, _, _):
                transformers_per_source = {input_name: labels.label_transformers}
                save_transformer_set(
                    transformers_per_source=transformers_per_source,
                    run_folder=run_folder,
                )
