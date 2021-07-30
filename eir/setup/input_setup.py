from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Sequence, Callable, TYPE_CHECKING

import numpy as np
from aislib.misc_utils import get_logger

from eir.data_load.label_setup import (
    Labels,
    set_up_train_and_valid_tabular_data,
    TabularFileInfo,
    get_array_path_iterator,
    save_transformer_set,
)
from eir.setup import schemas

if TYPE_CHECKING:
    from eir.train import Hooks

logger = get_logger(__name__)

al_input_objects_as_dict = Dict[str, Union["OmicsInputInfo", "TabularInputInfo"]]


def set_up_inputs_for_training(
    inputs_configs: schemas.al_input_configs,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> al_input_objects_as_dict:
    all_inputs = {}

    name_config_iter = get_input_name_config_iterator(input_configs=inputs_configs)
    for name, input_config in name_config_iter:
        cur_input_data_config = input_config.input_info
        setup_func = get_input_setup_function(
            input_type=cur_input_data_config.input_type
        )
        logger.info(
            "Setting up %s inputs '%s' from %s.",
            cur_input_data_config.input_type,
            cur_input_data_config.input_name,
            cur_input_data_config.input_source,
        )
        set_up_input = setup_func(
            input_config=input_config,
            train_ids=train_ids,
            valid_ids=valid_ids,
            hooks=hooks,
        )
        all_inputs[name] = set_up_input

    return all_inputs


def get_input_name_config_iterator(input_configs: schemas.al_input_configs):
    for input_config in input_configs:
        cur_input_data_config = input_config.input_info
        cur_name = (
            f"{cur_input_data_config.input_type}_{cur_input_data_config.input_name}"
        )
        yield cur_name, input_config


def get_input_setup_function(input_type) -> Callable:
    mapping = get_input_setup_function_map()

    return mapping[input_type]


def get_input_setup_function_map() -> Dict[str, Callable]:
    setup_mapping = {
        "omics": set_up_omics_input,
        "tabular": set_up_tabular_input_for_training,
    }

    return setup_mapping


@dataclass
class TabularInputInfo:
    labels: Labels
    input_config: schemas.InputConfig


def set_up_tabular_input_for_training(
    input_config: schemas.InputConfig,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    hooks: Union["Hooks", None],
) -> TabularInputInfo:
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
    )

    tabular_input_info = TabularInputInfo(
        labels=tabular_labels, input_config=input_config
    )

    return tabular_input_info


@dataclass
class OmicsInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"


def set_up_omics_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> OmicsInputInfo:

    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source)
    )
    omics_input_info = OmicsInputInfo(
        input_config=input_config, data_dimensions=data_dimensions
    )

    return omics_input_info


def get_tabular_input_file_info(
    input_source: str,
    tabular_data_type_config: schemas.TabularInputDataConfig,
) -> TabularFileInfo:

    table_info = TabularFileInfo(
        file_path=Path(input_source),
        con_columns=tabular_data_type_config.extra_con_columns,
        cat_columns=tabular_data_type_config.extra_cat_columns,
        parsing_chunk_size=tabular_data_type_config.label_parsing_chunk_size,
    )

    return table_info


@dataclass
class DataDimensions:
    channels: int
    height: int
    width: int

    def num_elements(self):
        return self.channels * self.height * self.width


def get_data_dimension_from_data_source(
    data_source: Path,
) -> DataDimensions:
    """
    TODO: Make more dynamic / robust. Also weird to say "width" for a 1D vector.
    """

    iterator = get_array_path_iterator(data_source=data_source)
    path = next(iterator)
    shape = np.load(file=path).shape

    if len(shape) == 1:
        channels, height, width = 1, 1, shape[0]
    elif len(shape) == 2:
        channels, height, width = 1, shape[0], shape[1]
    elif len(shape) == 3:
        channels, height, width = shape
    else:
        raise ValueError("Currently max 3 dimensional inputs supported")

    return DataDimensions(channels=channels, height=height, width=width)


def serialize_all_input_transformers(
    inputs_dict: al_input_objects_as_dict, run_folder: Path
):
    for input_name, input_ in inputs_dict.items():
        if input_name.startswith("tabular_"):
            save_transformer_set(
                transformers=input_.labels.label_transformers, run_folder=run_folder
            )
