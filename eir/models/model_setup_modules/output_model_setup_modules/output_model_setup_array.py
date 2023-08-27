from functools import partial
from typing import Dict, Optional, Union

import torch
from aislib.misc_utils import get_logger

from eir.models.input.array.models_locally_connected import FlattenFunc, LCLModel
from eir.models.model_setup_modules.output_model_setup_modules import al_output_modules
from eir.models.output.array.array_output_modules import (
    ArrayOutputModuleConfig,
    CNNUpscaleModel,
    CNNUpscaleModelConfig,
    LCLOutputModelConfig,
    al_array_model_types,
    al_output_array_model_classes,
    al_output_array_model_config_classes,
    al_output_array_models,
    get_array_output_module,
)
from eir.setup.input_setup_modules.common import DataDimensions
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo

al_output_array_model_init_kwargs = dict[
    str, Union["DataDimensions", LCLOutputModelConfig, FlattenFunc, int]
]

al_output_array_model_configs = LCLOutputModelConfig

logger = get_logger(name=__name__)


def get_array_output_module_from_model_config(
    output_object: ComputedArrayOutputInfo,
    input_dimension: int,
    device: str,
) -> al_output_modules:
    output_model_config = output_object.output_config.model_config
    assert isinstance(output_model_config, ArrayOutputModuleConfig)

    output_module_type = output_model_config.model_type
    output_name = output_object.output_config.output_info.output_name

    output_module: al_output_modules

    input_data_dimension = DataDimensions(
        channels=1,
        height=1,
        width=input_dimension,
    )

    feature_extractor = get_array_output_feature_extractor(
        model_init_config=output_model_config.model_init_config,
        input_data_dimensions=input_data_dimension,
        model_type=output_module_type,
        output_data_dimensions=output_object.data_dimensions,
    )

    array_output_module = get_array_output_module(
        feature_extractor=feature_extractor,
        output_name=output_name,
        target_data_dimensions=output_object.data_dimensions,
    )

    torch_device = torch.device(device=device)
    output_module = array_output_module.to(device=torch_device)

    return output_module


def get_array_output_feature_extractor(
    model_init_config: al_output_array_model_configs,
    input_data_dimensions: DataDimensions,
    model_type: al_array_model_types,
    output_data_dimensions: Optional[DataDimensions],
) -> al_output_array_models:
    array_model_class = get_array_output_model_class(model_type=model_type)
    model_init_kwargs = get_array_output_model_init_kwargs(
        model_type=model_type,
        model_config=model_init_config,
        input_data_dimensions=input_data_dimensions,
        output_data_dimensions=output_data_dimensions,
    )

    array_model = array_model_class(**model_init_kwargs)  # type: ignore
    return array_model


def get_array_output_model_mapping() -> Dict[str, al_output_array_model_classes]:
    mapping = {
        "lcl": LCLModel,
        "cnn": CNNUpscaleModel,
    }

    return mapping


def get_array_output_model_class(
    model_type: al_array_model_types,
) -> al_output_array_model_classes:
    mapping = get_array_output_model_mapping()
    return mapping[model_type]


def get_array_output_config_dataclass_mapping() -> (
    Dict[str, al_output_array_model_config_classes]
):
    mapping = {
        "lcl": LCLOutputModelConfig,
        "cnn": CNNUpscaleModelConfig,
    }

    return mapping


def get_array_output_model_config_dataclass(
    model_type: str,
) -> al_output_array_model_config_classes:
    mapping = get_array_output_config_dataclass_mapping()
    return mapping[model_type]


def get_array_output_model_init_kwargs(
    model_type: al_array_model_types,
    model_config: al_output_array_model_configs,
    input_data_dimensions: DataDimensions,
    output_data_dimensions: Optional[DataDimensions],
) -> dict[
    str,
    Union[
        DataDimensions, LCLOutputModelConfig | CNNUpscaleModelConfig, FlattenFunc, int
    ],
]:
    kwargs: dict[
        str,
        Union[
            DataDimensions,
            LCLOutputModelConfig | CNNUpscaleModelConfig,
            FlattenFunc,
            int,
        ],
    ] = {}

    model_config_dataclass = get_array_output_model_config_dataclass(
        model_type=model_type
    )
    model_config_dataclass_instance = model_config_dataclass(**model_config.__dict__)

    kwargs["model_config"] = model_config_dataclass_instance
    kwargs["data_dimensions"] = input_data_dimensions

    match model_type:
        case "lcl":
            assert isinstance(model_config, LCLOutputModelConfig)
            kwargs["flatten_fn"] = partial(torch.flatten, start_dim=1)

            if model_config.cutoff == "auto":
                assert output_data_dimensions is not None
                num_elements = output_data_dimensions.num_elements()
                logger.debug(
                    "Setting dynamic cutoff to %s for LCL array output module.",
                    num_elements,
                )
                kwargs["dynamic_cutoff"] = num_elements

        case "cnn":
            assert isinstance(model_config, CNNUpscaleModelConfig)
            assert output_data_dimensions is not None
            kwargs["target_dimensions"] = output_data_dimensions

    return kwargs
