from functools import partial
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch

from eir.models.input.array.models_locally_connected import FlattenFunc, LCLModel
from eir.models.model_setup_modules.output_model_setup_modules import al_output_modules
from eir.models.output.array.array_output_modules import (
    ArrayOutputModuleConfig,
    CNNPassThroughUpscaleModel,
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
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

al_output_array_model_init_kwargs = dict[
    str,
    Union[
        Optional[DataDimensions],
        LCLOutputModelConfig | CNNUpscaleModelConfig,
        dict[str, "FeatureExtractorInfo"],
        FlattenFunc,
        int,
        str,
    ],
]

al_output_array_model_configs = LCLOutputModelConfig | CNNUpscaleModelConfig

logger = get_logger(name=__name__)


def get_array_or_image_output_module_from_model_config(
    output_object: ComputedArrayOutputInfo | ComputedImageOutputInfo,
    input_dimension: Optional[int],
    fusion_model_type: str,
    feature_extractor_infos: Optional[dict[str, "FeatureExtractorInfo"]],
    device: str,
) -> al_output_modules:
    output_model_config = output_object.output_config.model_config
    assert isinstance(output_model_config, ArrayOutputModuleConfig)

    output_module_type = output_model_config.model_type
    output_name = output_object.output_config.output_info.output_name

    output_module: al_output_modules

    if input_dimension is None:
        input_data_dimension = None
    else:
        input_data_dimension = DataDimensions(
            channels=1,
            height=1,
            width=input_dimension,
        )

    is_diffusion = output_object.diffusion_config is not None
    is_pass_through_fusion = fusion_model_type == "pass-through"
    use_passthrough = is_diffusion or is_pass_through_fusion

    diffusion_time_steps = None
    if is_diffusion:
        assert output_object.diffusion_config is not None
        diffusion_time_steps = output_object.diffusion_config.time_steps

    feature_extractor = get_array_output_feature_extractor(
        model_init_config=output_model_config.model_init_config,
        input_data_dimensions=input_data_dimension,
        feature_extractor_infos=feature_extractor_infos,
        model_type=output_module_type,
        output_data_dimensions=output_object.data_dimensions,
        output_name=output_name,
        use_passthrough=use_passthrough,
        diffusion_time_steps=diffusion_time_steps,
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
    input_data_dimensions: Optional[DataDimensions],
    feature_extractor_infos: Optional[dict[str, "FeatureExtractorInfo"]],
    model_type: al_array_model_types,
    output_data_dimensions: Optional[DataDimensions],
    output_name: str,
    use_passthrough: bool,
    diffusion_time_steps: Optional[int],
) -> al_output_array_models:

    model_type = parse_model_type(
        model_type=model_type,
        use_passthrough=use_passthrough,
    )

    array_model_class = get_array_output_model_class(model_type=model_type)
    model_init_kwargs = get_array_output_model_init_kwargs(
        model_type=model_type,
        model_config=model_init_config,
        fused_input_data_dimensions=input_data_dimensions,
        feature_extractor_infos=feature_extractor_infos,
        output_data_dimensions=output_data_dimensions,
        output_name=output_name,
        diffusion_time_steps=diffusion_time_steps,
    )

    array_model = array_model_class(**model_init_kwargs)  # type: ignore
    return array_model


def parse_model_type(
    model_type: al_array_model_types, use_passthrough: bool
) -> al_array_model_types:
    if model_type == "cnn" and use_passthrough:
        return "cnn-passthrough"
    return model_type


def get_array_output_model_mapping() -> Dict[str, al_output_array_model_classes]:
    mapping = {
        "lcl": LCLModel,
        "cnn": CNNUpscaleModel,
        "cnn-passthrough": CNNPassThroughUpscaleModel,
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
        "cnn-passthrough": CNNUpscaleModelConfig,
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
    fused_input_data_dimensions: Optional[DataDimensions],
    feature_extractor_infos: Optional[dict[str, "FeatureExtractorInfo"]],
    output_data_dimensions: Optional[DataDimensions],
    output_name: str,
    diffusion_time_steps: Optional[int],
) -> al_output_array_model_init_kwargs:
    kwargs: al_output_array_model_init_kwargs = {}

    model_config_dataclass = get_array_output_model_config_dataclass(
        model_type=model_type
    )
    model_config_dataclass_instance = model_config_dataclass(**model_config.__dict__)

    kwargs["model_config"] = model_config_dataclass_instance

    match model_type:
        case "lcl":
            assert isinstance(model_config, LCLOutputModelConfig)
            assert fused_input_data_dimensions is not None
            kwargs["data_dimensions"] = fused_input_data_dimensions
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
            kwargs["data_dimensions"] = fused_input_data_dimensions
        case "cnn-passthrough":
            assert isinstance(model_config, CNNUpscaleModelConfig)
            assert feature_extractor_infos is not None
            kwargs["target_dimensions"] = output_data_dimensions
            kwargs["feature_extractor_infos"] = feature_extractor_infos
            kwargs["output_name"] = output_name

            if diffusion_time_steps:
                kwargs["diffusion_time_steps"] = diffusion_time_steps

    return kwargs
