from torch import nn

from eir.models.input.array.array_models import (
    ArrayModelConfig,
    ArrayWrapperModel,
    al_array_model_configs,
    al_array_model_types,
    al_array_models,
    get_array_model_class,
    get_array_model_init_kwargs,
)
from eir.setup.input_setup_modules.common import DataDimensions


def get_array_model(
    array_feature_extractor: al_array_models, model_config: ArrayModelConfig
) -> nn.Module:
    wrapper_model = ArrayWrapperModel(
        feature_extractor=array_feature_extractor,
        normalization=model_config.pre_normalization,
    )

    return wrapper_model


def get_array_input_feature_extractor(
    model_init_config: al_array_model_configs,
    data_dimensions: DataDimensions,
    model_type: al_array_model_types,
) -> al_array_models:
    array_model_class = get_array_model_class(model_type=model_type)
    model_init_kwargs = get_array_model_init_kwargs(
        model_type=model_type,
        model_config=model_init_config,
        data_dimensions=data_dimensions,
    )

    array_model = array_model_class(**model_init_kwargs)  # type: ignore
    return array_model
