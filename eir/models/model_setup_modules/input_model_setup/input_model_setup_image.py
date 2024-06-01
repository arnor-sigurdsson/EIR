from copy import copy
from typing import Any, Dict, Union

import timm
import torch
from timm.models import CoaT, CrossVit
from torch import nn

from eir.models.input.array.models_cnn import CNNModel, CNNModelConfig
from eir.models.input.image.image_models import (
    ImageModelClassGetterFunction,
    ImageModelConfig,
    ImageWrapperModel,
)
from eir.setup.input_setup_modules.common import DataDimensions
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def get_image_model(
    model_config: ImageModelConfig,
    data_dimensions: DataDimensions,
    model_registry_lookup: ImageModelClassGetterFunction,
    device: str,
) -> ImageWrapperModel:
    input_channels = data_dimensions.channels
    if model_config.model_type in _get_all_timm_models():
        feature_extractor = timm.create_model(
            model_name=model_config.model_type,
            pretrained=model_config.pretrained_model,
            num_classes=model_config.num_output_features,
            in_chans=input_channels,
        )

    elif model_config.model_type == "cnn":
        init_kwargs = _parse_init_kwargs(model_config=model_config)
        cnn_model_config = CNNModelConfig(**init_kwargs)

        feature_extractor = CNNModel(
            model_config=cnn_model_config,
            data_dimensions=data_dimensions,
        )

    else:
        assert isinstance(model_config.model_init_config, dict)
        feature_extractor = _meta_get_image_model_from_scratch(
            model_type=model_config.model_type,
            model_init_config=model_config.model_init_config,
            in_chans=input_channels,
            num_output_features=model_config.num_output_features,
        )

    feature_extractor = feature_extractor.to(device=device)

    _check_image_model_num_output_features_compatibility(
        feature_extractor=feature_extractor,
        num_output_features=model_config.num_output_features,
    )

    estimated_out_shape = estimate_feature_extractor_out_shape(
        feature_extractor=feature_extractor,
        data_dimensions=data_dimensions,
        num_output_features=model_config.num_output_features,
        device=device,
    )

    wrapper_model_class = model_registry_lookup(model_type="image-wrapper-default")
    model = wrapper_model_class(
        feature_extractor=feature_extractor,
        model_config=model_config,
        estimated_out_shape=estimated_out_shape,
    )

    if model_config.freeze_pretrained_model:
        for param in model.parameters():
            param.requires_grad = False

    return model


def _check_image_model_num_output_features_compatibility(
    feature_extractor: nn.Module, num_output_features: int
) -> None:
    match feature_extractor, num_output_features:
        case CrossVit(), 0:
            raise ValueError(
                "CrossVit model requires num_output_features to be set to a value "
                "greater than 0."
            )
        case CoaT(), 0:
            raise ValueError(
                "CoaT model requires num_output_features to be set to a value "
                "greater than 0."
            )


def estimate_feature_extractor_out_shape(
    feature_extractor: nn.Module,
    data_dimensions: DataDimensions,
    num_output_features: int,
    device: str,
) -> tuple[int, ...]:
    example_input = prepare_example_image_test_input(
        data_dimensions=data_dimensions,
        device=device,
    )

    has_forward_features = hasattr(feature_extractor, "forward_features")
    has_num_out_features = num_output_features > 0

    with torch.inference_mode():
        if has_forward_features and not has_num_out_features:
            feature_extractor_out = feature_extractor.forward_features(example_input)
        else:
            feature_extractor_out = feature_extractor(example_input)

    estimated_shape = feature_extractor_out.shape[1:]
    logger.debug("Estimated shape of feature extractor output: %s", estimated_shape)

    return estimated_shape


def prepare_example_image_test_input(
    data_dimensions: DataDimensions,
    device: str,
    batch_size: int = 2,
) -> torch.Tensor:
    full_shape = data_dimensions.full_shape()
    example_input = torch.rand((batch_size, *full_shape)).to(device=device)
    return example_input


def _get_all_timm_models() -> list[str]:
    base_models = timm.list_models()
    pretrained_models = timm.list_models(pretrained=True)

    all_models = base_models + pretrained_models
    all_models = list(set(all_models))
    all_models.sort()

    return all_models


def _parse_init_kwargs(model_config: ImageModelConfig) -> Dict[str, Any]:
    init_config = copy(model_config.model_init_config)
    if isinstance(init_config, dict):
        init_kwargs = init_config
    else:
        init_kwargs = init_config.__dict__
    assert isinstance(init_kwargs, dict)

    model_config_out_features = getattr(model_config, "num_output_features")
    init_kwargs_out_features = init_kwargs.get("num_output_features")
    if model_config_out_features:
        if model_config_out_features != init_kwargs_out_features:
            logger.warning(
                "num_output_features specified in model_config "
                "and model_init_config differ. "
                "Using value from model_config."
            )

            init_kwargs["num_output_features"] = model_config.num_output_features

    return init_kwargs


def _meta_get_image_model_from_scratch(
    model_type: str,
    model_init_config: dict,
    in_chans: int,
    num_output_features: Union[None, int],
) -> nn.Module:
    """
    A kind of ridiculous way to initialize modules from scratch that are found in timm,
    but could not find a better way at first glance.
    """

    feature_extractor_model_config = copy(model_init_config)
    feature_extractor_model_config["in_chans"] = in_chans
    if num_output_features:
        feature_extractor_model_config["num_classes"] = num_output_features

    logger.info(
        "Model '%s' not found among pretrained/external image model names, assuming "
        "module will be initialized from scratch using %s for initialization.",
        model_type,
        model_init_config,
    )

    feature_extractor_class = getattr(timm.models, model_type)
    parent_module = getattr(
        timm.models, feature_extractor_class.__module__.split(".")[-1]
    )
    found_modules = {
        k: getattr(parent_module, v)
        for k, v in feature_extractor_model_config.items()
        if isinstance(v, str) and getattr(parent_module, v, None)
    }
    feature_extractor_model_config_with_meta = {
        **feature_extractor_model_config,
        **found_modules,
    }

    feature_extractor = feature_extractor_class(
        **feature_extractor_model_config_with_meta
    )

    return feature_extractor
