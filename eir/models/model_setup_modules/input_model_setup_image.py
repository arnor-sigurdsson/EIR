from copy import copy
from typing import Callable, Type, Dict, Union

import timm
from aislib.misc_utils import get_logger
from torch import nn

from eir.models.image.image_models import ImageModelConfig, ImageWrapperModel

logger = get_logger(name=__name__)


def get_image_model(
    model_config: ImageModelConfig,
    input_channels: int,
    model_registry_lookup: Callable[[str], Type[nn.Module]],
    device: str,
) -> ImageWrapperModel:
    if model_config.model_type in timm.list_models():
        feature_extractor = timm.create_model(
            model_name=model_config.model_type,
            pretrained=model_config.pretrained_model,
            num_classes=model_config.num_output_features,
            in_chans=input_channels,
        ).to(device=device)

    else:
        feature_extractor = _meta_get_image_model_from_scratch(
            model_type=model_config.model_type,
            model_init_config=model_config.model_init_config,
            num_output_features=model_config.num_output_features,
        ).to(device=device)

    wrapper_model_class = model_registry_lookup(model_type="image-wrapper-default")
    model = wrapper_model_class(
        feature_extractor=feature_extractor, model_config=model_config
    )

    if model_config.freeze_pretrained_model:
        for param in model.parameters():
            param.requires_grad = False

    return model


def _meta_get_image_model_from_scratch(
    model_type: str, model_init_config: Dict, num_output_features: Union[None, int]
) -> nn.Module:
    """
    A kind of ridiculous way to initialize modules from scratch that are found in timm,
    but could not find a better way at first glance given how timm is set up.
    """

    feature_extractor_model_config = copy(model_init_config)
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
