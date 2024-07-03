import reprlib
import typing
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Type, Union

import torch

from eir.models.model_setup_modules.meta_setup import al_meta_model
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def load_model(
    model_path: Path,
    model_class: Type[al_meta_model],
    model_init_kwargs: Dict,
    device: str,
    test_mode: bool,
    state_dict_keys_to_keep: Union[None, Sequence[str]] = None,
    state_dict_key_rename: Union[None, Sequence[Tuple[str, str]]] = None,
    strict_shapes: bool = True,
) -> al_meta_model:
    model = model_class(**model_init_kwargs)

    model = _load_model_weights(
        model=model,
        model_state_dict_path=model_path,
        device=device,
        state_dict_keys_to_keep=state_dict_keys_to_keep,
        state_dict_key_rename=state_dict_key_rename,
        strict_shapes=strict_shapes,
    )

    if test_mode:
        model.eval()

    return model


def _load_model_weights(
    model: al_meta_model,
    model_state_dict_path: Path,
    device: str,
    state_dict_keys_to_keep: Union[None, Sequence[str]] = None,
    state_dict_key_rename: Union[None, Sequence[Tuple[str, str]]] = None,
    strict_shapes: bool = True,
) -> al_meta_model:
    loaded_weights_state_dict = torch.load(model_state_dict_path, map_location=device)

    if state_dict_keys_to_keep:
        no_keys_before = len(loaded_weights_state_dict)
        loaded_weights_state_dict = _filter_state_dict_keys(
            state_dict=loaded_weights_state_dict, keys_to_keep=state_dict_keys_to_keep
        )
        logger.info(
            "Extracting %d/%d modules for feature extractors: '%s' from %s.",
            len(loaded_weights_state_dict),
            no_keys_before,
            state_dict_keys_to_keep,
            model_state_dict_path,
        )

    if state_dict_key_rename:
        for replace_tuple in state_dict_key_rename:
            logger.debug(
                "Renaming '%s' in pretrained model to '%s' in current model.",
                replace_tuple[0],
                replace_tuple[1],
            )
            loaded_weights_state_dict = _replace_dict_key_names(
                dict_=loaded_weights_state_dict, replace_pattern=replace_tuple
            )

    if not strict_shapes:
        model_state_dict = model.state_dict()
        loaded_weights_state_dict = _filter_incompatible_parameter_shapes_for_loading(
            source_state_dict=model_state_dict,
            destination_state_dict=loaded_weights_state_dict,
        )

    incompatible_keys = model.load_state_dict(
        state_dict=loaded_weights_state_dict,
        strict=False,
    )

    total_keys = len(loaded_weights_state_dict)
    no_missing = len(incompatible_keys.missing_keys)
    no_unexpected = len(incompatible_keys.unexpected_keys)
    no_incompatible_keys = no_missing + no_unexpected
    if no_incompatible_keys > 0:
        repr_object = reprlib.Repr()
        repr_object.maxother = 256
        repr_object.maxstring = 256
        logger.info(
            "Encountered incompatible modules when loading model from '%s'.\n"
            "Missing keys: \n%s\n"
            "Unexpected keys: \n%s\n"
            "This is expected if you are loading select modules from a saved model, "
            "which means you can ignore this message. If you are loading a pre-trained "
            "model as-is, then this is most likely an error and something unexpected "
            "has changed between the pre-training and setting up the current model "
            "from the pre-trained one.",
            model_state_dict_path,
            repr_object.repr(incompatible_keys.missing_keys),
            repr_object.repr(incompatible_keys.unexpected_keys),
        )

    logger.info(
        "Successfully loaded %d/%d modules from %s.",
        total_keys,
        total_keys,
        model_state_dict_path,
    )

    torch_device = torch.device(device)
    model = model.to(device=torch_device)

    return model


def _replace_dict_key_names(
    dict_: Dict[str, Any], replace_pattern: Tuple[str, str]
) -> OrderedDict:
    renamed_dict = OrderedDict()

    for key, value in dict_.items():
        new_key = key.replace(*replace_pattern)
        renamed_dict[new_key] = value

    return renamed_dict


def _filter_state_dict_keys(
    state_dict: typing.OrderedDict[str, torch.nn.Parameter], keys_to_keep: Sequence[str]
) -> typing.OrderedDict[str, torch.nn.Parameter]:
    filtered_state_dict = OrderedDict()

    for module_name, module_parameter in state_dict.items():
        if any(key in module_name for key in keys_to_keep):
            filtered_state_dict[module_name] = module_parameter

    return filtered_state_dict


def _filter_incompatible_parameter_shapes_for_loading(
    source_state_dict: Dict[str, Any], destination_state_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Note that the way this is used is a bit unclear / backwards, but the destination
    state dict is the one that contains weights loaded from the pretrained model,
    and the source state dict is the current, randomly initialized model.

    When there is a shape that is incompatible, but with the same name, we replace
    the parameter in the destination state dict with the one from the source state.
    This ensures that when we model.load_state_dict(...), we effectively maintain
    the original weights in the source model, vs. alternatively having a clash
    trying to load incompatible weights into the source model that happen to have
    the same name.
    """

    destination_state_dict = deepcopy(destination_state_dict)

    for key in destination_state_dict:
        if key in source_state_dict:
            if destination_state_dict[key].shape != source_state_dict[key].shape:
                logger.info(
                    f"Skipping loading of parameter: {key} "
                    f"due to incompatible shapes. "
                    f"Source shape: {source_state_dict[key].shape}. "
                    f"Destination shape: {destination_state_dict[key].shape}."
                )
                destination_state_dict[key] = source_state_dict[key]

    return destination_state_dict
