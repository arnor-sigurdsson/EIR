import argparse
from dataclasses import _DataclassParams
from typing import Union, Type, Dict, Any, Protocol, TYPE_CHECKING

from eir.models.omics.models_cnn import CNNModel, CNNModelConfig
from eir.models.omics.models_identity import IdentityModel, IdentityModelConfig
from eir.models.omics.models_linear import LinearModel, LinearModelConfig
from eir.models.omics.models_locally_connected import (
    SimpleLCLModel,
    SimpleLCLModelConfig,
    LCLModel,
    LCLModelConfig,
)

if TYPE_CHECKING:
    from eir.setup.input_setup import DataDimensions

al_models_classes = Union[
    Type["CNNModel"],
    Type["LinearModel"],
    Type["SimpleLCLModel"],
    Type["LCLModel"],
    Type["IdentityModel"],
]

al_models = Union[
    "CNNModel", "LinearModel", "SimpleLCLModel", "LCLModel", "IdentityModel"
]

al_omics_model_configs = Union[
    CNNModelConfig,
    LinearModelConfig,
    SimpleLCLModelConfig,
    LCLModelConfig,
    IdentityModelConfig,
]


def get_omics_model_mapping() -> Dict[str, al_models_classes]:
    mapping = {
        "cnn": CNNModel,
        "linear": LinearModel,
        "mlp-split": SimpleLCLModel,
        "genome-local-net": LCLModel,
        "identity": IdentityModel,
    }

    return mapping


def get_model_class(model_type: str) -> al_models_classes:
    mapping = get_omics_model_mapping()
    return mapping[model_type]


class Dataclass(Protocol):
    __dataclass_fields__: Dict
    __dataclass_params__: _DataclassParams


def get_omics_config_dataclass_mapping() -> Dict[str, Type[Dataclass]]:
    mapping = {
        "cnn": CNNModelConfig,
        "linear": LinearModelConfig,
        "mlp-split": SimpleLCLModelConfig,
        "genome-local-net": LCLModelConfig,
        "identity": IdentityModelConfig,
    }

    return mapping


def get_model_config_dataclass(model_type: str) -> Type[Dataclass]:
    mapping = get_omics_config_dataclass_mapping()
    return mapping[model_type]


def get_omics_model_init_kwargs(
    model_type: str,
    model_config: al_omics_model_configs,
    data_dimensions: "DataDimensions",
) -> Dict[str, Any]:
    """
    See: https://github.com/python/mypy/issues/5374 for type hint issue.

    Possibly split / extend this function later to account for other kwargs that just
    model_config, to allow for more flexibility in model instantiation (not restricting
    to just model_config object).
    """

    kwargs = {}

    model_config_dataclass = get_model_config_dataclass(model_type=model_type)
    dataclass_instance = model_config_dataclass(**model_config.__dict__)

    kwargs["model_config"] = dataclass_instance
    kwargs["data_dimensions"] = data_dimensions

    return kwargs


def match_namespace_to_dataclass(
    namespace: argparse.Namespace, data_class: Type[Dataclass]
) -> Dict[str, Any]:
    dataclass_kwargs = {}
    field_names = data_class.__dataclass_fields__.keys()

    for field_name in field_names:
        if hasattr(namespace, field_name):
            dataclass_kwargs[field_name] = getattr(namespace, field_name)

    return dataclass_kwargs
