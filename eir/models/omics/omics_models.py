import argparse
from argparse import Namespace
from dataclasses import _DataclassParams
from typing import Union, Type, Dict, Any, Protocol, TYPE_CHECKING

from eir.models.omics.models_cnn import CNNModel, CNNModelConfig
from eir.models.omics.models_mlp import MLPModel, MLPModelConfig
from eir.models.omics.models_split_mlp import (
    SplitMLPModel,
    FullySplitMLPModel,
    SplitMLPModelConfig,
    FullySplitMLPModelConfig,
)

if TYPE_CHECKING:
    from eir.train import DataDimensions


al_models_classes = Union[
    Type["CNNModel"],
    Type["MLPModel"],
    Type["SplitMLPModel"],
    Type["FullySplitMLPModel"],
]

al_models = Union[
    "CNNModel",
    "MLPModel",
    "SplitMLPModel",
    "FullySplitMLPModel",
]


def _get_model_mapping() -> Dict[str, al_models_classes]:
    mapping = {
        "cnn": CNNModel,
        "mlp": MLPModel,
        "mlp-split": SplitMLPModel,
        "genome-local-net": FullySplitMLPModel,
    }

    return mapping


def get_model_class(model_type: str) -> al_models_classes:
    mapping = _get_model_mapping()
    return mapping[model_type]


class Dataclass(Protocol):
    __dataclass_fields__: Dict
    __dataclass_params__: _DataclassParams


def _get_dataclass_mapping() -> Dict[str, Type[Dataclass]]:
    mapping = {
        "cnn": CNNModelConfig,
        "mlp": MLPModelConfig,
        "mlp-split": SplitMLPModelConfig,
        "genome-local-net": FullySplitMLPModelConfig,
    }

    return mapping


def get_model_config_dataclass(model_type: str) -> Type[Dataclass]:
    mapping = _get_dataclass_mapping()
    return mapping[model_type]


def get_omics_model_init_kwargs(
    model_type: str, cl_args: Namespace, data_dimensions: "DataDimensions"
) -> Dict[str, Any]:
    """
    See: https://github.com/python/mypy/issues/5374 for type hint issue.

    Possibly split / extend this function later to account for other kwargs that just
    model_config, to allow for more flexibility in model instantiation (not restricting
    to just model_config object).
    """

    kwargs = {}

    model_config_dataclass = get_model_config_dataclass(model_type=model_type)
    model_config_dataclass_kwargs = match_namespace_to_dataclass(
        namespace=cl_args, data_class=model_config_dataclass
    )

    if "data_dimensions" in model_config_dataclass.__dataclass_fields__.keys():
        model_config_dataclass_kwargs["data_dimensions"] = data_dimensions

    dataclass_instance = model_config_dataclass(**model_config_dataclass_kwargs)

    kwargs["model_config"] = dataclass_instance

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
