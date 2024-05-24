from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Literal, Protocol, Type, Union

from eir.models.input.array.models_cnn import CNNModel, CNNModelConfig
from eir.models.input.array.models_identity import IdentityModel, IdentityModelConfig
from eir.models.input.array.models_linear import LinearModel, LinearModelConfig
from eir.models.input.array.models_locally_connected import (
    FlattenFunc,
    LCLModel,
    LCLModelConfig,
    SimpleLCLModel,
    SimpleLCLModelConfig,
    flatten_h_w_fortran,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

logger = get_logger(name=__name__)

al_omics_model_classes = Union[
    Type["CNNModel"],
    Type["LinearModel"],
    Type["SimpleLCLModel"],
    Type["LCLModel"],
    Type["IdentityModel"],
]

al_omics_models = Union[
    "CNNModel", "LinearModel", "SimpleLCLModel", "LCLModel", "IdentityModel"
]

al_omics_model_types = Literal[
    "cnn",
    "linear",
    "lcl-simple",
    "genome-local-net",
    "linear",
]

al_omics_model_config_classes = Union[
    Type[CNNModelConfig],
    Type[LinearModelConfig],
    Type[SimpleLCLModelConfig],
    Type[LCLModelConfig],
    Type[IdentityModelConfig],
]

al_omics_model_configs = Union[
    CNNModelConfig,
    LinearModelConfig,
    SimpleLCLModelConfig,
    LCLModelConfig,
    IdentityModelConfig,
]


@dataclass
class OmicsModelConfig:
    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration used to initialise model.
    """

    model_type: al_omics_model_types
    model_init_config: al_omics_model_configs


def get_omics_model_mapping() -> Dict[str, al_omics_model_classes]:
    mapping = {
        "cnn": CNNModel,
        "linear": LinearModel,
        "lcl-simple": SimpleLCLModel,
        "genome-local-net": LCLModel,
        "identity": IdentityModel,
    }

    return mapping


def get_omics_model_class(model_type: al_omics_model_types) -> al_omics_model_classes:
    mapping = get_omics_model_mapping()
    return mapping[model_type]


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict]


def get_omics_config_dataclass_mapping() -> Dict[str, al_omics_model_config_classes]:
    mapping = {
        "cnn": CNNModelConfig,
        "linear": LinearModelConfig,
        "lcl-simple": SimpleLCLModelConfig,
        "genome-local-net": LCLModelConfig,
        "identity": IdentityModelConfig,
    }

    return mapping


def get_model_config_dataclass(
    model_type: str,
) -> al_omics_model_config_classes:
    mapping = get_omics_config_dataclass_mapping()
    return mapping[model_type]


def get_omics_model_init_kwargs(
    model_type: al_omics_model_types,
    model_config: al_omics_model_configs,
    data_dimensions: "DataDimensions",
) -> dict[str, Union["DataDimensions", al_omics_model_configs | FlattenFunc]]:
    """
    See: https://github.com/python/mypy/issues/5374 for type hint issue.

    Possibly split / extend this function later to account for other kwargs that just
    model_config, to allow for more flexibility in model instantiation (not restricting
    to just model_config object).
    """

    kwargs: dict[str, Union["DataDimensions", al_omics_model_configs | FlattenFunc]] = (
        {}
    )
    base_kwargs = model_config.__dict__
    base_kwargs = _enforce_omics_specific_settings(
        base_kwargs=base_kwargs, model_type=model_type
    )

    model_config_dataclass = get_model_config_dataclass(model_type=model_type)
    dataclass_instance = model_config_dataclass(**base_kwargs)

    kwargs["model_config"] = dataclass_instance
    kwargs["data_dimensions"] = data_dimensions

    match model_type:
        case "genome-local-net" | "lcl-simple":
            kwargs["flatten_fn"] = flatten_h_w_fortran

    return kwargs


def _enforce_omics_specific_settings(
    base_kwargs: Dict[str, Any], model_type: str
) -> Dict[str, Any]:
    expected = {
        "cnn": {
            "kernel_height": 1,
            "first_kernel_expansion_height": 4,
        }
    }

    if model_type not in expected:
        return base_kwargs

    for key, value in expected[model_type].items():
        logger.info(
            f"Overriding {key} to {value} for {model_type} in omics. "
            "If you want more control of these parameters, "
            "it might be a good idea to use the array input functionality."
        )
        base_kwargs[key] = value

    return base_kwargs
