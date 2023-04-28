from dataclasses import dataclass
from typing import Union, Type, Literal, Dict, Any, TYPE_CHECKING

import torch
from torch import nn

from eir.models.omics.models_cnn import CNNModelConfig, CNNModel
from eir.models.omics.omics_models import Dataclass

if TYPE_CHECKING:
    from eir.setup.input_setup import DataDimensions

al_omics_model_classes = Union[Type["CNNModel"]]

al_omics_models = Union["CNNModel"]

al_array_model_types = Literal["cnn"]

al_array_model_configs = Union[CNNModelConfig]

al_pre_normalization = Union[None, Literal["instancenorm", "layernorm"]]


def get_array_model_mapping() -> Dict[str, al_omics_model_classes]:
    mapping = {
        "cnn": CNNModel,
    }

    return mapping


def get_array_model_class(model_type: str) -> al_omics_model_classes:
    mapping = get_array_model_mapping()
    return mapping[model_type]


def get_array_config_dataclass_mapping() -> Dict[str, Type[Dataclass]]:
    mapping = {
        "cnn": CNNModelConfig,
    }

    return mapping


def get_model_config_dataclass(model_type: str) -> Type[Dataclass]:
    mapping = get_array_config_dataclass_mapping()
    return mapping[model_type]


def get_array_model_init_kwargs(
    model_type: str,
    model_config: al_array_model_configs,
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


@dataclass
class ArrayModelConfig:

    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration used to initialise model.
    """

    model_type: al_array_model_types
    model_init_config: al_array_model_configs
    pre_normalization: al_pre_normalization = None


class ArrayWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        normalization: al_pre_normalization,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.data_dimensions = feature_extractor.data_dimensions
        self.pre_normalization = get_pre_normalization_layer(
            normalization=normalization, data_dimensions=self.data_dimensions
        )

    @property
    def num_out_features(self):
        return self.feature_extractor.num_out_features

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.feature_extractor.l1_penalized_weights

    def forward(self, x):
        out = self.pre_normalization(x)
        out = self.feature_extractor(out)
        return out


def get_pre_normalization_layer(
    normalization: al_pre_normalization, data_dimensions: "DataDimensions"
) -> Union[nn.InstanceNorm2d, nn.LayerNorm, nn.Identity]:
    channels = data_dimensions.channels
    height = data_dimensions.height
    width = data_dimensions.width

    match normalization:
        case "instancenorm":
            return nn.InstanceNorm2d(
                num_features=channels, affine=True, track_running_stats=True
            )
        case "layernorm":
            return nn.LayerNorm(normalized_shape=[channels, height, width])
        case None:
            return nn.Identity()
