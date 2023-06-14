from dataclasses import dataclass
from functools import partial
from typing import Union, Type, Literal, Dict, TYPE_CHECKING, Optional, Callable

import torch
from torch import nn

from eir.models.input.omics.models_cnn import CNNModelConfig, CNNModel
from eir.models.input.omics.models_locally_connected import LCLModel, LCLModelConfig

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

al_array_model_types = Literal["cnn", "lcl"]

al_array_model_classes = Type[CNNModel] | Type[LCLModel]
al_array_models = CNNModel | LCLModel

al_array_model_config_classes = Type[CNNModelConfig] | Type[LCLModelConfig]
al_array_model_configs = CNNModelConfig | LCLModelConfig

al_pre_normalization = Optional[Literal["instancenorm", "layernorm"]]


def get_array_model_mapping() -> Dict[str, al_array_model_classes]:
    mapping = {
        "cnn": CNNModel,
        "lcl": LCLModel,
    }

    return mapping


def get_array_model_class(model_type: al_array_model_types) -> al_array_model_classes:
    mapping = get_array_model_mapping()
    return mapping[model_type]


def get_array_config_dataclass_mapping() -> Dict[str, al_array_model_config_classes]:
    mapping = {
        "cnn": CNNModelConfig,
        "lcl": LCLModelConfig,
    }

    return mapping


def get_model_config_dataclass(model_type: str) -> al_array_model_config_classes:
    mapping = get_array_config_dataclass_mapping()
    return mapping[model_type]


def get_array_model_init_kwargs(
    model_type: al_array_model_types,
    model_config: al_array_model_configs,
    data_dimensions: "DataDimensions",
) -> dict[
    str,
    Union[
        "DataDimensions",
        al_array_model_configs,
        Callable[
            [torch.Tensor],
            torch.Tensor,
        ],
    ],
]:
    kwargs: dict[
        str,
        Union[
            "DataDimensions",
            al_array_model_configs,
            Callable[
                [torch.Tensor],
                torch.Tensor,
            ],
        ],
    ] = {}

    model_config_dataclass = get_model_config_dataclass(model_type=model_type)
    dataclass_instance = model_config_dataclass(**model_config.__dict__)

    kwargs["model_config"] = dataclass_instance
    kwargs["data_dimensions"] = data_dimensions

    match model_type:
        case "lcl":
            kwargs["flatten_fn"] = partial(torch.flatten, start_dim=1)

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
        feature_extractor: al_array_models,
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
    normalization: al_pre_normalization,
    data_dimensions: "DataDimensions",
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
