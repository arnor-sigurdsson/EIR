from dataclasses import dataclass
from typing import Dict, Any, Literal

from torch import Tensor
from torch import nn

from eir.setup.setup_utils import get_all_timm_model_names

al_image_models = tuple(Literal[i] for i in get_all_timm_model_names())


@dataclass
class ImageModelConfig:

    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param num_output_features:
          Number of output final output features from image feature extractor, which
          get passed to fusion module.

    :param pretrained_model:
          Specify whether the model type is assumed to be pretrained and from the
          Pytorch Image Models repository.

    :param freeze_pretrained_model:
          Whether to freeze the pretrained model weights.
    """

    model_type: al_image_models
    model_init_config: Dict[str, Any]

    num_output_features: int = 256

    pretrained_model: bool = False
    freeze_pretrained_model: bool = False


class ImageWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        model_config: ImageModelConfig,
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.model_config = model_config

    @property
    def num_out_features(self):
        return self.model_config.num_output_features

    def forward(self, input: Tensor) -> Tensor:
        out = self.feature_extractor(input)

        return out
