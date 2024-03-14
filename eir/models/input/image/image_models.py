from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol, Type

from torch import Tensor, nn

from eir.models.input.array.models_cnn import CNNModelConfig


class ImageModelClassGetterFunction(Protocol):
    def __call__(self, model_type: str) -> Type["ImageWrapperModel"]: ...


def get_image_model_class(model_type: str) -> Type["ImageWrapperModel"]:
    if model_type == "image-wrapper-default":
        return ImageWrapperModel
    else:
        raise ValueError()


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

    model_type: Literal["cnn"] | str
    model_init_config: CNNModelConfig | Dict[str, Any]

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
