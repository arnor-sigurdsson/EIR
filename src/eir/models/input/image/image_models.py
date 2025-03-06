from dataclasses import dataclass
from functools import reduce
from typing import Any, Literal, Protocol, TypeGuard

from torch import Tensor, nn

from eir.models.input.array.models_cnn import CNNModelConfig


class ImageModelClassGetterFunction(Protocol):
    def __call__(self, model_type: str) -> type["ImageWrapperModel"]: ...


def get_image_model_class(model_type: str) -> type["ImageWrapperModel"]:
    if model_type == "image-wrapper-default":
        return ImageWrapperModel
    raise ValueError()


@dataclass
class ImageModelConfig:
    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param num_output_features:
          Number of output final output features from image feature extractor, projected
          with a linear layer, which get passed to fusion module. If set to 0,
          the output from the feature extractor is passed directly as is to the
          fusion module.

    :param pretrained_model:
          Specify whether the model type is assumed to be pretrained and from the
          Pytorch Image Models repository.

    :param freeze_pretrained_model:
          Whether to freeze the pretrained model weights.
    """

    model_type: Literal["cnn"] | str
    model_init_config: CNNModelConfig | dict[str, Any]

    num_output_features: int = 0

    pretrained_model: bool = False
    freeze_pretrained_model: bool = False


class FeatureExtractorWithForwardFeatures(Protocol):
    def forward_features(self, x: Tensor) -> Tensor: ...


def has_forward_features(
    obj: nn.Module,
) -> TypeGuard[FeatureExtractorWithForwardFeatures]:
    return hasattr(obj, "forward_features")


class ImageWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        model_config: ImageModelConfig,
        estimated_out_shape: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor
        self.model_config = model_config
        self.estimated_out_shape = estimated_out_shape

    @property
    def num_out_features(self):
        if hasattr(self.feature_extractor, "num_out_features"):
            return self.feature_extractor.num_out_features

        if self.model_config.num_output_features > 0:
            return self.model_config.num_output_features

        return reduce(lambda x, y: x * y, self.output_shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.estimated_out_shape

    def forward(self, input: Tensor) -> Tensor:
        has_num_out_features = self.model_config.num_output_features > 0

        if has_forward_features(self.feature_extractor) and not has_num_out_features:
            out = self.feature_extractor.forward_features(input)
        else:
            out = self.feature_extractor(input)

        return out
