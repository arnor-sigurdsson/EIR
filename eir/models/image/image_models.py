from dataclasses import dataclass
from typing import Sequence

from torch import nn
from torch import Tensor

import timm


@dataclass
class ImageModelConfig:
    num_output_features: int = 256


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


def get_all_timm_model_names() -> Sequence[str]:
    pretrained_names = {i for i in timm.list_models() if not i.startswith("tf")}
    other_model_classes = {i for i in dir(timm.models) if "Net" in i}
    all_models = set.union(pretrained_names, other_model_classes)
    all_models_list = sorted(list(all_models))

    return all_models_list
