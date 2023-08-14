from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.input.array.array_models import (
    al_array_model_configs,
    al_array_model_types,
    al_array_models,
    al_pre_normalization,
)
from eir.models.layers.projection_layers import get_projection_layer

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions


@dataclass
class ArrayOutputModuleConfig:

    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration used to initialise model.
    """

    model_type: al_array_model_types
    model_init_config: al_array_model_configs
    pre_normalization: al_pre_normalization = None


class ArrayOutputWrapperModule(nn.Module):
    def __init__(
        self,
        feature_extractor: al_array_models,
        output_name: str,
        target_data_dimensions: "DataDimensions",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.output_name = output_name
        self.data_dimensions = target_data_dimensions

        self.target_width = self.data_dimensions.num_elements()
        self.target_shape = self.data_dimensions.full_shape()

        self.projection_head = get_projection_layer(
            input_dimension=self.feature_extractor.num_out_features,
            target_dimension=self.target_width,
            projection_layer_type="lcl_residual",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.feature_extractor(x)
        out = self.projection_head(out)

        out = out[:, : self.target_width]

        out = out.reshape(-1, *self.target_shape)

        return {self.output_name: out}


def get_array_output_module(
    feature_extractor: al_array_models,
    output_name: str,
    target_data_dimensions: "DataDimensions",
) -> ArrayOutputWrapperModule:
    return ArrayOutputWrapperModule(
        feature_extractor=feature_extractor,
        output_name=output_name,
        target_data_dimensions=target_data_dimensions,
    )
