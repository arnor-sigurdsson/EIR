from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from eir.models.input.array.array_models import al_pre_normalization
from eir.models.input.array.models_locally_connected import LCLModel, LCLModelConfig
from eir.models.layers.projection_layers import get_1d_projection_layer
from eir.models.output.array.output_array_models_cnn import (
    CNNPassThroughUpscaleModel,
    CNNUpscaleModel,
    CNNUpscaleModelConfig,
)

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

al_array_model_types = Literal["lcl", "cnn", "cnn-passthrough"]
al_output_array_model_classes = (
    type[LCLModel] | type[CNNUpscaleModel] | type[CNNPassThroughUpscaleModel]
)
al_output_array_models = LCLModel | CNNUpscaleModel | CNNPassThroughUpscaleModel
al_output_array_model_config_classes = (
    type["LCLOutputModelConfig"] | type[CNNUpscaleModelConfig]
)


@dataclass
class LCLOutputModelConfig(LCLModelConfig):
    cutoff: int | Literal["auto"] = "auto"


@dataclass
class ArrayOutputModuleConfig:
    """
    :param model_type:
        Which type of image model to use.

    :param model_init_config:
        Configuration used to initialise model.
    """

    model_type: al_array_model_types
    model_init_config: LCLOutputModelConfig | CNNUpscaleModelConfig
    pre_normalization: al_pre_normalization = None


class ArrayOutputWrapperModule(nn.Module):
    def __init__(
        self,
        feature_extractor: al_output_array_models,
        output_name: str,
        target_data_dimensions: "DataDimensions",
        num_classes: int | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.output_name = output_name
        self.data_dimensions = target_data_dimensions
        self.num_classes = num_classes

        n_elements = self.data_dimensions.num_elements()
        if self.num_classes is not None:
            self.target_width = n_elements * self.num_classes
            assert self.data_dimensions.original_shape is not None
            self.original_shape = self.data_dimensions.original_shape
        else:
            self.target_width = n_elements
        self.target_shape = self.data_dimensions.full_shape()

        projection_layer_type: Literal[
            "auto", "lcl", "lcl_residual", "linear", "cnn"
        ] = "auto"
        if self.num_classes is not None:
            projection_layer_type = "linear"

        self.projection_head = get_1d_projection_layer(
            input_dimension=self.feature_extractor.num_out_features,
            target_dimension=self.target_width,
            projection_layer_type=projection_layer_type,
            lcl_diff_tolerance=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.feature_extractor(x)

        out = out.reshape(out.shape[0], -1)
        out = self.projection_head(out)

        out = out[:, : self.target_width]

        if self.num_classes is not None:
            out = out.reshape(-1, self.num_classes, *self.original_shape)
        else:
            out = out.reshape(-1, *self.target_shape)

        return {self.output_name: out}


def get_array_output_module(
    feature_extractor: al_output_array_models,
    output_name: str,
    target_data_dimensions: "DataDimensions",
    num_classes: int | None = None,
) -> ArrayOutputWrapperModule:
    return ArrayOutputWrapperModule(
        feature_extractor=feature_extractor,
        output_name=output_name,
        target_data_dimensions=target_data_dimensions,
        num_classes=num_classes,
    )
