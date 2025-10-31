from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    create_multi_task_blocks_with_first_adaptor_block,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

al_features = Callable[[dict[str, torch.Tensor]], torch.Tensor]


logger = get_logger(__name__)


def default_fuse_features(features: dict[str, torch.Tensor]) -> torch.Tensor:
    feature_flatten = {k: v.flatten(start_dim=1) for k, v in features.items()}
    return torch.cat(tuple(feature_flatten.values()), dim=1)


class MLPResidualFusionModule(nn.Module):
    def __init__(
        self,
        model_config: ResidualMLPConfig,
        fusion_in_dim: int,
        fusion_callable: al_features = default_fuse_features,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.fusion_in_dim = fusion_in_dim
        self.fusion_callable = fusion_callable

        fusion_resblocks_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }
        fusion_modules = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=("fusion",),
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=fusion_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": fusion_in_dim},
        )

        self.fusion_modules = fusion_modules

    @property
    def num_out_features(self) -> int:
        return self.model_config.fc_task_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.fusion_callable(inputs)
        out = calculate_module_dict_outputs(
            input_=fused_features,
            module_dict=self.fusion_modules,
        )

        return out["fusion"]


class SumFusionModule(nn.Module):
    def __init__(
        self,
        model_config: ResidualMLPConfig,
        fusion_in_dim: int,
        feature_dimensions_and_types: dict[str, "FeatureExtractorInfo"] | None = None,
        **kwargs,
    ):
        super().__init__()

        if feature_dimensions_and_types is None:
            raise ValueError(
                "SumFusionModule requires feature_dimensions_and_types to be provided"
            )

        self.model_config = model_config
        self.fusion_dim = model_config.fc_task_dim
        self.feature_dimensions_and_types = feature_dimensions_and_types

        self.input_projections = nn.ModuleDict()
        for name, info in feature_dimensions_and_types.items():
            output_dim = info.output_dimension
            self.input_projections[name] = nn.Sequential(
                nn.RMSNorm(normalized_shape=output_dim),
                nn.GELU(),
                nn.Linear(in_features=output_dim, out_features=self.fusion_dim),
            )

        fusion_resblocks_kwargs = {
            "in_features": self.fusion_dim,
            "out_features": self.fusion_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }
        fusion_modules = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=("fusion",),
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=fusion_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": self.fusion_dim},
        )

        self.fusion_modules = fusion_modules

    @property
    def num_out_features(self) -> int:
        return self.fusion_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = []
        for name, tensor in inputs.items():
            flattened = tensor.flatten(start_dim=1)
            proj = self.input_projections[name](flattened)
            projected.append(proj)

        fused = torch.stack(projected, dim=0).sum(dim=0)

        out = calculate_module_dict_outputs(
            input_=fused,
            module_dict=self.fusion_modules,
        )

        return out["fusion"]
