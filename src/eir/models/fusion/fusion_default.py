from typing import TYPE_CHECKING, Callable, Dict

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    create_multi_task_blocks_with_first_adaptor_block,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

al_features = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


logger = get_logger(__name__)


def default_fuse_features(features: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.fusion_callable(inputs)
        out = calculate_module_dict_outputs(
            input_=fused_features,
            module_dict=self.fusion_modules,
        )

        return out["fusion"]
