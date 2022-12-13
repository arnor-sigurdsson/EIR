from typing import Callable, Dict, Union, TYPE_CHECKING

import torch
from aislib.misc_utils import get_logger
from torch import nn

from eir.models.layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_base import (
    create_multi_task_blocks_with_first_adaptor_block,
    calculate_module_dict_outputs,
)

if TYPE_CHECKING:
    pass

al_features = Callable[
    [Dict[str, torch.Tensor]], Union[torch.Tensor, Dict[str, torch.Tensor]]
]


logger = get_logger(__name__)


def default_fuse_features(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(features.values()), dim=1)


class FusionModule(nn.Module):
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
            input_=fused_features, module_dict=self.fusion_modules
        )

        return out["fusion"]
