import dataclasses
from typing import Dict

import torch
from torch import nn

from eir.models.fusion.fusion_default import al_features, default_fuse_features


@dataclasses.dataclass
class IdentityConfig:
    pass


class IdentityFusionModel(nn.Module):
    def __init__(
        self,
        model_config: IdentityConfig,
        fusion_in_dim: int,
        fusion_callable: al_features = default_fuse_features,
    ):
        super().__init__()

        self.model_config = model_config
        self.fusion_in_dim = fusion_in_dim
        self.fusion_callable = fusion_callable

        self._init_weights()

    def _init_weights(self):
        pass

    @property
    def num_out_features(self) -> int:
        return self.fusion_in_dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.fusion_callable(tuple(inputs.values()))

        return fused_features