import dataclasses
from collections.abc import Callable

import torch
from torch import nn

from eir.models.fusion.fusion_default import al_features, default_fuse_features

al_identity_features = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


@dataclasses.dataclass
class IdentityConfig:
    pass


def pass_through_fuse(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return features


class IdentityFusionModel(nn.Module):
    def __init__(
        self,
        model_config: IdentityConfig,
        fusion_in_dim: int,
        fusion_callable: al_identity_features | al_features = default_fuse_features,
        **kwargs,
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

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.fusion_in_dim,)

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        fused_features = self.fusion_callable(inputs)

        return fused_features
