from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from eir.models.layers.norm_layers import LayerScale


@dataclass
class ResidualMLPConfig:
    """
    :param layers:
        Number of residual MLP layers to use in for each output predictor after fusing.

    :param fc_task_dim:
        Number of hidden nodes in each MLP residual block.

    :param rb_do:
        Dropout in each MLP residual block.

    :param fc_do:
        Dropout before final layer.

    :param stochastic_depth_p:
        Probability of dropping input.

    """

    layers: List[int] = field(default_factory=lambda: [2])

    fc_task_dim: int = 256

    rb_do: float = 0.10
    fc_do: float = 0.10

    stochastic_depth_p: float = 0.10


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.0,
        full_preactivation: bool = False,
        stochastic_depth_p: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.LayerNorm(normalized_shape=in_features)

        self.fc_1 = nn.Linear(
            in_features=in_features, out_features=out_features, bias=True
        )

        self.act_1 = nn.GELU()
        self.do = nn.Dropout(p=dropout_p)
        self.fc_2 = nn.Linear(
            in_features=out_features, out_features=out_features, bias=True
        )

        if in_features == out_features:
            self.downsample_identity = lambda x: x
        else:
            self.downsample_identity = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
            )

        self.ls = LayerScale(dim=out_features, init_values=1.0)

        self.stochastic_depth = StochasticDepth(p=self.stochastic_depth_p, mode="batch")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        identity = out if self.full_preactivation else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.act_1(out)
        out = self.do(out)
        out = self.fc_2(out)
        out = self.ls(out)

        out = self.stochastic_depth(out)

        return out + identity
