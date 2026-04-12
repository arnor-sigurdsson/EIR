import math
from dataclasses import dataclass, field

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

    layers: list[int] = field(default_factory=lambda: [2])

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
        reduce_at_fc_1: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p
        self.full_preactivation = full_preactivation
        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.RMSNorm(normalized_shape=in_features)

        fc_1_out = out_features if reduce_at_fc_1 else in_features
        self.fc_1 = nn.Linear(
            in_features=in_features,
            out_features=fc_1_out,
            bias=False,
        )

        self.act_1 = nn.GELU()
        self.do = nn.Dropout(p=dropout_p)
        self.fc_2 = nn.Linear(
            in_features=fc_1_out,
            out_features=out_features,
            bias=False,
        )

        self._norm_identity = full_preactivation or (in_features != out_features)
        self.downsample_identity: nn.Module
        if in_features != out_features:
            self.downsample_identity = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=True,
            )
            ls_init = 1.0
        else:
            self.downsample_identity = nn.Identity()
            ls_init = 1e-05

        self.ls = LayerScale(
            dim=out_features,
            init_values=ls_init,
        )

        self.stochastic_depth = StochasticDepth(
            p=self.stochastic_depth_p,
            mode="batch",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm_1(x)

        identity = out if self._norm_identity else x
        identity = self.downsample_identity(identity)

        out = self.fc_1(out)

        out = self.act_1(out)
        out = self.do(out)
        out = self.fc_2(out)
        out = self.ls(out)

        out = self.stochastic_depth(out)

        return out + identity


def _kaiming_uniform_3d(weight: torch.Tensor) -> None:
    for i in range(weight.shape[0]):
        nn.init.kaiming_uniform_(weight[i], a=math.sqrt(5))


class BatchedMLPResidualBlock(nn.Module):
    def __init__(
        self,
        n_groups: int,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.0,
        full_preactivation: bool = False,
        stochastic_depth_p: float = 0.0,
    ):
        super().__init__()

        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features
        self._norm_identity = full_preactivation or (in_features != out_features)

        self.norm_weight = nn.Parameter(torch.ones(n_groups, in_features))

        self.fc_1_weight = nn.Parameter(
            torch.empty(n_groups, out_features, in_features)
        )
        self.fc_2_weight = nn.Parameter(
            torch.empty(n_groups, out_features, out_features)
        )

        self.has_downsample = in_features != out_features
        if self.has_downsample:
            self.downsample_weight = nn.Parameter(
                torch.empty(n_groups, out_features, in_features)
            )
            self.downsample_bias = nn.Parameter(torch.zeros(n_groups, 1, out_features))
            ls_init = 1.0
        else:
            ls_init = 1e-05

        self.ls_gamma = nn.Parameter(torch.full((n_groups, 1, out_features), ls_init))

        self.act = nn.GELU()
        self.do = nn.Dropout(p=dropout_p)
        self.stochastic_depth_p = stochastic_depth_p
        self.stochastic_depth = StochasticDepth(
            p=stochastic_depth_p,
            mode="batch",
        )

        self._init_weights()

    def _init_weights(self) -> None:
        _kaiming_uniform_3d(weight=self.fc_1_weight)
        _kaiming_uniform_3d(weight=self.fc_2_weight)
        if self.has_downsample:
            _kaiming_uniform_3d(weight=self.downsample_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + 1e-8) * self.norm_weight.unsqueeze(1)

        identity = out if self._norm_identity else x
        if self.has_downsample:
            identity = (
                torch.bmm(identity, self.downsample_weight.transpose(1, 2))
                + self.downsample_bias
            )

        out = torch.bmm(out, self.fc_1_weight.transpose(1, 2))
        out = self.act(out)
        out = self.do(out)
        out = torch.bmm(out, self.fc_2_weight.transpose(1, 2))
        out = out * self.ls_gamma

        n, b, d = out.shape
        out = self.stochastic_depth(out.reshape(n * b, d)).reshape(n, b, d)

        return out + identity
