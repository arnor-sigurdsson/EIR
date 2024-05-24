import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels

        self.gamma = nn.Parameter(
            torch.zeros(1, in_channels, 1, 1),
            requires_grad=True,
        )
        self.beta = nn.Parameter(
            torch.zeros(1, in_channels, 1, 1),
            requires_grad=True,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(
            input=x,
            p=2.0,
            dim=(2, 3),
            keepdim=True,
        )
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x
