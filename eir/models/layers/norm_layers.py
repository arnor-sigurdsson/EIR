import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1.0,
    ) -> None:
        """
        Note: We use an init value of 1.0 here instead of the generally applied
        1e-5. As we often have linear down sampling layers for residual identities,
        the models might be more inclined to use the main branch, and having the
        scaling too low might cause learning issues.
        """
        super().__init__()
        self.dim = dim
        self.init_values = init_values
        self.gamma = nn.Parameter(
            data=init_values * torch.ones(dim),
            requires_grad=True,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim={self.dim}, init_values={self.init_values})"
        )

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
