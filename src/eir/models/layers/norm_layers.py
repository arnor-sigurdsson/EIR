import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1.0,
        n_dims: int | None = None,
    ) -> None:
        """
        Note: We use an init value of 1.0 here instead of the generally applied
        1e-5. As we often have linear down sampling layers for residual identities,
        the models might be more inclined to use the main branch, and having the
        scaling too low might cause learning issues.

        Args:
            dim: Number of features to scale
            init_values: Initial value for scaling parameters
            n_dims: Number of dimensions in the input tensor (including batch)
        """
        super().__init__()
        self.dim = dim
        self.init_values = init_values
        self.n_dims = n_dims

        gamma_shape: tuple[int, ...]
        if n_dims is None:
            gamma_shape = (dim,)
        elif n_dims == 4:
            gamma_shape = (1, dim, 1, 1)
        elif n_dims == 3:
            gamma_shape = (1, 1, dim)
        elif n_dims == 2:
            gamma_shape = (1, dim)
        else:
            raise ValueError(
                f"Unsupported n_dims: {n_dims}. Supported values are None, 2, 3, and 4."
            )

        self.gamma = nn.Parameter(
            data=init_values * torch.ones(gamma_shape),
            requires_grad=True,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dim={self.dim}, "
            f"init_values={self.init_values}, "
            f"n_dims={self.n_dims})"
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
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        do_scale: bool = True,
        normalize_dim: int = 2,
    ):
        super().__init__()
        self.do_scale = do_scale
        self.normalize_dim = normalize_dim

        self.g: torch.Tensor | int = 1
        if do_scale:
            self.g = nn.Parameter(data=torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        normalize_dim = self.normalize_dim

        scale = self.g
        if self.do_scale:
            assert isinstance(self.g, torch.Tensor)
            dims = x.ndim - self.normalize_dim - 1
            scale = append_dims(t=self.g, dims=dims)

        out = (
            F.normalize(input=x, dim=normalize_dim)
            * scale
            * (x.shape[normalize_dim] ** 0.5)
        )
        return out


def append_dims(t: torch.Tensor, dims: int) -> torch.Tensor:
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))
