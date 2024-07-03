import torch
from einops import rearrange
from torch import nn

from eir.models.layers.norm_layers import RMSNorm


class LinearAttention(nn.Module):
    """
    From https://github.com/lucidrains
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=hidden_dim * 3,
            kernel_size=1,
            bias=False,
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=embed_dim, kernel_size=1),
            RMSNorm(dim=embed_dim, normalize_dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out)
