from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

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


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.w1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
            bias=bias,
        )
        self.w2 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
            bias=bias,
        )
        self.w3 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )

    def _init_weights(self):
        torch.nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.w1(x)
        gate = self.w2(x)
        hidden = F.silu(hidden) * gate
        return self.w3(hidden)


@dataclass
class TransformerBlockConfig:
    rope_theta: int = 10000


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        norm_first: bool = True,
        block_config: Optional[TransformerBlockConfig] = None,
    ):
        super().__init__()

        msg = "d_model ({d_model}) must be divisible by n_head ({n_head})"
        assert d_model % n_head == 0, msg

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.norm_first = norm_first
        self.config = block_config or TransformerBlockConfig()

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.RMSNorm(normalized_shape=d_model)
        self.norm2 = nn.RMSNorm(normalized_shape=d_model)

        self.ffn = SwiGLU(
            in_features=d_model,
            hidden_features=dim_feedforward,
            out_features=d_model,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        self._init_rope()

    def _init_rope(self):
        dim = self.head_dim
        theta = self.config.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("rope_freqs", freqs)

    def _apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        position = torch.arange(seq_len, device=x.device)
        freqs = torch.outer(position, self.rope_freqs)

        # Create rotation matrices
        cos = torch.cos(freqs).view(seq_len, 1, -1)
        sin = torch.sin(freqs).view(seq_len, 1, -1)

        # Split input into even and odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin], dim=-1
        ).flatten(-2)

        return rotated

    def _attention(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        # Apply rotary embeddings
        q = self._apply_rope(x=q, seq_len=seq_len)
        k = self._apply_rope(x=k, seq_len=seq_len)

        # Transpose for attention: [batch_size, num_heads, sequence_length, head_dim]
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.dropout(self.out_proj(attn_output))

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            x = x + self._attention(self.norm1(x), attn_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self._attention(x, attn_mask))
            x = self.norm2(x + self.ffn(x))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_head=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
