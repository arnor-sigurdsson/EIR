from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from eir.models.layers.norm_layers import LayerScale, RMSNorm

SparseAttentionType = Literal["softmax", "entmax15", "sparsemax"]


def _make_ix_like(input: Tensor, dim: int = -1) -> Tensor:
    d = input.size(dim)
    rho = torch.arange(start=1, end=d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[dim] = -1
    return rho.view(view)


class _Entmax15Function(torch.autograd.Function):
    """
    All entmax/sparsemax functionality adapted from:
    https://github.com/deep-spin/entmax/
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: Tensor, dim: int = -1
    ) -> Tensor:  # type: ignore[override]
        ctx.dim = dim  # type: ignore[attr-defined]

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2

        x_srt, _ = torch.sort(input=input, descending=True, dim=dim)
        rho = _make_ix_like(input=input, dim=dim)
        mean = x_srt.cumsum(dim=dim) / rho
        mean_sq = (x_srt**2).cumsum(dim=dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho
        delta_nz = torch.clamp(input=delta, min=0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= x_srt).sum(dim=dim).unsqueeze(dim=dim)
        tau_star = tau.gather(dim=dim, index=support_size - 1)

        output = torch.clamp(input=input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor, None]:  # type: ignore[override]
        (y,) = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]

        gppr = y.sqrt()
        dx = grad_output * gppr
        q = dx.sum(dim=dim, keepdim=True) / gppr.sum(dim=dim, keepdim=True).clamp(
            min=1e-12
        )
        dx -= q * gppr
        dx *= (y > 0).float()

        return dx, None


class _SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, input: Tensor, dim: int = -1
    ) -> Tensor:  # type: ignore[override]
        ctx.dim = dim  # type: ignore[attr-defined]

        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val

        z_sorted, _ = torch.sort(input=input, descending=True, dim=dim)
        z_cumsum = z_sorted.cumsum(dim=dim)
        k = _make_ix_like(input=input, dim=dim)
        support = (1 + k * z_sorted > z_cumsum).float()
        k_star = support.sum(dim=dim, keepdim=True)
        tau = (z_cumsum.gather(dim=dim, index=(k_star - 1).long()) - 1) / k_star

        output = torch.clamp(input=input - tau, min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor, None]:  # type: ignore[override]
        (output,) = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]

        nonzero = (output > 0).float()
        grad_input = grad_output * nonzero
        v_hat = grad_input.sum(dim=dim, keepdim=True) / nonzero.sum(
            dim=dim, keepdim=True
        ).clamp(min=1e-12)
        grad_input -= v_hat * nonzero
        return grad_input, None


def entmax15(input: Tensor, dim: int = -1) -> Tensor:
    return _Entmax15Function.apply(input, dim)


def sparsemax(input: Tensor, dim: int = -1) -> Tensor:
    return _SparsemaxFunction.apply(input, dim)


class EntmaxMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.0,
        attention_type: SparseAttentionType = "entmax15",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_type = attention_type

        self.q_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.k_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.v_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.dropout = nn.Dropout(p=dropout_p)

        if attention_type == "entmax15":
            self._attn_fn = entmax15
        elif attention_type == "sparsemax":
            self._attn_fn = sparsemax
        else:
            self._attn_fn = partial(F.softmax, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self._attn_fn(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, s, self.embed_dim)
        return self.out_proj(out)


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
        q, k, v = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv
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

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.w3.weight, mean=0.0, std=0.02)

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
        block_config: TransformerBlockConfig | None = None,
    ):
        super().__init__()

        msg = f"d_model ({d_model}) must be divisible by n_head ({n_head})"
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

        self.ls1 = LayerScale(dim=d_model, init_values=1e-05)
        self.ls2 = LayerScale(dim=d_model, init_values=1e-05)

        self.rope_freqs: torch.Tensor
        self._init_rope()

    def _init_rope(self):
        dim = self.head_dim
        theta = self.config.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("rope_freqs", freqs)

    def _apply_rope(self, x: Tensor, seq_len: int) -> Tensor:
        position = torch.arange(seq_len, device=x.device)
        freqs = torch.outer(input=position, vec2=self.rope_freqs)

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

    def _attention(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        # Apply rotary embeddings
        q = self._apply_rope(x=q, seq_len=seq_len)
        k = self._apply_rope(x=k, seq_len=seq_len)

        # Transpose for attention: [batch_size, num_heads, sequence_length, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.dropout(self.out_proj(attn_output))

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        if self.norm_first:
            x = x + self.ls1(self._attention(self.norm1(x), attn_mask))
            x = x + self.ls2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.ls1(self._attention(x, attn_mask)))
            x = self.norm2(x + self.ls2(self.ffn(x)))
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
