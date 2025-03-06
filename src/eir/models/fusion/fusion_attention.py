from typing import Any

import torch
from einops import rearrange
from torch import einsum, nn
from torch.nn import functional as F

from eir.models.layers.attention_layers import SwiGLU, Transformer


class MetaSequenceProjection(nn.Module):
    def __init__(
        self,
        context_total_num_elements: int,
        context_embedding_dim: int,
        target_embedding_dim: int,
        target_max_length: int,
        apply_causal_mask: bool,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.context_total_num_elements = context_total_num_elements
        self.context_embedding_dim = context_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length
        self.apply_causal_mask = apply_causal_mask

        self.cross_attention_projection = SequenceResidualCrossAttention(
            context_embedding_dim=self.context_embedding_dim,
            target_embedding_dim=self.target_embedding_dim,
            target_max_length=self.target_max_length,
            apply_causal_mask=self.apply_causal_mask,
        )

        self.scaling_factors = nn.Parameter(torch.ones(2))

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.softmax(self.scaling_factors, dim=0)

        cross_attended = self.cross_attention_projection(
            x=target_tensor,
            context=input_tensor,
        )
        cross_attended = cross_attended * weights[0]

        identity = target_tensor * weights[1]

        out = cross_attended + identity

        return out


class SequenceResidualCrossAttention(nn.Module):
    def __init__(
        self,
        context_embedding_dim: int,
        target_embedding_dim: int,
        target_max_length: int,
        apply_causal_mask: bool,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.context_embedding_dim = context_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length

        self.projection_layer = UniDirectionalCrossAttention(
            dim=self.target_embedding_dim,
            context_dim=self.context_embedding_dim,
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            pre_norm=False,
            apply_causal_mask=apply_causal_mask,
        )

        self.norm_1_target = nn.RMSNorm(normalized_shape=target_embedding_dim)
        self.act_1 = SwiGLU(
            in_features=target_embedding_dim,
            hidden_features=target_embedding_dim,
            out_features=target_embedding_dim,
            bias=False,
        )

        self.norm_1_context = nn.RMSNorm(normalized_shape=context_embedding_dim)
        self.act_context = SwiGLU(
            in_features=context_embedding_dim,
            hidden_features=context_embedding_dim,
            out_features=context_embedding_dim,
            bias=False,
        )

        self.encoder = Transformer(
            d_model=target_embedding_dim,
            nhead=8,
            num_layers=1,
            dim_feedforward=target_embedding_dim * 4,
            dropout=0.1,
            norm_first=True,
        )

        self.ca_mask: torch.Tensor | None
        if apply_causal_mask:
            ca_mask_tensor = torch.ones((1, self.target_max_length)).bool()
            self.register_buffer("ca_mask", ca_mask_tensor)
        else:
            self.register_buffer("ca_mask", None)

        encoder_mask = torch.triu(
            torch.ones(self.target_max_length, self.target_max_length) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("encoder_mask", encoder_mask)

        self.norm_2_target = nn.RMSNorm(normalized_shape=target_embedding_dim)
        self.act_2 = SwiGLU(
            in_features=target_embedding_dim,
            hidden_features=target_embedding_dim,
            out_features=target_embedding_dim,
            bias=False,
        )

        self.downsample_identity = UniDirectionalCrossAttention(
            dim=self.target_embedding_dim,
            context_dim=self.context_embedding_dim,
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            pre_norm=False,
            apply_causal_mask=apply_causal_mask,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = context.reshape(context.shape[0], -1, self.context_embedding_dim)

        identity = self.downsample_identity(
            x=x,
            context=context,
            mask=self.ca_mask,
        )

        out = self.norm_1_target(x)
        out = self.act_1(out)

        out_context = self.norm_1_context(context)
        out_context = self.act_context(out_context)

        out = self.projection_layer(
            x=out,
            context=out_context,
            mask=self.ca_mask,
        )

        out = self.norm_2_target(out)
        out = self.act_2(out)
        out = self.encoder(
            out,
            mask=self.encoder_mask,
        )

        return out + identity


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def stable_softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


class UniDirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        pre_norm: bool = False,
        apply_causal_mask: bool = False,
    ):
        """
        Adapted from: https://github.com/lucidrains/bidirectional-cross-attention

        Note that enabling masking here means that position i in x is only allowed to
        attend to position j in context, where j <= i. This can be useful if x and
        the context are directly related position-wise, and we want to ensure that
        the model cannot look at "future" positions through the context.

        Without masking, position i in x can attend to all positions in context.
        """
        super().__init__()
        context_dim = default(context_dim, dim)

        self.apply_causal_mask = apply_causal_mask

        self.norm = nn.RMSNorm(dim) if pre_norm else nn.Identity()
        self.context_norm = nn.RMSNorm(context_dim) if pre_norm else nn.Identity()

        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(p=dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        _, context_len, _ = context.shape

        x_norm = self.norm(x)
        context_norm = self.context_norm(context)

        # Project to queries, keys, values
        q = self.to_q(x_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        # Split heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # Calculate attention scores
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # Apply causal masking if it was set during initialization
        if self.apply_causal_mask:
            # Create causal mask - strictly enforces that position i in x
            # can only attend to positions 0 through i in context
            causal_mask = (
                torch.ones(seq_len, context_len, device=x.device)
                .triu_(diagonal=1)
                .bool()
            )

            # [1, 1, seq_len, context_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if exists(val=mask):
            assert mask is not None
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, context_len]
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Calculate attention weights with softmax
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # Apply attention weights to values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # Merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        # Project to output dimension
        out = self.to_out(out)

        return out
