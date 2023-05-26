from typing import Optional, Any

import torch
from einops import rearrange
from torch import nn, einsum, Tensor

from eir.models.layers import get_lcl_projection_layer, get_projection_layer


class MetaSequenceProjection(nn.Module):
    def __init__(
        self,
        in_total_num_elements: int,
        in_embedding_dim: int,
        target_embedding_dim: int,
        target_max_length: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.in_total_num_elements = in_total_num_elements
        self.in_embedding_dim = in_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length

        self.projection = SequenceProjection(
            in_features=in_total_num_elements,
            target_embedding_dim=self.target_embedding_dim,
            target_max_length=self.target_max_length,
        )

        self.cross_attention_projection = SequenceResidualCrossAttentionProjection(
            in_embedding_dim=self.in_embedding_dim,
            target_embedding_dim=self.target_embedding_dim,
            target_max_length=self.target_max_length,
        )

    def forward(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        lcl_projected = self.projection(input_tensor)

        cross_attended = self.cross_attention_projection(
            x=target_tensor, context=input_tensor
        )

        return lcl_projected + cross_attended


class SequenceProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        target_embedding_dim: int,
        target_max_length: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length

        self.out_dim = self.target_max_length * self.target_embedding_dim

        projection_kwargs = {
            "input_dimension": in_features,
            "target_dimension": self.out_dim,
        }

        self.norm_1 = nn.LayerNorm(normalized_shape=in_features)
        self.act = nn.Mish()

        self.projection_layer = get_projection_layer(**projection_kwargs)

        encoder_layer_base = nn.TransformerEncoderLayer(
            d_model=target_embedding_dim,
            nhead=8,
            dim_feedforward=target_embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer_base, num_layers=1
        )

        self.downsample_identity = nn.Identity()
        if self.in_features != self.out_dim:
            self.downsample_identity = get_lcl_projection_layer(**projection_kwargs)

    def forward(self, input: Tensor) -> Tensor:
        input_flat = input.flatten(1)

        identity = self.downsample_identity(input_flat)[..., : self.out_dim]
        identity = identity.reshape(
            identity.shape[0], self.target_max_length, self.target_embedding_dim
        )

        out = self.norm_1(input_flat)
        out = self.act(out)
        out = self.projection_layer(out)[..., : self.out_dim]

        out = out.reshape(
            out.shape[0], self.target_max_length, self.target_embedding_dim
        )

        out = self.encoder(out)

        return out + identity


class SequenceResidualCrossAttentionProjection(nn.Module):
    def __init__(
        self,
        in_embedding_dim: int,
        target_embedding_dim: int,
        target_max_length: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.in_embedding_dim = in_embedding_dim
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length

        self.projection_layer = UniDirectionalCrossAttention(
            dim=self.target_embedding_dim,
            context_dim=self.in_embedding_dim,
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            talking_heads=False,
            pre_norm=False,
        )

        self.act = nn.Mish()
        self.norm_1_target = nn.LayerNorm(normalized_shape=target_embedding_dim)
        self.norm_1_context = nn.LayerNorm(normalized_shape=in_embedding_dim)

        encoder_layer_base = nn.TransformerEncoderLayer(
            d_model=target_embedding_dim,
            nhead=8,
            dim_feedforward=target_embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer_base, num_layers=1
        )

        ca_mask = torch.ones((1, self.target_max_length)).bool()
        self.register_buffer("ca_mask", ca_mask)

        encoder_mask = torch.triu(
            torch.ones(self.target_max_length, self.target_max_length) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("encoder_mask", encoder_mask)

        self.norm_2_target = nn.LayerNorm(normalized_shape=target_embedding_dim)

        self.downsample_identity = UniDirectionalCrossAttention(
            dim=self.target_embedding_dim,
            context_dim=self.in_embedding_dim,
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            talking_heads=False,
            pre_norm=False,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = context.reshape(context.shape[0], -1, self.in_embedding_dim)

        identity = self.downsample_identity(x=x, context=context, mask=self.ca_mask)

        out = self.norm_1_target(x)
        out = self.act(out)

        out_context = self.norm_1_context(context)
        out_context = self.act(out_context)

        out = self.projection_layer(x=out, context=out_context, mask=self.ca_mask)

        out = self.norm_2_target(out)
        out = self.act(out)
        out = self.encoder(out, mask=self.encoder_mask)

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
        heads: int = 8,
        dim_head: int = 64,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        talking_heads: bool = False,
        pre_norm: bool = False,
    ):
        """
        Adapted from: https://github.com/lucidrains/bidirectional-cross-attention
        """
        super().__init__()
        context_dim = default(val=context_dim, d=dim)

        self.norm = nn.LayerNorm(normalized_shape=dim) if pre_norm else nn.Identity()
        self.context_norm = (
            nn.LayerNorm(normalized_shape=context_dim) if pre_norm else nn.Identity()
        )

        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.talking_heads = (
            nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        )

    def forward(
        self,
        x,
        context,
        mask=None,
        context_mask=None,
        rel_pos_bias=None,
    ) -> torch.Tensor:
        _, i, j, h, device = (
            x.shape[0],
            x.shape[-2],
            context.shape[-2],
            self.heads,
            x.device,
        )

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head
        qk, context_qk, v, context_v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (qk, context_qk, v, context_v),
        )

        # get similarities
        sim = einsum("b h i d, b h j d -> b h i j", qk, context_qk) * self.scale

        # relative positional bias, if supplied
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # causal mask
        if exists(mask) or exists(context_mask):
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        # get attention along sequence length, using shared similarity matrix
        attn = stable_softmax(sim, dim=-1)

        attn = self.dropout(attn)

        # talking heads
        attn = self.talking_heads(attn)

        # src sequence aggregates values from context
        out = einsum("b h i j, b h j d -> b h i d", attn, context_v)

        # merge heads and combine out
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)

        return out
