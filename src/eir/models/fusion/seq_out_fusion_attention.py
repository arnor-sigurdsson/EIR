from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from eir.models.layers.attention_layers import SwiGLU, Transformer
from eir.models.layers.norm_layers import LayerScale
from eir.models.tensor_broker.projection_modules.sequence import (
    get_reshape_to_attention_dims_func,
)


class MetaSequenceFusion(nn.Module):
    def __init__(
        self,
        context_shape: tuple[int, ...],
        target_embedding_dim: int,
        target_max_length: int,
        apply_causal_mask: bool,
        forced_context_shape: None | tuple[int, ...],
        n_layers: int = 1,
        *args,
        **kwargs,
    ):
        """
        The forced_context_shape allows us to overwrite inferred shape in
        special cases where we know something about the modality. For example,
        sequence feature extractors return the flattened sequence. During
        setup, the inferred approach would do the CA with this as a single, flat
        token, which probably results in a lot of parameters and does not
        take advantage of the sequence structure. Forcing the reshape
        treats this again as a sequence of tokens in the CA.
        """
        super().__init__()

        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length
        self.apply_causal_mask = apply_causal_mask
        self.n_layers = n_layers
        self.forced_context_reshape = forced_context_shape

        if self.forced_context_reshape:
            self.context_shape = self.forced_context_reshape
        else:
            self.context_shape = torch.Size(context_shape)

        self.cross_attention_layers = nn.ModuleList(
            [
                SequenceResidualCrossAttention(
                    context_shape=self.context_shape,
                    target_embedding_dim=self.target_embedding_dim,
                    target_max_length=self.target_max_length,
                    apply_causal_mask=self.apply_causal_mask,
                    forced_context_shape=self.forced_context_reshape,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Note:   input_tensor here is the context that we inject into the
                target_tensor.

        Note:   No LayerScale here as we are just wrapping the CA blocks
                which apply their own LayerScale.
        """
        identity = target_tensor

        cross_attended = target_tensor
        for layer in self.cross_attention_layers:
            cross_attended = layer(
                x=cross_attended,
                context=input_tensor,
            )

        out = cross_attended + identity

        return out


def get_force_reshape_func(
    force_shape: tuple[int, ...],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def func(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), *force_shape)

    return func


class SequenceResidualCrossAttention(nn.Module):
    def __init__(
        self,
        context_shape: tuple[int, ...],
        target_embedding_dim: int,
        target_max_length: int,
        apply_causal_mask: bool,
        forced_context_shape: None | tuple[int, ...],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.context_shape = torch.Size(context_shape)
        self.target_embedding_dim = target_embedding_dim
        self.target_max_length = target_max_length
        self.forced_context_shape = forced_context_shape
        self.apply_causal_mask = apply_causal_mask

        if not self.forced_context_shape:
            (
                self.reshape_func,
                self.reshaped_size,
            ) = get_reshape_to_attention_dims_func(
                input_shape=self.context_shape,
            )
        else:
            self.reshape_func = get_force_reshape_func(
                force_shape=self.forced_context_shape,
            )
            self.reshaped_size = torch.Size(self.forced_context_shape)

        self.context_embedding_dim = self.reshaped_size[1]

        self.projection_layer = CrossAttention(
            dim=self.target_embedding_dim,
            seq_length=target_max_length,
            context_dim=self.context_embedding_dim,
            context_length=context_shape[0],
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            pre_norm=False,
            apply_causal_mask=apply_causal_mask,
        )

        self.norm_1_target = nn.RMSNorm(normalized_shape=target_embedding_dim)
        self.act_1 = SwiGLU(
            in_features=target_embedding_dim,
            hidden_features=target_embedding_dim * 4,
            out_features=target_embedding_dim,
            bias=False,
        )

        self.norm_1_context = nn.RMSNorm(normalized_shape=self.context_embedding_dim)
        self.act_context = SwiGLU(
            in_features=self.context_embedding_dim,
            hidden_features=self.context_embedding_dim * 4,
            out_features=self.context_embedding_dim,
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

        encoder_mask = torch.triu(
            torch.ones(self.target_max_length, self.target_max_length) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("encoder_mask", encoder_mask)

        self.norm_2_target = nn.RMSNorm(normalized_shape=target_embedding_dim)
        self.act_2 = SwiGLU(
            in_features=target_embedding_dim,
            hidden_features=target_embedding_dim * 4,
            out_features=target_embedding_dim,
            bias=False,
        )

        self.downsample_identity = CrossAttention(
            dim=self.target_embedding_dim,
            seq_length=target_max_length,
            context_dim=self.context_embedding_dim,
            context_length=context_shape[0],
            dim_head=self.target_embedding_dim,
            dropout=0.1,
            pre_norm=False,
            apply_causal_mask=apply_causal_mask,
        )

        self.ls = LayerScale(
            dim=target_embedding_dim,
            init_values=1e-05,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self.reshape_func(context)

        identity = self.downsample_identity(
            x=x,
            context=context,
        )

        out = self.norm_1_target(x)
        out = self.act_1(out)

        out_context = self.norm_1_context(context)
        out_context = self.act_context(out_context)

        out = self.projection_layer(
            x=out,
            context=out_context,
        )

        out = self.norm_2_target(out)
        out = self.act_2(out)
        out = self.encoder(
            out,
            mask=self.encoder_mask,
        )
        out = self.ls(out)

        return out + identity


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def stable_softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        seq_length: int | None = None,
        context_length: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        pre_norm: bool = False,
        apply_causal_mask: bool = False,
    ):
        """
        Note that enabling masking here means that position i in x is only allowed to
        attend to position j in context, where j <= i. This can be useful if x and
        the context are directly related position-wise, and we want to ensure that
        the model cannot look at "future" positions through the context.

        Attention pattern (✓ = can attend, ✗ = masked):
              context
               W  X  Y  Z
          A    ✓  ✗  ✗  ✗
          B    ✓  ✓  ✗  ✗
        x C    ✓  ✓  ✓  ✗
          D    ✓  ✓  ✓  ✓

        Without masking, position i in x can attend to all positions in context.
        """
        super().__init__()
        context_dim = default(val=context_dim, d=dim)

        self.apply_causal_mask = apply_causal_mask

        self.norm = nn.RMSNorm(dim) if pre_norm else nn.Identity()
        self.context_norm = nn.RMSNorm(context_dim) if pre_norm else nn.Identity()

        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.dropout_p = dropout

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        attn_mask = None
        if self.apply_causal_mask:
            assert seq_length is not None
            assert context_length is not None
            attn_mask = ~torch.triu(
                torch.ones(
                    seq_length,
                    context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            # Expand dimensions for batch and heads: [1, 1, seq_length, context_length]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        self.attn_mask: torch.Tensor | None
        self.register_buffer("attn_mask", attn_mask)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        x_batch, seq_len, _ = x.shape
        context_batch, context_len, _ = context.shape
        assert context_batch == x_batch

        x_norm = self.norm(x)
        context_norm = self.context_norm(context)

        q = self.to_q(x_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        q = q.view(x_batch, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(x_batch, context_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(x_batch, context_len, self.heads, self.dim_head).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=self.attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = attn_output.transpose(1, 2).contiguous().view(x_batch, seq_len, -1)
        out = self.to_out(out)

        return out
