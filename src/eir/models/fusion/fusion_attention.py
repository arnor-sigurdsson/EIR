from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from eir.models.layers.attention_layers import Transformer
from eir.models.tensor_broker.tensor_broker_projection_layers import (
    SequenceProjectionLayer,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

al_identity_features = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]

logger = get_logger(__name__)


@dataclass
class AttentionFusionConfig:
    n_layers: int = 2
    common_embedding_dim: int = 512
    n_heads: int = 8
    dim_feedforward: int | Literal["auto"] = "auto"
    dropout: float = 0.1
    attention_pooling: bool = True


def default_attention_pass_through(
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return features


class AttentionPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_head == 0, (
            f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        )

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.query = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Expand to have one query per example in the batch
        query = self.query.expand(batch_size, 1, self.d_model)

        # [batch_size, 1, n_head, head_dim]
        q = self.q_proj(query).view(batch_size, 1, self.n_head, self.head_dim)

        # [batch_size, seq_len, n_head, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # q shape: [batch_size, n_head, 1, head_dim]
        # k,v shape: [batch_size, n_head, seq_len, head_dim]

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        # attn output shape: [batch_size, n_head, 1, head_dim]

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        )

        return self.out_proj(attn_output).squeeze(1)


class AttentionFusionModule(nn.Module):
    def __init__(
        self,
        model_config: AttentionFusionConfig,
        feature_dimensions_and_types: dict[str, "FeatureExtractorInfo"],
        fusion_callable: al_identity_features = default_attention_pass_through,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.feature_dimensions_and_types = feature_dimensions_and_types
        self.fusion_callable = fusion_callable

        self.projection_layers = nn.ModuleDict()
        for name, feature in feature_dimensions_and_types.items():
            output_shape = feature.output_shape
            assert output_shape is not None
            output_size = torch.Size(output_shape)
            self.projection_layers[name] = SequenceProjectionLayer(
                input_shape_no_batch=output_size,
                target_embedding_dim=model_config.common_embedding_dim,
            )

        dim_feedforward = model_config.dim_feedforward
        dim_feedforward_computed: int
        if dim_feedforward == "auto":
            dim_feedforward_computed = model_config.common_embedding_dim * 4
        else:
            assert isinstance(dim_feedforward, int)
            dim_feedforward_computed = dim_feedforward

        self.transformer = Transformer(
            d_model=model_config.common_embedding_dim,
            nhead=model_config.n_heads,
            num_layers=model_config.n_layers,
            dim_feedforward=dim_feedforward_computed,
            dropout=model_config.dropout,
        )

        self.use_attention_pooling = model_config.attention_pooling
        if self.use_attention_pooling:
            self.attention_pooling = AttentionPooling(
                d_model=model_config.common_embedding_dim,
                n_head=model_config.n_heads,
                dropout=model_config.dropout,
            )

    @property
    def num_out_features(self) -> int:
        if self.use_attention_pooling:
            return self.model_config.common_embedding_dim
        else:
            total = 0
            for projection_layer in self.projection_layers.values():
                total += projection_layer.num_out_features
            return total

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.fusion_callable(inputs)

        projected = {}
        for name, feature in inputs.items():
            projected[name] = self.projection_layers[name](feature)

        projected_tensors = list(projected.values())
        concatenated = torch.cat(projected_tensors, dim=1)

        out = self.transformer(concatenated)

        if self.use_attention_pooling:
            out = self.attention_pooling(out)
        else:
            out = out.flatten(start_dim=1)

        return out
