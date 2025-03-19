from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
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


def default_attention_pass_through(
    features: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return features


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

    @property
    def num_out_features(self) -> int:
        total = 0
        for projection_layer in self.projection_layers.values():
            total += projection_layer.num_out_features

        return total

    @property
    def output_shape(self) -> tuple[int, ...]:
        total_out_features = self.num_out_features
        return (total_out_features,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = {}
        for name, feature in inputs.items():
            projected[name] = self.projection_layers[name](feature)

        projected_tensors = list(projected.values())
        concatenated = torch.cat(projected_tensors, dim=1)
        out = self.transformer(concatenated)

        out_flat = out.flatten(start_dim=1)

        return out_flat
