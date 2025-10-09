import math
from typing import Literal

import torch
from torch import nn

from eir.models.fusion.seq_out_fusion_attention import CrossAttention
from eir.models.layers.attention_layers import SwiGLU
from eir.models.layers.cnn_layers import adjust_num_heads
from eir.models.layers.norm_layers import LayerScale
from eir.models.tensor_broker.projection_modules.sequence import (
    get_reshape_to_attention_dims_func,
)


class ProjectAndFuseLayer(nn.Module):
    def __init__(
        self,
        projection_layer: nn.Module,
        fusion_layer: nn.Module,
    ):
        super().__init__()
        self.projection_layer = projection_layer
        self.fusion_layer = fusion_layer

    def forward(
        self,
        input_tensor: torch.Tensor,
        cached_tensor: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projection_layer(cached_tensor)
        fused = self.fusion_layer(input_tensor, projected)
        return fused


def get_fusion_layer_wrapper(
    projected_shape: torch.Size,
    target_shape: torch.Size,
    cache_fusion_type: Literal["cross-attention", "sum", "cat+conv"],
    projection_layer: nn.Module,
    device: str,
) -> ProjectAndFuseLayer:
    fusion_layer = get_fusion_layer(
        projected_shape=projected_shape,
        target_shape=target_shape,
        cache_fusion_type=cache_fusion_type,
    )
    project_and_fuse_layer = ProjectAndFuseLayer(
        projection_layer=projection_layer,
        fusion_layer=fusion_layer,
    )
    project_and_fuse_layer = project_and_fuse_layer
    return project_and_fuse_layer


def get_fusion_layer(
    target_shape: torch.Size,
    projected_shape: torch.Size,
    cache_fusion_type: Literal["cross-attention", "sum", "cat+conv"],
    projection_type: Literal["lcl", "lcl_residual", "cnn", "linear", "pool"] = "linear",
) -> nn.Module:
    match cache_fusion_type:
        case "cross-attention":
            return CrossAttentionFusionLayer(
                input_shape=target_shape,
                context_shape=projected_shape,
            )
        case "sum":
            return GatedSumFusionLayer(
                input_shape=target_shape,
                context_shape=projected_shape,
            )
        case "cat+conv":
            return ConcatenationFusionLayer(
                input_shape=target_shape,
                context_shape=projected_shape,
            )
        case _:
            raise ValueError(f"Invalid cache_fusion_type: {cache_fusion_type}")


class GatedSumFusionLayer(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        context_shape: torch.Size,
        initial_input_bias: float = 0.9,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape

        ndim = len(input_shape)
        if ndim == 1:
            self.feature_axis = 1
            num_features = input_shape[0]
        elif ndim == 2:
            self.feature_axis = 2
            num_features = input_shape[1]
        elif ndim == 3:
            self.feature_axis = 1
            num_features = input_shape[0]
        else:
            raise ValueError(
                f"Unsupported input_shape ndim={ndim}. Expected 1, 2 or 3 "
                "(excluding batch dimension)."
            )

        if not (0.0 < initial_input_bias < 1.0):
            raise ValueError("initial_input_bias must be in (0, 1).")

        eps = 1e-6
        p = min(max(initial_input_bias, eps), 1.0 - eps)
        bias_val = math.log(p / (1.0 - p))

        tensor_ndim = len(input_shape) + 1
        gate_shape = [1] * tensor_ndim
        gate_shape[self.feature_axis] = num_features

        self.gate_param = nn.Parameter(
            torch.full(gate_shape, bias_val, dtype=torch.float32),
            requires_grad=True,
        )

    def forward(
        self, input_tensor: torch.Tensor, projected_context_tensor: torch.Tensor
    ) -> torch.Tensor:
        if input_tensor.shape != projected_context_tensor.shape:
            raise ValueError(
                f"Shape mismatch: input {input_tensor.shape} vs "
                f"context {projected_context_tensor.shape}"
            )

        if self.feature_axis == 2 and input_tensor.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor for sequence-like input, got {input_tensor.ndim}D."
            )
        if self.feature_axis == 1 and input_tensor.ndim not in (2, 4):
            raise ValueError(
                f"Expected 2D or 4D tensor for feature/channel input, got "
                f"{input_tensor.ndim}D."
            )

        expected_features = input_tensor.shape[self.feature_axis]
        if expected_features != self.gate_param.shape[self.feature_axis]:
            raise ValueError(
                f"Gate size {self.gate_param.shape[self.feature_axis]} does not "
                f"match feature dimension {expected_features}."
            )

        gate = torch.sigmoid(self.gate_param)

        output = (1.0 - gate) * projected_context_tensor + gate * input_tensor
        return output


class ConcatenationFusionLayer(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        context_shape: torch.Size,
    ):
        """
        Here we assume that the context has already been projected to
        (batch_size, input_seq_length, input_embedding_dim) when calling forward.
        """
        super().__init__()
        self.context_shape = context_shape
        self.input_shape = input_shape

        self.context_channels = context_shape[0]
        self.out_channels = input_shape[0]

        cat_dim = self.context_channels + self.out_channels

        self.conv_1 = nn.Conv2d(
            in_channels=cat_dim,
            out_channels=self.out_channels,
            kernel_size=1,
        )
        self.norm = nn.GroupNorm(1, self.out_channels)
        self.act = nn.GELU()

        self.ls = LayerScale(
            dim=self.out_channels,
            init_values=1e-05,
            n_dims=4,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        projected_context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        residual = input_tensor
        out = torch.cat((input_tensor, projected_context_tensor), dim=1)
        out = self.conv_1(out)
        out = self.norm(out)
        out = self.act(out)
        out = self.ls(out)

        return out + residual


class CrossAttentionFusionLayer(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        context_shape: torch.Size,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.context_shape = context_shape

        (
            self.reshape_func,
            self.reshaped_size,
        ) = get_reshape_to_attention_dims_func(
            input_shape=input_shape,
        )

        self.embedding_dim = self.reshaped_size[1]
        self.context_dim = context_shape[1]

        self.n_heads = adjust_num_heads(num_heads=8, embedding_dim=self.embedding_dim)

        self.norm_1 = nn.RMSNorm(self.embedding_dim)

        self.cross_attention = CrossAttention(
            dim=self.embedding_dim,
            dim_head=self.embedding_dim // self.n_heads,
            context_dim=self.context_dim,
            heads=self.n_heads,
        )

        self.act_1 = SwiGLU(
            in_features=self.embedding_dim,
            hidden_features=self.embedding_dim,
            out_features=self.embedding_dim,
            bias=False,
        )

        self.ls = LayerScale(
            dim=self.embedding_dim,
            init_values=1e-05,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        projected_context_tensor: torch.Tensor,
    ) -> torch.Tensor:
        identity = input_tensor

        out = self.reshape_func(input_tensor)
        out = self.norm_1(out)

        out = self.cross_attention(x=out, context=projected_context_tensor)
        out = self.act_1(out)
        out = self.ls(out)

        out = out.reshape(identity.shape)

        return out + identity
