import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from eir.models.fusion.fusion_attention import UniDirectionalCrossAttention
from eir.models.layers.cnn_layers import adjust_num_heads
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
    project_and_fuse_layer = project_and_fuse_layer.to(device=device)
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

        input_init = torch.Tensor([math.log(initial_input_bias)])
        projected_init = torch.Tensor([math.log(1 - initial_input_bias)])
        self.gate_input = nn.Parameter(input_init, requires_grad=True)
        self.gate_projected = nn.Parameter(projected_init, requires_grad=True)

    def forward(
        self, input_tensor: torch.Tensor, projected_context_tensor: torch.Tensor
    ) -> torch.Tensor:
        gates = F.softmax(torch.stack([self.gate_input, self.gate_projected]), dim=0)

        scaled_input = input_tensor * gates[0]
        scaled_projected = projected_context_tensor * gates[1]

        return scaled_input + scaled_projected


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

    def forward(
        self, input_tensor: torch.Tensor, projected_context_tensor: torch.Tensor
    ) -> torch.Tensor:
        residual = input_tensor
        out = torch.cat((input_tensor, projected_context_tensor), dim=1)
        out = self.conv_1(out)
        out = self.norm(out)
        out = self.act(out)
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

        self.norm_1 = nn.LayerNorm(self.embedding_dim)
        self.act_1 = nn.GELU()

        self.cross_attention = UniDirectionalCrossAttention(
            dim=self.embedding_dim,
            dim_head=self.embedding_dim // self.n_heads,
            context_dim=self.context_dim,
            heads=self.n_heads,
        )

    def forward(
        self, input_tensor: torch.Tensor, projected_context_tensor: torch.Tensor
    ) -> torch.Tensor:
        identity = input_tensor

        out = self.reshape_func(input_tensor)
        out = self.norm_1(out)

        out = self.cross_attention(x=out, context=projected_context_tensor)
        out = self.act_1(out)

        out = out.reshape(identity.shape)

        return out + identity
