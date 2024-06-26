import itertools
import math
from typing import Callable, Literal, Optional

import torch
from torch import nn

from eir.models.input.array.models_cnn import (
    CNNResidualBlock,
    ConvParamSuggestion,
    conv_output_formula,
)
from eir.models.layers.projection_layers import get_1d_projection_layer


def get_projection_layer(
    from_shape_no_batch: torch.Size,
    to_shape_no_batch: torch.Size,
    cache_fusion_type: Literal["cross-attention", "sum", "cat+conv"],
    projection_type: Literal[
        "lcl", "lcl_residual", "cnn", "linear", "pool", "sequence"
    ] = "linear",
) -> tuple[nn.Module, torch.Size]:
    """
    We have the cache_fusion_type input (currently mostly unused) and we return the
    projection size as later we might release the requirement that the
    projection layer must return exactly the same shape as the target. For example.
    we can have these valid, later implemented scenarios:

        - concat where n_channels do not match, the fusion makes the final shape
          match the target shape (only need the final output_channels to be correct).
        - cross attention where we only need to match the embedding dimension.
        - sum along a specific dimension, would work due to broadcasting.
    """

    matching_shapes = from_shape_no_batch == to_shape_no_batch
    not_ca = cache_fusion_type != "cross-attention"
    if matching_shapes and not_ca:
        return nn.Identity(), to_shape_no_batch

    match cache_fusion_type:
        case "cat+conv":
            if from_shape_no_batch[1:] == to_shape_no_batch[1:]:
                return nn.Identity(), from_shape_no_batch

    projection_layers: list[nn.Module] = []
    if projection_type in ("lcl", "linear"):
        flatten_layer = nn.Flatten(start_dim=1)
        projection_layers.append(flatten_layer)

    projection_layer: nn.Module
    match projection_type:
        case "lcl" | "linear" | "lcl_residual":
            projection_layer = get_1d_projection_layer(
                input_dimension=from_shape_no_batch.numel(),
                target_dimension=to_shape_no_batch.numel(),
                projection_layer_type=projection_type,  # type: ignore
                lcl_diff_tolerance=0,
            )
            projection_layers.append(projection_layer)
            projected_shape = to_shape_no_batch

        case "cnn":
            not_same_n_dims = len(from_shape_no_batch) != len(to_shape_no_batch)
            not_3_dims = len(from_shape_no_batch) != 3
            if not_same_n_dims and not_3_dims:
                raise ValueError(
                    f"Cannot project from {from_shape_no_batch} to {to_shape_no_batch} "
                    f"with projection type {projection_type}. Currently, CNN based "
                    f"tensor broker fusion only supports 3D inputs."
                )

            target_channels, target_height, target_width = to_shape_no_batch
            input_channels, input_height, input_width = from_shape_no_batch

            conv_params_h = get_conv_params_for_dimension(
                input_size=input_height,
                target_size=target_height,
            )

            conv_params_w = get_conv_params_for_dimension(
                input_size=input_width,
                target_size=target_width,
            )

            projection_layer = CNNResidualBlock(
                in_channels=input_channels,
                out_channels=target_channels,
                rb_do=0.0,
                dilation_w=conv_params_w.dilation,
                dilation_h=conv_params_h.dilation,
                conv_1_kernel_h=conv_params_h.kernel_size,
                conv_1_kernel_w=conv_params_w.kernel_size,
                conv_1_padding_w=conv_params_w.padding,
                conv_1_padding_h=conv_params_h.padding,
                down_stride_h=conv_params_h.stride,
                down_stride_w=conv_params_w.stride,
                stochastic_depth_p=0.0,
            )

            projection_layers.append(projection_layer)

            width_matched = conv_params_w.target_size == target_width
            height_matched = conv_params_h.target_size == target_height
            if not width_matched or not height_matched:
                pool = nn.AdaptiveAvgPool2d(output_size=(target_height, target_width))
                projection_layers.append(pool)

            projected_shape = torch.Size([target_channels, target_height, target_width])

        case "pool":
            projection_layer = nn.AdaptiveAvgPool2d(output_size=to_shape_no_batch)
            projection_layers.append(projection_layer)

            projected_shape = to_shape_no_batch

        case "sequence":
            _, target_reshaped_size = get_reshape_to_attention_dims_func(
                input_shape=to_shape_no_batch
            )

            projection_layer = SequenceProjectionLayer(
                input_shape_no_batch=from_shape_no_batch,
                target_seq_len=target_reshaped_size[0],
                target_embedding_dim=target_reshaped_size[1],
            )
            projection_layers.append(projection_layer)

            projected_shape = target_reshaped_size

        case _:
            raise ValueError(f"Invalid projection_type: {projection_type}")

    if projection_type in ("lcl", "linear"):
        unflatten_layer = nn.Unflatten(dim=1, unflattened_size=to_shape_no_batch)
        projection_layers.append(unflatten_layer)

    return nn.Sequential(*projection_layers), projected_shape


def calc_conv_params_for_dimension(
    input_size: int,
    target_size: int,
    min_threshold: float,
    max_kernel_size: int = 7,
    max_stride: int = 4,
    max_dilation: int = 3,
) -> list[ConvParamSuggestion]:
    if input_size < target_size:
        raise ValueError("Target size cannot be larger than input size")

    valid_params = []

    for kernel_size, stride, dilation in itertools.product(
        range(1, max_kernel_size + 1),
        range(1, max_stride + 1),
        range(1, max_dilation + 1),
    ):
        for padding in range(kernel_size // 2 + 1):
            output_size = conv_output_formula(
                input_size=input_size,
                padding=padding,
                dilation=dilation,
                kernel_size=kernel_size,
                stride=stride,
            )

            size_ratio = output_size / target_size

            if size_ratio >= min_threshold:
                valid_params.append(
                    ConvParamSuggestion(
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding,
                        target_size=output_size,
                    )
                )

    if not valid_params:
        raise ValueError(
            f"No valid convolutional parameters found to transform "
            f"{input_size} to {target_size} with min_threshold {min_threshold}"
        )

    return valid_params


def choose_best_params(
    params: list[ConvParamSuggestion], target_size: int
) -> ConvParamSuggestion:
    """
    Slightly confusing here that the output size of the param suggestion is
    named target_size, but it is actually the output size of the convolution.
    """
    return min(
        params,
        key=lambda p: (
            abs(p.target_size - target_size),
            p.kernel_size + p.stride + p.dilation + p.padding,
        ),
    )


def get_conv_params_for_dimension(
    input_size: int,
    target_size: int,
    min_threshold: float = 0.9,
    max_kernel_size: int = 7,
    max_stride: int = 4,
    max_dilation: int = 3,
) -> ConvParamSuggestion:
    valid_params = calc_conv_params_for_dimension(
        input_size=input_size,
        target_size=target_size,
        min_threshold=min_threshold,
        max_kernel_size=max_kernel_size,
        max_stride=max_stride,
        max_dilation=max_dilation,
    )
    best_solution = choose_best_params(
        params=valid_params,
        target_size=target_size,
    )

    return best_solution


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SequenceProjectionLayer(nn.Module):
    def __init__(
        self,
        input_shape_no_batch: torch.Size,
        target_embedding_dim: int,
        target_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.input_shape = input_shape_no_batch
        self.target_embedding_dim = target_embedding_dim
        self.target_seq_len = target_seq_len
        self.reshape_func, _ = get_reshape_to_attention_dims_func(
            input_shape=input_shape_no_batch
        )
        self.projection = self._create_projection()

    def _create_projection(self):
        n_input_dims = len(self.input_shape)
        target_embedding_dim = self.target_embedding_dim
        if n_input_dims == 1:
            input_embedding_dim = self.input_shape[0]
        elif n_input_dims == 2:
            input_embedding_dim = self.input_shape[1]
        else:
            input_embedding_dim = math.prod(self.input_shape[1:])

        layers: list[nn.Module]
        layers = [
            nn.Linear(
                in_features=input_embedding_dim,
                out_features=target_embedding_dim,
            ),
        ]

        if self.target_seq_len is not None:
            if n_input_dims == 2:
                linear_layer = nn.Linear(
                    in_features=1,
                    out_features=self.target_seq_len,
                )
            elif n_input_dims > 2 and self.input_shape[1] != self.target_seq_len:
                linear_layer = nn.Linear(
                    in_features=self.input_shape[1],
                    out_features=self.target_seq_len,
                )
            else:
                return nn.Sequential(*layers)

            # here operating with batch dim, hence offset by 1 compared to the logic
            # above to get 1 and 2
            layers.extend(
                [
                    nn.LayerNorm(normalized_shape=target_embedding_dim),
                    nn.GELU(),
                    Transpose(1, 2),
                    linear_layer,
                    Transpose(1, 2),
                ]
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reshape_func(x)
        out = self.projection(out)
        return out


def get_reshape_to_attention_dims_func(
    input_shape: torch.Size,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Size]:
    n_input_dims = len(input_shape)

    if n_input_dims == 1:

        def func(x):
            return x.unsqueeze(0)

        output_shape = torch.Size([1, input_shape[0]])
    elif n_input_dims == 2:

        def func(x):
            return x

        output_shape = input_shape
    else:

        def func(x):
            """
            Note here start at 2 since we have the batch dim when this is called.
            """
            return x.flatten(start_dim=2)

        output_shape = torch.Size(
            [input_shape[0], int(torch.prod(torch.tensor(input_shape[1:])).item())]
        )

    return func, output_shape
