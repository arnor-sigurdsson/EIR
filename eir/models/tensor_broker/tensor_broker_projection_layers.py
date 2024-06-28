import itertools
import math
from typing import Callable, Literal, Optional, Type

import torch
import torch.nn.init as init
from torch import nn

from eir.models.input.array.models_cnn import (
    CNNResidualBlock,
    ConvParamSuggestion,
    conv_output_formula,
    solve_for_padding,
)
from eir.models.layers.norm_layers import LayerScale
from eir.models.layers.projection_layers import get_1d_projection_layer


def get_projection_layer(
    from_shape_no_batch: torch.Size,
    to_shape_no_batch: torch.Size,
    cache_fusion_type: Literal["cross-attention", "sum", "cat+conv"],
    projection_type: Literal[
        "lcl", "lcl_residual", "cnn", "linear", "pool", "sequence", "grouped_linear"
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
    if projection_type in ("lcl", "linear", "lcl_residual"):
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

        case "grouped_linear":
            projection_layer = GroupedLinearProjectionWrapper(
                input_shape=from_shape_no_batch,
                target_shape=to_shape_no_batch,
            )
            projection_layers.append(projection_layer)

            projected_shape = to_shape_no_batch

        case _:
            raise ValueError(f"Invalid projection_type: {projection_type}")

    if projection_type in ("lcl", "linear", "lcl_residual"):
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
    stride_to_kernel_ratio: float = 1.0,
) -> list[ConvParamSuggestion]:
    if input_size < target_size:
        raise ValueError("Target size cannot be larger than input size")

    valid_params = []

    for kernel_size, stride, dilation in itertools.product(
        range(1, max_kernel_size + 1),
        range(1, max_stride + 1),
        range(1, max_dilation + 1),
    ):
        if stride > kernel_size * stride_to_kernel_ratio:
            continue

        padding = solve_for_padding(
            input_size=input_size,
            target_size=target_size,
            dilation=dilation,
            stride=stride,
            kernel_size=kernel_size,
        )

        if padding is None:
            continue

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
    min_threshold: float = 1.0,
    max_kernel_size: int = 33,
    max_stride: int = 32,
    max_dilation: int = 4,
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


def append_dims(tensor: torch.Tensor, target_shape_no_batch: list[int]) -> torch.Tensor:
    # exclude batch dimension
    current_dims = len(tensor.shape) - 1
    target_dims = len(target_shape_no_batch)
    if current_dims >= target_dims:
        return tensor

    new_shape = list(tensor.shape) + [1] * (target_dims - current_dims)
    return tensor.view(new_shape)


def retract_dims(
    tensor: torch.Tensor, target_shape_no_batch: list[int]
) -> torch.Tensor:
    """
    For example with:
        - tensor: [64, 3, 4]
        - target_shape_no_batch: [12]

        - current_dims = 3 - 1 = 2
        - target_dims = 1
        - Start flattening at dim 1

    We have 1 dimension per sample, so we flatten the last 2 dimensions to get:
        - [64, 12]
    """
    # exclude batch dimension
    current_dims = len(tensor.shape) - 1
    target_dims = len(target_shape_no_batch)

    if current_dims <= target_dims:
        return tensor

    start_dim = target_dims

    return tensor.flatten(start_dim=start_dim)


def _get_retracted_shape(input_shape: list[int], target_shape: list[int]) -> list[int]:
    if len(input_shape) <= len(target_shape):
        return input_shape

    n_dims_to_keep = len(target_shape) - 1
    dims_to_flatten = input_shape[n_dims_to_keep:]
    flattened_dim = math.prod(dims_to_flatten)

    retracted_shape = input_shape[:n_dims_to_keep] + [flattened_dim]

    return retracted_shape


def _apply_all_projections(
    x: torch.Tensor,
    projections: nn.ModuleDict,
) -> torch.Tensor:
    for i, proj in projections.items():
        dim = int(i.split("_")[1]) + 1
        x = _apply_projection(x=x, projection=proj, dim=dim)

    return x


def _apply_projection(
    x: torch.Tensor,
    projection: nn.Module,
    dim: int,
) -> torch.Tensor:

    x = x.transpose(dim, -1)
    x = projection(x)
    x = x.transpose(dim, -1)
    return x


class GroupedLinearProjectionWrapper(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        target_shape: torch.Size,
        full_preactivation: bool = True,
        use_factorized_projection: bool = False,
    ):
        """
        Note: We sometimes get gradient explosions w/o the full_preactivation here,
        e.g. when sending tensors to the last layer in the network. The pre-activation
        seems to alleviate this somewhat.
        """
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.full_preactivation = full_preactivation
        self.use_factorized_projection = use_factorized_projection

        input_shape_list = list(input_shape)
        target_shape_list = list(target_shape)

        self.projection: nn.Module
        identity_projection_class: Type[nn.Module]
        if len(input_shape) <= len(target_shape):
            self.projection = GroupedUpProjectionLayer(
                input_shape=input_shape_list,
                target_shape=target_shape_list,
            )
            identity_projection_class = GroupedUpProjectionLayer

        elif len(input_shape) > len(target_shape):
            down_class: Type[nn.Module]
            if self.use_factorized_projection:
                down_class = GroupedDownProjectionLayerFactorized
            else:
                down_class = GroupedDownProjectionLayer

            self.projection = down_class(
                input_shape=input_shape_list,
                target_shape=target_shape_list,
            )
            identity_projection_class = down_class
        else:
            raise ValueError()

        self.project_identity: nn.Module
        if input_shape == target_shape:
            self.project_identity = nn.Identity()
        else:
            self.project_identity = identity_projection_class(
                input_shape=input_shape_list,
                target_shape=target_shape_list,
            )

        self.norm = nn.LayerNorm(normalized_shape=input_shape, eps=1e-05)
        self.activation = nn.GELU()

        self.ls = LayerScale(dim=1, init_values=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)

        if self.full_preactivation:
            identity = self.project_identity(out)
        else:
            identity = self.project_identity(x)

        out = self.projection(out)
        out = self.activation(out)

        out = self.ls(out)

        return out + identity


class GroupedUpProjectionLayer(nn.Module):
    def __init__(
        self,
        input_shape: list[int],
        target_shape: list[int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.projections = _create_projections(
            input_shape=input_shape,
            target_shape=target_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = append_dims(tensor=x, target_shape_no_batch=self.target_shape)
        x = _apply_all_projections(x=x, projections=self.projections)
        return x


class GroupedDownProjectionLayer(nn.Module):
    def __init__(
        self,
        input_shape: list[int],
        target_shape: list[int],
    ):
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape

        retracted_shape = _get_retracted_shape(
            input_shape=self.input_shape,
            target_shape=target_shape,
        )
        self.projections = _create_projections(
            input_shape=retracted_shape,
            target_shape=target_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = retract_dims(tensor=x, target_shape_no_batch=self.target_shape)
        x = _apply_all_projections(x=x, projections=self.projections)

        return x


class GroupedDownProjectionLayerFactorized(nn.Module):
    def __init__(
        self,
        input_shape: list[int],
        target_shape: list[int],
        full_preactivation: bool = True,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.full_preactivation = full_preactivation

        (
            self.factorized_shape,
            self.pre_match_projections,
        ) = get_pre_dim_matching_projections(
            input_shape=input_shape,
            target_shape=target_shape,
        )

        retracted_shape = _get_retracted_shape(
            input_shape=self.factorized_shape,
            target_shape=target_shape,
        )

        self.downsample_identity = _create_projections(
            input_shape=retracted_shape,
            target_shape=target_shape,
        )

        self.projections = _create_projections(
            input_shape=retracted_shape,
            target_shape=target_shape,
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=retracted_shape)
        self.act_1 = nn.GELU()

        self.ls = LayerScale(dim=1, init_values=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = _apply_all_projections(x=x, projections=self.pre_match_projections)
        bottleneck = retract_dims(
            tensor=bottleneck, target_shape_no_batch=self.target_shape
        )

        out = self.norm_1(bottleneck)

        if self.full_preactivation:
            identity = _apply_all_projections(
                x=out,
                projections=self.downsample_identity,
            )
        else:
            identity = _apply_all_projections(
                x=bottleneck,
                projections=self.downsample_identity,
            )

        out = _apply_all_projections(x=out, projections=self.projections)

        out = self.act_1(out)
        out = self.ls(out)

        return out + identity


def scaled_kaiming_uniform_(
    tensor,
    a=0,
    mode="fan_in",
    nonlinearity="leaky_relu",
    scale=1.0,
) -> torch.Tensor:
    fan = init._calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std * scale
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.scale = scale
        self.scaled_reset_parameters()

    def scaled_reset_parameters(self) -> None:
        scaled_kaiming_uniform_(self.weight, a=math.sqrt(5), scale=self.scale)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = (1 / math.sqrt(fan_in)) * self.scale if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


def _create_projections(
    input_shape: list[int],
    target_shape: list[int],
) -> nn.ModuleDict:
    projections = nn.ModuleDict()

    n_missing_dims = len(target_shape) - len(input_shape)
    padded_input_shape = input_shape + [1] * n_missing_dims

    total_target_elements = math.prod(target_shape)

    input_and_target_dim_iter = zip(padded_input_shape, target_shape)
    for dim_index, (input_dim, target_dim) in enumerate(input_and_target_dim_iter):
        if input_dim == target_dim:
            continue

        layers: list[nn.Module] = []

        scale_factor = math.sqrt(target_dim / total_target_elements)
        linear_layer = ScaledLinear(
            in_features=input_dim,
            out_features=target_dim,
            scale=scale_factor,
        )

        layers.append(linear_layer)

        projections[f"dim_{dim_index}"] = nn.Sequential(*layers)

    return projections


def get_pre_dim_matching_projections(
    input_shape: list[int], target_shape: list[int]
) -> tuple[list[int], nn.ModuleDict]:
    factorized_shape = _calculate_factorized_shape(
        shape_to_factorize=input_shape,
        target_shape=target_shape,
    )

    projections = _create_projections(
        input_shape=input_shape,
        target_shape=factorized_shape,
    )

    return factorized_shape, projections


def _calculate_factorized_shape(
    shape_to_factorize: list[int], target_shape: list[int]
) -> list[int]:
    """
    shape_to_factorize: [64, 64, 16]
    target_shape: [2]
    n_dims_to_factorize = 3 - 1 = 2
    factorized = [64, 64, 16][:2 - 1] = [64]
    for dim in [64, 16]:
        factorized.append(int(sqrt(dim))

    Then we only factorize the last n dimensions not matching, applying int(sqrt())

    So we e.g. return [64, 8, 4]
    """

    n_dims_to_factorize = len(shape_to_factorize) - len(target_shape)
    cutoff_index = n_dims_to_factorize - 1

    factorized = shape_to_factorize[:cutoff_index]

    for dim in shape_to_factorize[cutoff_index:]:
        factorized.append(int(math.sqrt(dim)))

    return factorized
