import math
from typing import Type

import torch
from torch import nn
from torch.nn import init as init

from eir.models.layers.norm_layers import LayerScale


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
    key_and_dim_order: list[tuple[str, int]],
) -> torch.Tensor:
    for key, dim in key_and_dim_order:
        x = _apply_projection(x=x, projection=projections[key], dim=dim)

    return x


def create_key_and_dim_order(projections: nn.ModuleDict) -> list[tuple[str, int]]:

    key_dim_pairs = []
    for key in projections.keys():
        dim_batch_padded = int(key.split("_")[1]) + 1
        key_dim_pairs.append((key, dim_batch_padded))

    sorted_pairs = sorted(key_dim_pairs, key=lambda x: x[1])

    return sorted_pairs


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

        self.key_and_dim_order = create_key_and_dim_order(projections=self.projections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = append_dims(tensor=x, target_shape_no_batch=self.target_shape)
        x = _apply_all_projections(
            x=x,
            projections=self.projections,
            key_and_dim_order=self.key_and_dim_order,
        )
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

        self.key_and_dim_order = create_key_and_dim_order(projections=self.projections)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = retract_dims(tensor=x, target_shape_no_batch=self.target_shape)
        x = _apply_all_projections(
            x=x,
            projections=self.projections,
            key_and_dim_order=self.key_and_dim_order,
        )

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

        self.pre_match_key_and_dim_order = create_key_and_dim_order(
            projections=self.pre_match_projections,
        )

        retracted_shape = _get_retracted_shape(
            input_shape=self.factorized_shape,
            target_shape=target_shape,
        )

        self.downsample_identity = _create_projections(
            input_shape=retracted_shape,
            target_shape=target_shape,
        )
        self.downsample_identity_key_and_dim_order = create_key_and_dim_order(
            projections=self.downsample_identity,
        )

        self.projections = _create_projections(
            input_shape=retracted_shape,
            target_shape=target_shape,
        )
        self.projections_key_and_dim_order = create_key_and_dim_order(
            projections=self.projections,
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=retracted_shape)
        self.act_1 = nn.GELU()

        self.ls = LayerScale(dim=1, init_values=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = _apply_all_projections(
            x=x,
            projections=self.pre_match_projections,
            key_and_dim_order=self.pre_match_key_and_dim_order,
        )
        bottleneck = retract_dims(
            tensor=bottleneck, target_shape_no_batch=self.target_shape
        )

        out = self.norm_1(bottleneck)

        if self.full_preactivation:
            identity = _apply_all_projections(
                x=out,
                projections=self.downsample_identity,
                key_and_dim_order=self.downsample_identity_key_and_dim_order,
            )
        else:
            identity = _apply_all_projections(
                x=bottleneck,
                projections=self.downsample_identity,
                key_and_dim_order=self.downsample_identity_key_and_dim_order,
            )

        out = _apply_all_projections(
            x=out,
            projections=self.projections,
            key_and_dim_order=self.projections_key_and_dim_order,
        )

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
