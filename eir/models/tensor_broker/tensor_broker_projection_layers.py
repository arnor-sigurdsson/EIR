import torch
from torch import nn

from eir.models.input.array.models_cnn import CNNResidualBlock
from eir.models.layers.projection_layers import get_1d_projection_layer
from eir.models.tensor_broker.projection_modules.cnn import (
    get_conv_params_for_dimension,
)
from eir.models.tensor_broker.projection_modules.grouped_linear import (
    GroupedLinearProjectionWrapper,
)
from eir.models.tensor_broker.projection_modules.interpolate import (
    InterpolateProjection,
)
from eir.models.tensor_broker.projection_modules.sequence import (
    SequenceProjectionLayer,
    get_reshape_to_attention_dims_func,
)
from eir.setup.schema_modules.tensor_broker_schemas import (
    al_broker_fusion_types,
    al_broker_projection_types,
)


def get_projection_layer(
    from_shape_no_batch: torch.Size,
    to_shape_no_batch: torch.Size,
    cache_fusion_type: al_broker_fusion_types,
    projection_type: al_broker_projection_types,
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

        case "interpolate":

            if len(from_shape_no_batch) != 3:
                raise ValueError(
                    f"Cannot project from {from_shape_no_batch} to {to_shape_no_batch} "
                    f"with projection type {projection_type}. "
                    f"Currently, Interpolate based "
                    f"tensor broker fusion only supports 3D inputs."
                )

            target_channels, target_height, target_width = to_shape_no_batch
            input_channels, *_ = from_shape_no_batch

            projection_layer = InterpolateProjection(
                in_channels=input_channels,
                out_channels=target_channels,
                output_size=(target_height, target_width),
            )
            projection_layers.append(projection_layer)

            projected_shape = to_shape_no_batch

        case _:
            raise ValueError(f"Invalid projection_type: {projection_type}")

    if projection_type in ("lcl", "linear", "lcl_residual"):
        unflatten_layer = nn.Unflatten(dim=1, unflattened_size=to_shape_no_batch)
        projection_layers.append(unflatten_layer)

    return nn.Sequential(*projection_layers), projected_shape
