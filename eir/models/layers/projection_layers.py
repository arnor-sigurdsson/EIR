import math
from typing import Literal, Sequence, Tuple, Type

import numpy as np
from torch import nn

from eir.models.layers.lcl_layers import LCL, LCLResidualBlock


def get_1d_projection_layer(
    input_dimension: int,
    target_dimension: int,
    projection_layer_type: Literal[
        "auto",
        "lcl",
        "lcl_residual",
        "linear",
        "cnn",
    ] = "auto",
    lcl_diff_tolerance: int = 0,
) -> LCLResidualBlock | LCL | nn.Linear | nn.Identity:
    """
    Note: These currently (and possibly will only continue to) support
          projection from 1D to 1D.
    """
    if projection_layer_type == "auto":
        if input_dimension == target_dimension:
            return nn.Identity()

        lcl_residual_projection = get_lcl_projection_layer(
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            layer_type="lcl_residual",
            diff_tolerance=lcl_diff_tolerance,
        )
        if lcl_residual_projection is not None:
            return lcl_residual_projection

        lcl_projection = get_lcl_projection_layer(
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            layer_type="lcl",
            diff_tolerance=lcl_diff_tolerance,
        )
        if lcl_projection is not None:
            return lcl_projection

        return nn.Linear(
            in_features=input_dimension,
            out_features=target_dimension,
            bias=True,
        )

    elif projection_layer_type == "lcl_residual":
        layer = get_lcl_projection_layer(
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            layer_type="lcl_residual",
            diff_tolerance=lcl_diff_tolerance,
        )
        if layer is None:
            raise ValueError(
                f"Cannot create lcl_residual projection layer for "
                f"input_dimension={input_dimension} and "
                f"target_dimension={target_dimension} for projection. "
                f"Try using projection_layer_type='auto'."
            )
        else:
            return layer

    elif projection_layer_type == "lcl" or projection_layer_type == "lcl_residual":
        layer = get_lcl_projection_layer(
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            layer_type=projection_layer_type,
            diff_tolerance=lcl_diff_tolerance,
        )
        if layer is None:
            raise ValueError(
                f"Cannot create lcl projection layer for "
                f"input_dimension={input_dimension} and "
                f"target_dimension={target_dimension} for projection. "
                f"Try using projection_layer_type='auto'."
            )
        else:
            return layer

    elif projection_layer_type == "linear":
        if input_dimension == target_dimension:
            return nn.Identity()
        else:
            return nn.Linear(
                in_features=input_dimension,
                out_features=target_dimension,
                bias=True,
            )

    elif projection_layer_type == "cnn":
        raise NotImplementedError()

    else:
        raise ValueError(f"Invalid projection_layer_type: {projection_layer_type}")


def get_lcl_projection_layer(
    input_dimension: int,
    target_dimension: int,
    layer_type: Literal["lcl_residual", "lcl"] = "lcl_residual",
    kernel_width_candidates: Sequence[int] = tuple(range(1, 1024 + 1)),
    out_feature_sets_candidates: Sequence[int] = tuple(range(1, 512 + 1)),
    diff_tolerance: int = 0,
) -> LCLResidualBlock | LCL | None:
    layer_class: Type[LCLResidualBlock] | Type[LCL]
    match layer_type:
        case "lcl_residual":
            layer_class = LCLResidualBlock
            n_lcl_layers = 2
        case "lcl":
            layer_class = LCL
            n_lcl_layers = 1
        case _:
            raise ValueError(f"Unknown layer type: {layer_type}")

    search_func = _find_best_lcl_kernel_width_and_out_feature_sets
    solution = search_func(
        input_dimension=input_dimension,
        target_dimension=target_dimension,
        n_layers=n_lcl_layers,
        kernel_width_candidates=kernel_width_candidates,
        out_feature_sets_candidates=out_feature_sets_candidates,
        diff_tolerance=diff_tolerance,
    )

    if solution is None:
        return None

    best_kernel_size, best_out_feature_sets = solution
    best_layer = layer_class(
        in_features=input_dimension,
        kernel_size=best_kernel_size,
        out_feature_sets=best_out_feature_sets,
    )

    return best_layer


def _find_best_lcl_kernel_width_and_out_feature_sets(
    input_dimension: int,
    target_dimension: int,
    n_layers: int,
    kernel_width_candidates: Sequence[int] = tuple(range(1, 1024 + 1)),
    out_feature_sets_candidates: Sequence[int] = tuple(range(1, 64 + 1)),
    diff_tolerance: int = 0,
) -> Tuple[int, int] | None:
    best_diff = np.Inf
    best_kernel_width = None
    best_out_feature_sets = None

    def _compute(
        input_dimension_: int, kernel_width_: int, out_feature_sets_: int
    ) -> int:
        num_chunks_ = int(math.ceil(input_dimension_ / kernel_width_))
        out_features_ = num_chunks_ * out_feature_sets_
        return out_features_

    for out_feature_sets in out_feature_sets_candidates:
        for kernel_width in kernel_width_candidates:
            if kernel_width > input_dimension:
                continue

            out_features = input_dimension
            for n in range(n_layers):
                out_features = _compute(
                    input_dimension_=out_features,
                    kernel_width_=kernel_width,
                    out_feature_sets_=out_feature_sets,
                )

            if out_features < target_dimension:
                continue

            diff = abs(out_features - target_dimension)

            if diff < best_diff:
                best_diff = diff
                best_kernel_width = kernel_width
                best_out_feature_sets = out_feature_sets

            if diff <= diff_tolerance:
                break

    if best_diff != 0:
        return None

    if best_kernel_width is None:
        return None

    assert best_kernel_width is not None
    assert best_out_feature_sets is not None

    return best_kernel_width, best_out_feature_sets
