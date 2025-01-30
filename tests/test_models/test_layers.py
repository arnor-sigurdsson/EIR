from typing import Literal

import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, sampled_from
from torch import nn

from eir.models.layers.lcl_layers import LCL, LCLResidualBlock
from eir.models.layers.projection_layers import get_1d_projection_layer


def test_get_projection_layer():
    input_dimension = 8
    target_dimension = 16

    layer = get_1d_projection_layer(input_dimension, target_dimension, "auto")
    assert isinstance(layer, LCLResidualBlock | LCL | nn.Linear)

    layer = get_1d_projection_layer(input_dimension, target_dimension, "lcl_residual")
    assert isinstance(layer, LCLResidualBlock)

    layer = get_1d_projection_layer(input_dimension, target_dimension, "lcl")
    assert isinstance(layer, LCL)

    layer = get_1d_projection_layer(input_dimension, target_dimension, "linear")
    assert isinstance(layer, nn.Linear)

    layer = get_1d_projection_layer(input_dimension, input_dimension, "linear")
    assert isinstance(layer, nn.Identity)

    with pytest.raises(ValueError) as e_info:
        get_1d_projection_layer(input_dimension, target_dimension, "invalid_type")
    assert str(e_info.value) == "Invalid projection_layer_type: invalid_type"


def _get_projection_layer_types() -> list[str]:
    projection_layer_types = ["auto", "lcl_residual", "lcl", "linear"]
    return projection_layer_types


@given(
    input_dimension=integers(min_value=1, max_value=1000),
    target_dimension=integers(min_value=1, max_value=1000),
    projection_layer_type=sampled_from(_get_projection_layer_types()),
)
@settings(deadline=None)
def test_get_projection_layer_output_dimension(
    input_dimension: int,
    target_dimension: int,
    projection_layer_type: Literal["auto", "lcl_residual", "lcl", "linear"],
) -> None:
    try:
        layer = get_1d_projection_layer(
            input_dimension=input_dimension,
            target_dimension=target_dimension,
            projection_layer_type=projection_layer_type,
        )
    except ValueError as e:
        assert str(e).startswith("Cannot create lcl")
        assert projection_layer_type in ["lcl_residual", "lcl"]
    else:
        input_tensor: torch.Tensor = torch.randn(1, input_dimension)

        output_tensor: torch.Tensor = layer(input_tensor)

        assert output_tensor.size(1) == target_dimension
