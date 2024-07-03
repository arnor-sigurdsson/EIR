import pytest
import torch
import torch.nn as nn

from eir.models.tensor_broker.projection_modules.grouped_linear import (
    GroupedDownProjectionLayer,
    GroupedDownProjectionLayerFactorized,
    GroupedUpProjectionLayer,
    _calculate_factorized_shape,
    _get_retracted_shape,
    append_dims,
    get_pre_dim_matching_projections,
    retract_dims,
)


@pytest.fixture
def random_tensor():
    def _random_tensor(shape):
        return torch.rand(shape)

    return _random_tensor


@pytest.mark.parametrize(
    "input_shape,target_shape,expected_shape",
    [
        ([64, 3], [2, 3, 4], [64, 3, 1, 1]),  # 2 dims added
        ([64], [5, 6, 7], [64, 1, 1, 1]),  # 3 dims added
        ([64, 3, 4], [2, 3], [64, 3, 4]),  # 0 dims added
    ],
)
def test_append_dims(
    random_tensor,
    input_shape,
    target_shape,
    expected_shape,
):
    input_tensor = random_tensor(input_shape)
    result = append_dims(tensor=input_tensor, target_shape_no_batch=target_shape)
    assert result.shape == tuple(expected_shape)


@pytest.mark.parametrize(
    "input_shape,target_shape,expected_shape",
    [
        ([64, 3, 4], [12], [64, 12]),
        ([64, 5, 6, 7], [210], [64, 210]),
        ([64, 2, 3], [2, 3], [64, 2, 3]),
    ],
)
def test_retract_dims(random_tensor, input_shape, target_shape, expected_shape):
    input_tensor = random_tensor(input_shape)
    result = retract_dims(tensor=input_tensor, target_shape_no_batch=target_shape)
    assert result.shape == tuple(expected_shape)


@pytest.mark.parametrize(
    "input_shape,target_shape,expected_shape",
    [
        ([2, 32, 16], [1], [1024]),
        ([10, 20, 30, 40], [1, 1], [10, 24000]),
        ([5, 5, 5], [5, 5], [5, 25]),
    ],
)
def test_get_retracted_shape(input_shape, target_shape, expected_shape):
    result = _get_retracted_shape(input_shape=input_shape, target_shape=target_shape)
    assert result == expected_shape


def test_grouped_up_projection_layer():
    input_shape = [32, 64]
    target_shape = [32, 64, 128]
    layer = GroupedUpProjectionLayer(
        input_shape=input_shape,
        target_shape=target_shape,
    )
    input_tensor = torch.rand(10, 32, 64)  # Batch size of 10
    output = layer(input_tensor)
    assert output.shape == (10, 32, 64, 128)


def test_grouped_down_projection_layer_factorized():
    input_shape = [64, 32, 16]
    target_shape = [128, 8]
    layer = GroupedDownProjectionLayerFactorized(
        input_shape=input_shape,
        target_shape=target_shape,
    )
    input_tensor = torch.rand(10, 64, 32, 16)  # Batch size of 10
    output = layer(input_tensor)
    assert output.shape == (10, 128, 8)


def test_grouped_down_projection_layer():
    input_shape = [64, 32, 16]
    target_shape = [128, 8]
    layer = GroupedDownProjectionLayer(
        input_shape=input_shape,
        target_shape=target_shape,
    )
    input_tensor = torch.rand(10, 64, 32, 16)  # Batch size of 10
    output = layer(input_tensor)
    assert output.shape == (10, 128, 8)


def test_get_pre_dim_matching_projections():
    input_shape = [64, 64, 16]
    target_shape = [2]
    factorized_shape, projections = get_pre_dim_matching_projections(
        input_shape, target_shape
    )
    assert len(factorized_shape) == len(input_shape)
    assert isinstance(projections, nn.ModuleDict)


@pytest.mark.parametrize(
    "shape_to_factorize,target_shape,expected_shape",
    [
        ([64, 64, 16], [2], [64, 8, 4]),
        ([32, 32, 32], [4], [32, 5, 5]),
        ([100, 100], [10], [10, 10]),
    ],
)
def test_calculate_factorized_shape(shape_to_factorize, target_shape, expected_shape):
    result = _calculate_factorized_shape(
        shape_to_factorize=shape_to_factorize,
        target_shape=target_shape,
    )
    assert result == expected_shape


def test_full_projection_pipeline():
    input_tensor = torch.rand(10, 64, 32, 16)  # Batch size of 10
    up_layer = GroupedUpProjectionLayer(
        input_shape=[64, 32, 16], target_shape=[64, 32, 16, 8]
    )
    down_layer = GroupedDownProjectionLayerFactorized(
        input_shape=[64, 32, 16, 8], target_shape=[128, 8]
    )

    intermediate = up_layer(input_tensor)
    assert intermediate.shape == (10, 64, 32, 16, 8)

    output = down_layer(intermediate)
    assert output.shape == (10, 128, 8)
