import torch
import torch.nn as nn

from eir.models.tensor_broker.tensor_broker import (
    CachedTensor,
    attach_tensor_broker_module_injection,
)


class SimpleFusionModule(nn.Module):
    def forward(self, x: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
        return x + cached


def test_cache_dropout_always_drops_in_training():
    target_module = nn.Linear(10, 10)
    fusion_module = SimpleFusionModule()
    tensor_cache = {
        "test_layer": CachedTensor(
            tensor=torch.ones(4, 10),
            shape=torch.Size([4, 10]),
            layer_path="test_layer",
        )
    }

    attach_tensor_broker_module_injection(
        target_module=target_module,
        tensor_broker_module=fusion_module,
        tensor_cache=tensor_cache,
        tensor_cache_key="test_layer",
        cache_dropout_p=1.0,
    )

    target_module.train()
    input_tensor = torch.zeros(4, 10)

    output = target_module(input_tensor)

    expected = torch.matmul(input_tensor, target_module.weight.t()) + target_module.bias
    assert torch.allclose(output, expected), (
        "With dropout=1.0, cache should be dropped and only linear applied"
    )


def test_cache_dropout_never_drops_in_eval():
    target_module = nn.Linear(10, 10)
    fusion_module = SimpleFusionModule()

    tensor_cache = {
        "test_layer": CachedTensor(
            tensor=torch.ones(4, 10) * 5.0,
            shape=torch.Size([4, 10]),
            layer_path="test_layer",
        )
    }

    attach_tensor_broker_module_injection(
        target_module=target_module,
        tensor_broker_module=fusion_module,
        tensor_cache=tensor_cache,
        tensor_cache_key="test_layer",
        cache_dropout_p=1.0,
    )

    target_module.eval()
    input_tensor = torch.zeros(4, 10)

    output = target_module(input_tensor)

    fused_input = input_tensor + torch.ones(4, 10) * 5.0
    expected = torch.matmul(fused_input, target_module.weight.t()) + target_module.bias
    assert torch.allclose(output, expected), (
        "In eval mode, cache should always be used even with dropout=1.0"
    )


def test_cache_dropout_never_drops_when_zero():
    target_module = nn.Linear(10, 10)
    fusion_module = SimpleFusionModule()

    tensor_cache = {
        "test_layer": CachedTensor(
            tensor=torch.ones(4, 10) * 3.0,
            shape=torch.Size([4, 10]),
            layer_path="test_layer",
        )
    }

    attach_tensor_broker_module_injection(
        target_module=target_module,
        tensor_broker_module=fusion_module,
        tensor_cache=tensor_cache,
        tensor_cache_key="test_layer",
        cache_dropout_p=0.0,
    )

    target_module.train()
    input_tensor = torch.zeros(4, 10)

    output = target_module(input_tensor)

    fused_input = input_tensor + torch.ones(4, 10) * 3.0
    expected = torch.matmul(fused_input, target_module.weight.t()) + target_module.bias
    assert torch.allclose(output, expected), (
        "With dropout=0.0, cache should always be used in training"
    )


def test_cache_dropout_probabilistic():
    target_module = nn.Linear(10, 10)
    fusion_module = SimpleFusionModule()

    tensor_cache = {
        "test_layer": CachedTensor(
            tensor=torch.ones(4, 10) * 10.0,
            shape=torch.Size([4, 10]),
            layer_path="test_layer",
        )
    }

    attach_tensor_broker_module_injection(
        target_module=target_module,
        tensor_broker_module=fusion_module,
        tensor_cache=tensor_cache,
        tensor_cache_key="test_layer",
        cache_dropout_p=0.5,
    )

    target_module.train()
    torch.manual_seed(42)

    num_runs = 100
    dropped_count = 0
    used_count = 0

    for _ in range(num_runs):
        input_tensor = torch.zeros(4, 10)
        output = target_module(input_tensor)

        fused_input = input_tensor + torch.ones(4, 10) * 10.0
        expected_with_cache = (
            torch.matmul(fused_input, target_module.weight.t()) + target_module.bias
        )
        expected_without_cache = (
            torch.matmul(input_tensor, target_module.weight.t()) + target_module.bias
        )

        if torch.allclose(output, expected_without_cache):
            dropped_count += 1
        elif torch.allclose(output, expected_with_cache):
            used_count += 1

    assert dropped_count > 0, "Cache should be dropped at least once with p=0.5"
    assert used_count > 0, "Cache should be used at least once with p=0.5"
    assert 30 < dropped_count < 70, (
        f"With p=0.5, expect ~50% drops, got {dropped_count}/100"
    )
