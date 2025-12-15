from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from eir.train_utils.optim import _get_all_params_to_optimize, get_optimizer


@pytest.fixture
def create_test_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.bn = nn.BatchNorm1d(10)

        def forward(self, x):
            return x

    return TestModel()


@pytest.fixture
def create_test_loss_module():
    class TestLossModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5))
            self.bias = nn.Parameter(torch.randn(5))

        def forward(self, x, y):
            return x

    return TestLossModule()


@pytest.fixture
def create_test_extra_modules():
    class ExtraModule1(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(3))

        def forward(self, x):
            return x

    class ExtraModule2(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 5)

        def forward(self, x):
            return x

    return {"module1": ExtraModule1(), "module2": ExtraModule2()}


def test_get_all_params_basic_model_only(create_test_model):
    model = create_test_model
    weight_decay = 0.01

    def dummy_loss(x, y):
        return x

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=dummy_loss,
        extra_modules=None,
    )

    all_model_params = set(model.parameters())
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert len(all_model_params) == len(collected_params)
    assert all_model_params == collected_params


def test_get_all_params_with_loss_module(create_test_model, create_test_loss_module):
    model = create_test_model
    loss_module = create_test_loss_module
    weight_decay = 0.01

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=loss_module,
        extra_modules=None,
    )

    all_expected_params = set(model.parameters()) | set(loss_module.parameters())
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert len(all_expected_params) == len(collected_params)
    assert all_expected_params == collected_params


def test_get_all_params_with_extra_modules(
    create_test_model, create_test_extra_modules
):
    model = create_test_model
    extra_modules = create_test_extra_modules
    weight_decay = 0.01

    def dummy_loss(x, y):
        return x

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=dummy_loss,
        extra_modules=extra_modules,
    )

    all_extra_params = set()
    for module in extra_modules.values():
        all_extra_params.update(module.parameters())

    all_expected_params = set(model.parameters()) | all_extra_params
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert len(all_expected_params) == len(collected_params)
    assert all_expected_params == collected_params


def test_get_all_params_complete(
    create_test_model, create_test_loss_module, create_test_extra_modules
):
    model = create_test_model
    loss_module = create_test_loss_module
    extra_modules = create_test_extra_modules
    weight_decay = 0.01

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=loss_module,
        extra_modules=extra_modules,
    )

    all_extra_params = set()
    for module in extra_modules.values():
        all_extra_params.update(module.parameters())

    all_expected_params = (
        set(model.parameters()) | set(loss_module.parameters()) | all_extra_params
    )
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert len(all_expected_params) == len(collected_params)
    assert all_expected_params == collected_params


def test_get_all_params_weight_decay_groups(create_test_model):
    model = create_test_model
    weight_decay = 0.05

    def dummy_loss(x, y):
        return x

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=dummy_loss,
        extra_modules=None,
    )

    assert len(param_groups) == 3

    decay_group = param_groups[0]
    embedding_group = param_groups[1]
    no_decay_group = param_groups[2]

    assert decay_group["weight_decay"] == weight_decay
    assert embedding_group["weight_decay"] == weight_decay
    assert no_decay_group["weight_decay"] == 0.0

    decay_params = len(decay_group["params"])
    embedding_params = len(embedding_group["params"])
    no_decay_params = len(no_decay_group["params"])

    assert decay_params == 2
    assert embedding_params == 0
    assert no_decay_params == 4


def test_get_all_params_no_missing_parameters_complex_model():
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 10)
            self.embedding = nn.Embedding(100, 32)
            self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2)

        def forward(self, x):
            return x

    model = ComplexModel()
    weight_decay = 0.01

    def dummy_loss(x, y):
        return x

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=dummy_loss,
        extra_modules=None,
    )

    all_model_params = set(model.parameters())
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert len(all_model_params) == len(collected_params)
    assert all_model_params == collected_params

    total_param_count = sum(p.numel() for p in model.parameters())
    collected_param_count = sum(
        p.numel() for group in param_groups for p in group["params"]
    )
    assert total_param_count == collected_param_count


def test_get_all_params_empty_loss_module():
    class EmptyLossModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.tensor(0.0)

    model = nn.Linear(10, 5)
    loss_module = EmptyLossModule()
    weight_decay = 0.01

    param_groups = _get_all_params_to_optimize(
        model=model,
        weight_decay=weight_decay,
        loss_callable=loss_module,
        extra_modules=None,
    )

    all_model_params = set(model.parameters())
    collected_params = set()
    for group in param_groups:
        collected_params.update(group["params"])

    assert all_model_params == collected_params


@pytest.fixture
def create_mock_global_config():
    def _create_config(optimizer_name="adamw", lr=0.001, wd=0.01, b1=0.9, b2=0.999):
        mock_config = MagicMock()
        mock_config.opt.optimizer = optimizer_name
        mock_config.opt.lr = lr
        mock_config.opt.wd = wd
        mock_config.opt.b1 = b1
        mock_config.opt.b2 = b2
        return mock_config

    return _create_config


def test_get_optimizer_registers_all_model_params(
    create_test_model, create_mock_global_config
):
    model = create_test_model
    global_config = create_mock_global_config(optimizer_name="adamw", wd=0.01)

    def dummy_loss(x, y):
        return x

    optimizer = get_optimizer(
        model=model,
        loss_callable=dummy_loss,
        global_config=global_config,
        extra_modules=None,
    )

    all_model_params = set(model.parameters())
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group["params"])

    assert len(all_model_params) == len(optimizer_params)
    assert all_model_params == optimizer_params


def test_get_optimizer_registers_model_and_loss_params(
    create_test_model, create_test_loss_module, create_mock_global_config
):
    model = create_test_model
    loss_module = create_test_loss_module
    global_config = create_mock_global_config(optimizer_name="adamw", wd=0.01)

    optimizer = get_optimizer(
        model=model,
        loss_callable=loss_module,
        global_config=global_config,
        extra_modules=None,
    )

    all_expected_params = set(model.parameters()) | set(loss_module.parameters())
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group["params"])

    assert len(all_expected_params) == len(optimizer_params)
    assert all_expected_params == optimizer_params


def test_get_optimizer_registers_all_params_complete(
    create_test_model,
    create_test_loss_module,
    create_test_extra_modules,
    create_mock_global_config,
):
    model = create_test_model
    loss_module = create_test_loss_module
    extra_modules = create_test_extra_modules
    global_config = create_mock_global_config(optimizer_name="adamw", wd=0.01)

    optimizer = get_optimizer(
        model=model,
        loss_callable=loss_module,
        global_config=global_config,
        extra_modules=extra_modules,
    )

    all_extra_params = set()
    for module in extra_modules.values():
        all_extra_params.update(module.parameters())

    all_expected_params = (
        set(model.parameters()) | set(loss_module.parameters()) | all_extra_params
    )
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group["params"])

    assert len(all_expected_params) == len(optimizer_params)
    assert all_expected_params == optimizer_params


@pytest.mark.parametrize("optimizer_name", ["adamw", "adam", "sgdm"])
def test_get_optimizer_different_optimizers(
    create_test_model, create_mock_global_config, optimizer_name
):
    model = create_test_model
    global_config = create_mock_global_config(optimizer_name=optimizer_name, wd=0.01)

    def dummy_loss(x, y):
        return x

    optimizer = get_optimizer(
        model=model,
        loss_callable=dummy_loss,
        global_config=global_config,
        extra_modules=None,
    )

    all_model_params = set(model.parameters())
    optimizer_params = set()
    for param_group in optimizer.param_groups:
        optimizer_params.update(param_group["params"])

    assert len(all_model_params) == len(optimizer_params)
    assert all_model_params == optimizer_params


def test_get_optimizer_param_count_matches(create_mock_global_config):
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.bn = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 10)
            self.embedding = nn.Embedding(100, 32)

        def forward(self, x):
            return x

    model = ComplexModel()
    global_config = create_mock_global_config(optimizer_name="adamw", wd=0.01)

    def dummy_loss(x, y):
        return x

    optimizer = get_optimizer(
        model=model,
        loss_callable=dummy_loss,
        global_config=global_config,
        extra_modules=None,
    )

    total_model_param_count = sum(p.numel() for p in model.parameters())
    total_optimizer_param_count = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"]
    )

    assert total_model_param_count == total_optimizer_param_count
