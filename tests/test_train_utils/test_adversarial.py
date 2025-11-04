import torch

from eir.train_utils.adversarial import (
    AdversarialDisentanglementModule,
    gradient_reversal,
)


def test_gradient_reversal_layer():
    x = torch.randn(4, 10, requires_grad=True)

    y = gradient_reversal(x)

    assert torch.allclose(y, x)
    assert y.requires_grad

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    expected_grad = torch.ones_like(x) * -1
    assert torch.allclose(x.grad, expected_grad)


def test_adversarial_module_forward():
    embedding_dim = 128
    target_dim = 64
    batch_size = 8

    module = AdversarialDisentanglementModule(
        embedding_dim=embedding_dim,
        target_dim=target_dim,
        fc_dim=256,
        n_hidden_layers=2,
        dropout_p=0.1,
    )

    embedding = torch.randn(batch_size, embedding_dim)
    target = torch.randn(batch_size, target_dim)

    loss = module(embedding=embedding, target=target)

    assert loss.ndim == 0
    assert loss.item() >= 0


def test_adversarial_module_gradient_flow():
    embedding_dim = 32
    target_dim = 16
    batch_size = 4

    module = AdversarialDisentanglementModule(
        embedding_dim=embedding_dim,
        target_dim=target_dim,
        fc_dim=64,
        n_hidden_layers=1,
        dropout_p=0.0,
    )

    embedding = torch.randn(batch_size, embedding_dim, requires_grad=True)
    target = torch.randn(batch_size, target_dim)

    loss = module(embedding=embedding, target=target)
    loss.backward()

    assert embedding.grad is not None
    assert embedding.grad.mean().item() < 0


def test_multiple_adversaries_shapes():
    configs = [
        {"embedding_dim": 128, "target_dim": 64},
        {"embedding_dim": 256, "target_dim": 32},
        {"embedding_dim": 512, "target_dim": 128},
    ]

    modules = {}
    for i, config in enumerate(configs):
        modules[f"adv_{i}"] = AdversarialDisentanglementModule(
            embedding_dim=config["embedding_dim"],
            target_dim=config["target_dim"],
            fc_dim=128,
            n_hidden_layers=2,
        )

    batch_size = 4
    for i, config in enumerate(configs):
        embedding = torch.randn(batch_size, config["embedding_dim"])
        target = torch.randn(batch_size, config["target_dim"])

        loss = modules[f"adv_{i}"](embedding=embedding, target=target)
        assert loss.item() >= 0
