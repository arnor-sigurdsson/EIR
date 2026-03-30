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
        layers=[2],
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
        layers=[2],
        dropout_p=0.0,
    )

    torch.manual_seed(0)
    embedding = torch.randn(batch_size, embedding_dim)
    target = torch.randn(batch_size, target_dim)

    embedding_with_rev = embedding.clone().requires_grad_(True)
    loss_rev = module(embedding=embedding_with_rev, target=target)
    loss_rev.backward()

    module.zero_grad()

    # notice manual bypass of gradient reversal by not calling forward
    embedding_no_rev = embedding.clone().requires_grad_(True)
    x = module.projection(embedding_no_rev)
    x = module.mlp_blocks(x)
    target_pred = module.output_layer(x)
    loss_no_rev = torch.nn.functional.mse_loss(input=target_pred, target=target)
    loss_no_rev.backward()

    assert embedding_with_rev.grad is not None
    assert embedding_no_rev.grad is not None
    assert not torch.allclose(
        embedding_with_rev.grad, torch.zeros_like(embedding_with_rev.grad)
    )
    assert torch.allclose(embedding_with_rev.grad, -embedding_no_rev.grad)


def _train_probe(
    embedding: torch.Tensor,
    target: torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-3,
) -> float:
    embedding_dim = embedding.shape[1]
    target_dim = target.shape[1]

    probe = torch.nn.Linear(embedding_dim, target_dim)
    probe_optimizer = torch.optim.Adam(params=probe.parameters(), lr=lr)

    embedding_detached = embedding.detach()

    for _ in range(n_steps):
        probe_optimizer.zero_grad()
        pred = probe(embedding_detached)
        loss = torch.nn.functional.mse_loss(input=pred, target=target)
        loss.backward()
        probe_optimizer.step()

    with torch.no_grad():
        pred = probe(embedding_detached)
        final_loss = torch.nn.functional.mse_loss(input=pred, target=target).item()

    return final_loss


def test_adversarial_disentanglement_e2e():
    torch.manual_seed(42)

    n_samples = 512
    task_dim = 4
    confounder_dim = 1
    embedding_dim = 32
    lr = 1e-3

    task_signal = torch.randn(n_samples, task_dim)
    confounder = torch.randn(n_samples, confounder_dim)
    inputs = torch.cat([task_signal, confounder], dim=1)
    input_dim = task_dim + confounder_dim

    targets = task_signal[:, :1] + confounder + 0.1 * torch.randn(n_samples, 1)

    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, embedding_dim),
    )
    task_head = torch.nn.Linear(embedding_dim, 1)

    task_only_params = list(encoder.parameters()) + list(task_head.parameters())
    task_only_optimizer = torch.optim.Adam(params=task_only_params, lr=lr)

    for _ in range(500):
        task_only_optimizer.zero_grad()
        embedding = encoder(inputs)
        task_pred = task_head(embedding)
        task_loss = torch.nn.functional.mse_loss(input=task_pred, target=targets)
        task_loss.backward()
        task_only_optimizer.step()

    encoder.eval()
    with torch.no_grad():
        pre_adv_embedding = encoder(inputs)

    probe_loss_before = _train_probe(
        embedding=pre_adv_embedding,
        target=confounder,
    )
    confounder_variance = confounder.var().item()
    assert probe_loss_before < 0.5 * confounder_variance, (
        f"Probe loss before adversarial training {probe_loss_before} is too high "
        f"relative to confounder variance {confounder_variance}, "
        f"confounder is not predictable from embedding to begin with"
    )

    adversarial_module = AdversarialDisentanglementModule(
        embedding_dim=embedding_dim,
        target_dim=confounder_dim,
        fc_dim=64,
        layers=[2],
        dropout_p=0.0,
    )

    all_params = (
        list(encoder.parameters())
        + list(task_head.parameters())
        + list(adversarial_module.parameters())
    )
    adv_optimizer = torch.optim.Adam(params=all_params, lr=lr)

    encoder.train()
    lambda_adv = 5.0
    for _ in range(1000):
        adv_optimizer.zero_grad()
        embedding = encoder(inputs)
        task_pred = task_head(embedding)
        task_loss = torch.nn.functional.mse_loss(input=task_pred, target=targets)
        adv_loss = adversarial_module(embedding=embedding, target=confounder)
        total_loss = task_loss + lambda_adv * adv_loss
        total_loss.backward()
        adv_optimizer.step()

    encoder.eval()
    with torch.no_grad():
        post_adv_embedding = encoder(inputs)
        task_pred = task_head(post_adv_embedding)
        final_task_loss = torch.nn.functional.mse_loss(
            input=task_pred, target=targets
        ).item()

    probe_loss_after = _train_probe(
        embedding=post_adv_embedding,
        target=confounder,
    )

    assert final_task_loss < 1.0, (
        f"Task loss {final_task_loss} too high, encoder not learning the task"
    )
    assert probe_loss_after > 2.0 * probe_loss_before, (
        f"Probe loss after adversarial training {probe_loss_after} should be much "
        f"higher than before {probe_loss_before}, indicating disentanglement"
    )


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
            layers=[2],
        )

    batch_size = 4
    for i, config in enumerate(configs):
        embedding = torch.randn(batch_size, config["embedding_dim"])
        target = torch.randn(batch_size, config["target_dim"])

        loss = modules[f"adv_{i}"](embedding=embedding, target=target)
        assert loss.item() >= 0
