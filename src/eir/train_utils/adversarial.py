from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock
from eir.models.tensor_broker.tensor_broker_projection_layers import (
    get_projection_layer,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(name=__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    From https://arxiv.org/abs/1409.7495
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def gradient_reversal(x: torch.Tensor) -> torch.Tensor:
    return GradientReversalLayer.apply(x)


class AdversarialDisentanglementModule(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        target_dim: int,
        fc_dim: int = 128,
        layers: list[int] | None = None,
        dropout_p: float = 0.1,
        stochastic_depth_p: float = 0.0,
        projection_type: Literal[
            "linear", "lcl", "lcl_residual", "mlp_residual", "grouped_linear"
        ] = "linear",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.hidden_dim = fc_dim
        self.layers = layers if layers is not None else [2]
        self.dropout_p = dropout_p
        self.projection_type = projection_type

        from_shape = torch.Size([embedding_dim])
        to_shape = torch.Size([fc_dim])

        self.projection, _ = get_projection_layer(
            from_shape_no_batch=from_shape,
            to_shape_no_batch=to_shape,
            cache_fusion_type="sum",
            projection_type=projection_type,
        )

        mlp_blocks = []
        n_blocks = self.layers[0] if self.layers else 0
        for i in range(n_blocks):
            mlp_blocks.append(
                MLPResidualBlock(
                    in_features=fc_dim,
                    out_features=fc_dim,
                    dropout_p=dropout_p,
                    full_preactivation=(i == 0),
                    stochastic_depth_p=stochastic_depth_p,
                )
            )

        self.mlp_blocks = nn.Sequential(*mlp_blocks) if mlp_blocks else nn.Identity()

        self.output_layer = nn.Linear(
            in_features=fc_dim,
            out_features=target_dim,
            bias=True,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        embedding_reversed = gradient_reversal(embedding)

        x = self.projection(embedding_reversed)
        x = self.mlp_blocks(x)
        target_pred = self.output_layer(x)

        adv_loss = nn.functional.mse_loss(input=target_pred, target=target)
        return adv_loss


def hook_add_adversarial_losses(
    experiment: Any,
    state: dict[str, Any],
    adversarial_state: dict[str, Any],
    *args,
    **kwargs,
) -> dict[str, Any]:
    if not experiment.model.training:
        return state

    if "losses" not in state:
        state["losses"] = {}

    adversarial_configs = adversarial_state["configs"]
    adversarial_cache = adversarial_state["cache"]

    device = experiment.fabric.device

    if adversarial_state["modules"] is None:
        logger.debug("Creating adversarial modules based on tensor shapes.")

        adversarial_modules = {}
        for adv_config in adversarial_configs:
            if not adv_config.enabled:
                continue

            embedding_cache_key = f"{adv_config.name}_embedding"
            target_cache_key = f"{adv_config.name}_target"

            embedding_sample = adversarial_cache[embedding_cache_key]
            target_sample = adversarial_cache[target_cache_key]

            embedding_dim = embedding_sample.view(embedding_sample.size(0), -1).size(1)
            target_dim = target_sample.view(target_sample.size(0), -1).size(1)

            module = AdversarialDisentanglementModule(
                embedding_dim=embedding_dim,
                target_dim=target_dim,
                fc_dim=adv_config.fc_dim,
                layers=adv_config.layers,
                dropout_p=adv_config.dropout_p,
                stochastic_depth_p=adv_config.stochastic_depth_p,
                projection_type=adv_config.projection_type,
            )
            module = module.to(device)
            adversarial_modules[adv_config.name] = module

            logger.debug(
                "Created adversarial module '%s' with embedding_dim=%d, target_dim=%d",
                adv_config.name,
                embedding_dim,
                target_dim,
            )

        adversarial_state["modules"] = adversarial_modules

        total_params = 0
        for module_name, module in adversarial_modules.items():
            total_params += sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            experiment.optimizer.add_param_group({"params": module.parameters()})
            logger.debug(
                "Added adversarial module '%s' parameters to optimizer", module_name
            )

        logger.info(
            "Added %d adversarial modules to optimizer with %d parameters",
            len(adversarial_modules),
            total_params,
        )

    adversarial_modules = adversarial_state["modules"]

    current_iteration = state["iteration"]

    scaled_adversarial_losses = []
    for adv_config in adversarial_configs:
        if not adv_config.enabled:
            continue

        adversarial_module = adversarial_modules[adv_config.name]

        embedding_cache_key = f"{adv_config.name}_embedding"
        target_cache_key = f"{adv_config.name}_target"

        embedding = adversarial_cache[embedding_cache_key]
        target = adversarial_cache[target_cache_key]

        embedding_flat = embedding.view(embedding.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        adv_loss = adversarial_module(
            embedding=embedding_flat,
            target=target_flat,
        )

        warmup_factor = min(1.0, current_iteration / adv_config.warmup_steps)
        current_lambda = adv_config.lambda_adv * warmup_factor

        scaled_adv_loss = current_lambda * adv_loss

        scaled_adversarial_losses.append(scaled_adv_loss)

        state["losses"][f"adversarial_{adv_config.name}"] = adv_loss.item()
        state["losses"][f"adversarial_{adv_config.name}_lambda"] = current_lambda

    if scaled_adversarial_losses:
        total_adversarial_loss = torch.stack(scaled_adversarial_losses).mean()
    else:
        total_adversarial_loss = 0.0

    state["loss"] = state["loss"] + total_adversarial_loss

    if isinstance(total_adversarial_loss, torch.Tensor):
        state["losses"]["adversarial_total"] = total_adversarial_loss.item()

    return state
