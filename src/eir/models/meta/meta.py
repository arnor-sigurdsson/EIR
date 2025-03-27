import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from eir.models.layers.attention_layers import TransformerBlock
from eir.models.meta.meta_utils import (
    al_fusion_modules,
    al_input_modules,
    al_output_modules,
    run_meta_forward,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import al_meta_model


class MetaModel(nn.Module):
    def __init__(
        self,
        input_modules: al_input_modules,
        fusion_modules: al_fusion_modules,
        output_modules: al_output_modules,
        fusion_to_output_mapping: dict[str, Literal["computed", "pass-through"]],
        tensor_broker: nn.ModuleDict,
    ):
        super().__init__()

        self.input_modules = input_modules
        self.fusion_modules = fusion_modules
        self.output_modules = output_modules
        self.fusion_to_output_mapping = fusion_to_output_mapping
        self.tensor_broker = tensor_broker

        apply_transformer_specific_modifications(model=self)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        output_modules_out = run_meta_forward(
            input_modules=self.input_modules,
            fusion_modules=self.fusion_modules,
            output_modules=self.output_modules,
            fusion_to_output_mapping=self.fusion_to_output_mapping,
            inputs=inputs,
        )

        return output_modules_out


def apply_transformer_specific_modifications(model: MetaModel) -> None:
    apply_scaled_residual_init(model=model, base_std=0.02)
    apply_weight_tying(model=model)


def apply_scaled_residual_init(model: MetaModel, base_std: float = 0.02) -> None:
    """
    For now, we will just apply this to TransformerBlocks, later we can look into
    for other types of layers as well.
    """
    num_layers = sum(1 for m in model.modules() if isinstance(m, TransformerBlock))

    if num_layers == 0:
        return

    scale_factor = math.sqrt(2 * num_layers)
    scaled_std = base_std / scale_factor

    logger.debug(
        f"Applying scaled initialization "
        f"(std={scaled_std:.6f}) for {num_layers} transformer blocks"
    )

    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in ["out_proj.weight", "w3.weight"]):
            nn.init.normal_(param, mean=0.0, std=scaled_std)


def apply_weight_tying(model: "al_meta_model") -> None:
    """
    Apply weight tying between input module embeddings and output module heads.
    Supports both standard and unusual weight layouts.

    TODO: Maybe make configurable in the model configuration.
    """

    if not isinstance(model, MetaModel):
        logger.warning(f"Expected MetaModel, got {type(model).__name__}")
        return

    tied_pairs = []

    for input_name, input_module in model.input_modules.items():
        if input_name not in model.output_modules:
            continue

        output_module = model.output_modules[input_name]

        embedding = getattr(input_module, "embedding", None)
        head = getattr(output_module, "head", None)

        if embedding is None or head is None:
            continue

        if embedding.weight.shape == head.weight.shape:
            logger.info(f"Tying weights for {input_name} with direct sharing")
            head.weight = embedding.weight
            tied_pairs.append(input_name)

    if tied_pairs:
        logger.info(f"Weight tying applied for modules: {', '.join(tied_pairs)}")
    else:
        logger.debug("No compatible modules found for weight tying.")
