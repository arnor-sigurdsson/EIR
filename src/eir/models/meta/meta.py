import math
from typing import Literal

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

        apply_scaled_residual_init(model=self, base_std=0.02)

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


def apply_scaled_residual_init(model: MetaModel, base_std: float = 0.02) -> None:
    """
    For now, we will just apply this to TransformerBlocks, later we can look into
    for other types of layers as well.
    """
    num_layers = sum(1 for m in model.modules() if isinstance(m, TransformerBlock))

    scale_factor = math.sqrt(2 * num_layers)
    scaled_std = base_std / scale_factor

    logger.debug(
        f"Applying scaled initialization "
        f"(std={scaled_std:.6f}) for {num_layers} transformer blocks"
    )

    for name, param in model.named_parameters():
        if any(name.endswith(suffix) for suffix in ["out_proj.weight", "w3.weight"]):
            nn.init.normal_(param, mean=0.0, std=scaled_std)
