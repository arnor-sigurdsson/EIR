from typing import Dict, Literal

import torch
from torch import nn

from eir.models.meta.meta_utils import (
    al_fusion_modules,
    al_input_modules,
    al_output_modules,
    run_meta_forward,
)


class MetaModel(nn.Module):
    def __init__(
        self,
        input_modules: al_input_modules,
        fusion_modules: al_fusion_modules,
        output_modules: al_output_modules,
        fusion_to_output_mapping: Dict[str, Literal["computed", "pass-through"]],
        tensor_broker: nn.ModuleDict,
    ):
        super().__init__()

        self.input_modules = input_modules
        self.fusion_modules = fusion_modules
        self.output_modules = output_modules
        self.fusion_to_output_mapping = fusion_to_output_mapping
        self.tensor_broker = tensor_broker

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:

        output_modules_out = run_meta_forward(
            input_modules=self.input_modules,
            fusion_modules=self.fusion_modules,
            output_modules=self.output_modules,
            fusion_to_output_mapping=self.fusion_to_output_mapping,
            inputs=inputs,
        )

        return output_modules_out
