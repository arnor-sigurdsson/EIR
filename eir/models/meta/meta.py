from typing import Dict

import torch
from torch import nn

from eir.models.models_base import (
    calculate_module_dict_outputs,
)


class MetaModel(nn.Module):
    def __init__(
        self,
        input_modules: nn.ModuleDict,
        fusion_module: nn.Module,
        output_modules: nn.ModuleDict,
    ):
        super().__init__()

        self.input_modules = input_modules
        self.fusion_module = fusion_module
        self.output_modules = output_modules

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        feature_extractors_out = {}
        for module_name, module_input in inputs.items():
            module = self.input_modules[module_name]
            feature_extractors_out[module_name] = module(module_input)

        fused_features = self.fusion_module(feature_extractors_out)

        output_modules_out = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.output_modules
        )

        return output_modules_out
