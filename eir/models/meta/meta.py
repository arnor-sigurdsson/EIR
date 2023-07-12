from typing import Dict, Literal, MutableMapping, NewType, Protocol

import torch
from torch import nn

from eir.models.fusion.fusion import al_fused_features, al_fusion_model_configs

FeatureExtractorOutType = NewType("FeatureExtractorOutType", torch.Tensor)


class FeatureExtractorProtocol(Protocol):
    @property
    def num_out_features(self) -> int:
        ...

    def __call__(self, input: torch.Tensor) -> FeatureExtractorOutType:
        ...


class FeatureExtractorProtocolWithL1(Protocol):
    @property
    def num_out_features(self) -> int:
        ...

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        ...

    def __call__(self, input: torch.Tensor) -> FeatureExtractorOutType:
        ...


class FusionModuleProtocol(Protocol):
    model_config: al_fusion_model_configs

    def __call__(self, input: Dict[str, FeatureExtractorOutType]) -> al_fused_features:
        ...


class OutputModuleProtocol(Protocol):
    def __call__(self, input: al_fused_features) -> Dict[str, torch.Tensor]:
        ...


al_input_modules = MutableMapping[
    str, FeatureExtractorProtocolWithL1 | FeatureExtractorProtocol
]
al_fusion_modules = MutableMapping[str, FusionModuleProtocol]
al_output_modules = MutableMapping[str, OutputModuleProtocol]


class MetaModel(nn.Module):
    def __init__(
        self,
        input_modules: al_input_modules,
        fusion_modules: al_fusion_modules,
        output_modules: al_output_modules,
        fusion_to_output_mapping: Dict[str, Literal["computed", "pass-through"]],
    ):
        super().__init__()

        self.input_modules = input_modules
        self.fusion_modules = fusion_modules
        self.output_modules = output_modules
        self.fusion_to_output_mapping = fusion_to_output_mapping

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, torch.Tensor]]:
        feature_extractors_out = {}
        for module_name, module_input in inputs.items():
            cur_input_module = self.input_modules[module_name]
            feature_extractors_out[module_name] = cur_input_module(module_input)

        fused_features = {}
        for output_type, fusion_module in self.fusion_modules.items():
            fused_features[output_type] = fusion_module(feature_extractors_out)

        output_modules_out = {}
        for output_name, output_module in self.output_modules.items():
            cur_fusion_target = self.fusion_to_output_mapping[output_name]
            corresponding_fused_features = fused_features[cur_fusion_target]

            cur_output = output_module(corresponding_fused_features)
            output_modules_out[output_name] = cur_output

        return output_modules_out
