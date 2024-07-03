from typing import TYPE_CHECKING, Dict, Literal, MutableMapping, NewType, Protocol

import torch

if TYPE_CHECKING:
    from eir.models.fusion.fusion import al_fused_features, al_fusion_model_configs

FeatureExtractorOutType = NewType("FeatureExtractorOutType", torch.Tensor)


class FeatureExtractorProtocol(Protocol):
    @property
    def num_out_features(self) -> int: ...

    def __call__(self, input: torch.Tensor) -> FeatureExtractorOutType: ...


class FeatureExtractorProtocolWithL1(Protocol):
    @property
    def num_out_features(self) -> int: ...

    @property
    def l1_penalized_weights(self) -> torch.Tensor: ...

    def __call__(self, input: torch.Tensor) -> FeatureExtractorOutType: ...


class FusionModuleProtocol(Protocol):
    model_config: "al_fusion_model_configs"

    @property
    def fusion_in_dim(self) -> int: ...

    @property
    def num_out_features(self) -> int: ...

    def __call__(
        self, input: Dict[str, FeatureExtractorOutType]
    ) -> "al_fused_features": ...


class OutputModuleProtocol(Protocol):
    def __call__(
        self,
        input: "al_fused_features",
    ) -> Dict[str, torch.Tensor]: ...


al_input_modules = MutableMapping[
    str, FeatureExtractorProtocolWithL1 | FeatureExtractorProtocol
]
al_fusion_modules = MutableMapping[str, FusionModuleProtocol]
al_output_modules = MutableMapping[str, OutputModuleProtocol]


def run_meta_forward(
    input_modules: al_input_modules,
    fusion_modules: al_fusion_modules,
    output_modules: al_output_modules,
    fusion_to_output_mapping: Dict[str, Literal["computed", "pass-through"]],
    inputs: dict[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:

    feature_extractors_out = {}
    for module_name, cur_input_module in input_modules.items():
        module_input = inputs[module_name]
        feature_extractors_out[module_name] = cur_input_module(module_input)

    fused_features = {}
    for output_type, fusion_module in fusion_modules.items():
        fused_features[output_type] = fusion_module(feature_extractors_out)

    output_modules_out = {}
    for output_name, output_module in output_modules.items():
        cur_fusion_target = fusion_to_output_mapping[output_name]
        corresponding_fused_features = fused_features[cur_fusion_target]

        key = f"__extras_{output_name}"
        if key in inputs:
            corresponding_fused_features[key] = inputs[key]

        cur_output = output_module(corresponding_fused_features)
        output_modules_out[output_name] = cur_output

    return output_modules_out
