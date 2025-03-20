from collections.abc import MutableMapping
from typing import (
    TYPE_CHECKING,
    Literal,
    NewType,
    Protocol,
    Union,
)

import torch

if TYPE_CHECKING:
    from eir.models.fusion.fusion import al_fused_features, al_fusion_model_configs
    from eir.models.fusion.fusion_default import al_features
    from eir.models.fusion.fusion_identity import al_identity_features
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

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
    def __init__(
        self,
        model_config: "al_fusion_model_configs",
        fusion_in_dim: int,
        fusion_callable: Union["al_features", "al_identity_features"],
        feature_dimensions_and_types: dict[str, "FeatureExtractorInfo"] | None = None,
    ) -> None: ...

    @property
    def fusion_in_dim(self) -> int: ...

    @property
    def num_out_features(self) -> int: ...

    def __call__(
        self, input: dict[str, FeatureExtractorOutType]
    ) -> "al_fused_features": ...


class OutputModuleProtocol(Protocol):
    def __call__(
        self,
        input: "al_fused_features",
    ) -> dict[str, torch.Tensor]: ...


al_input_modules = MutableMapping[
    str,
    FeatureExtractorProtocolWithL1 | FeatureExtractorProtocol,
]
al_fusion_modules = MutableMapping[
    str,
    FusionModuleProtocol,
]
al_output_modules = MutableMapping[
    str,
    OutputModuleProtocol,
]


def run_meta_forward(
    input_modules: al_input_modules,
    fusion_modules: al_fusion_modules,
    output_modules: al_output_modules,
    fusion_to_output_mapping: dict[str, Literal["computed", "pass-through"]],
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
