from collections.abc import MutableMapping
from typing import (
    TYPE_CHECKING,
    Any,
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
        **kwargs: object,
    ) -> None: ...

    @property
    def fusion_in_dim(self) -> int: ...

    @property
    def num_out_features(self) -> int: ...

    @property
    def per_output_group(self) -> bool: ...

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
    modalities_to_skip: set[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    feature_extractors_out = {}
    for module_name, cur_input_module in input_modules.items():
        module_input = inputs[module_name]
        feature_extractors_out[module_name] = cur_input_module(module_input)

    if modalities_to_skip:
        fusion_inputs = {
            k: v
            for k, v in feature_extractors_out.items()
            if k not in modalities_to_skip
        }
    else:
        fusion_inputs = feature_extractors_out

    fused_features = {}
    for output_type, fusion_module in fusion_modules.items():
        fused_features[output_type] = fusion_module(fusion_inputs)

    output_modules_out = {}
    for output_name, output_module in output_modules.items():
        cur_fusion_target = fusion_to_output_mapping[output_name]
        fused = fused_features[cur_fusion_target]

        corresponding_fused_features: Any
        fusion_module = fusion_modules[cur_fusion_target]

        # MGMoE (and similar) returns a dict keyed by output group name
        # pass-through fusion also returns a dict but keyed by input name.
        # When an output name matches an input name (e.g. image output task),
        # but expects to be passed a dict (i.e. not the extracted tensor),
        # we therefore must only extract per-group outputs for per_output_group modules
        if (
            getattr(fusion_module, "per_output_group", False)
            and isinstance(fused, dict)
            and output_name in fused
        ):
            corresponding_fused_features = fused[output_name]
        else:
            corresponding_fused_features = fused

        key = f"__extras_{output_name}"
        if key in inputs:
            corresponding_fused_features[key] = inputs[key]

        cur_output = output_module(corresponding_fused_features)
        output_modules_out[output_name] = cur_output

    return output_modules_out
