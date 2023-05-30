from functools import partial
from typing import Type, Union, Dict, Callable, NewType, Literal

import torch
from torch import nn

from eir.models.fusion import fusion_mgmoe, fusion_default, fusion_identity
from eir.models.layers import ResidualMLPConfig

ComputedType = NewType("Computed", torch.Tensor)
PassThroughType = NewType("PassThrough", Dict[str, torch.Tensor])

al_fused_features = Dict[str, ComputedType | PassThroughType]


def pass_through_fuse(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return features


def get_fusion_modules(
    model_type: str,
    model_config: Union[
        ResidualMLPConfig, fusion_identity.IdentityConfig, fusion_mgmoe.MGMoEModelConfig
    ],
    modules_to_fuse: nn.ModuleDict,
    out_feature_per_feature_extractor: Dict[str, int],
    output_types: dict[str, Literal["tabular", "sequence"]],
) -> nn.ModuleDict:
    fusion_modules = nn.ModuleDict()

    fusion_in_dim = _get_fusion_input_dimension(modules_to_fuse=modules_to_fuse)

    if any(i for i in output_types.values() if i == "tabular"):
        fusion_class = get_fusion_class(fusion_model_type=model_type)
        computing_fusion_module = fusion_class(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            out_feature_per_feature_extractor=out_feature_per_feature_extractor,
        )
        fusion_modules["computed"] = computing_fusion_module

    if any(i for i in output_types.values() if i == "sequence"):
        pass_through_fusion_module = fusion_identity.IdentityFusionModel(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            fusion_callable=pass_through_fuse,
        )
        fusion_modules["pass-through"] = pass_through_fusion_module

    return fusion_modules


def _get_fusion_input_dimension(modules_to_fuse: nn.ModuleDict) -> int:
    fusion_in_dim = sum(i.num_out_features for i in modules_to_fuse.values())
    return fusion_in_dim


def get_fusion_class(
    fusion_model_type: str,
) -> Type[nn.Module] | Callable:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "default":
        return fusion_default.FusionModule
    elif fusion_model_type == "identity":
        return fusion_identity.IdentityFusionModel
    elif fusion_model_type == "pass-through":
        return partial(
            fusion_identity.IdentityFusionModel, fusion_callable=pass_through_fuse
        )
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")
