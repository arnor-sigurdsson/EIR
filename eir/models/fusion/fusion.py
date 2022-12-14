from typing import Type, Union, Dict
from functools import partial

from torch import nn
import torch

from eir.models.fusion import fusion_mgmoe, fusion_default, fusion_identity
from eir.models.layers import ResidualMLPConfig


def pass_through_fuse(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return features


def get_fusion_module(
    model_type: str,
    model_config: Union[
        ResidualMLPConfig, fusion_identity.IdentityConfig, fusion_mgmoe.MGMoEModelConfig
    ],
    modules_to_fuse: nn.ModuleDict,
    out_feature_per_feature_extractor: Dict[str, int],
) -> nn.Module:

    fusion_in_dim = _get_fusion_input_dimension(modules_to_fuse=modules_to_fuse)
    fusion_class = get_fusion_class(fusion_model_type=model_type)
    fusion_module = fusion_class(
        model_config=model_config,
        fusion_in_dim=fusion_in_dim,
        out_feature_per_feature_extractor=out_feature_per_feature_extractor,
    )

    return fusion_module


def _get_fusion_input_dimension(modules_to_fuse: nn.ModuleDict) -> int:
    fusion_in_dim = sum(i.num_out_features for i in modules_to_fuse.values())
    return fusion_in_dim


def get_fusion_class(
    fusion_model_type: str,
) -> Type[nn.Module]:
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
