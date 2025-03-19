from typing import TYPE_CHECKING, Literal, NewType, cast

import torch
from torch import nn

from eir.models.fusion import (
    fusion_attention,
    fusion_default,
    fusion_identity,
    fusion_mgmoe,
)
from eir.models.fusion.fusion_default import al_features, default_fuse_features
from eir.models.fusion.fusion_identity import al_identity_features, pass_through_fuse
from eir.models.layers.mlp_layers import ResidualMLPConfig
from eir.models.meta.meta_utils import FusionModuleProtocol, al_fusion_modules
from eir.utils.logging import get_logger

al_fusion_model = Literal[
    "pass-through",
    "mlp-residual",
    "identity",
    "mgmoe",
    "attention",
]
al_fusion_model_configs = (
    ResidualMLPConfig
    | fusion_identity.IdentityConfig
    | fusion_mgmoe.MGMoEModelConfig
    | fusion_attention.AttentionFusionConfig
)

ComputedType = NewType("ComputedType", torch.Tensor)
PassThroughType = NewType("PassThroughType", dict[str, torch.Tensor])
al_fused_features = dict[str, ComputedType | PassThroughType | torch.Tensor]

if TYPE_CHECKING:
    from eir.models.meta.meta_utils import al_input_modules
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

logger = get_logger(name=__name__)


def get_fusion_modules(
    fusion_model_type: str,
    model_config: al_fusion_model_configs,
    modules_to_fuse: "al_input_modules",
    feature_dimensions_and_types: dict[str, "FeatureExtractorInfo"] | None,
    output_types: dict[str, Literal["tabular", "sequence", "array"]],
    any_diffusion: bool,
    strict: bool = True,
) -> al_fusion_modules:
    if strict:
        _check_fusion_modules(
            output_types=output_types,
            fusion_model_type=fusion_model_type,
        )

    fusion_modules: al_fusion_modules = cast(al_fusion_modules, nn.ModuleDict())

    fusion_in_dim = _get_fusion_input_dimension(modules_to_fuse=modules_to_fuse)
    any_tabular = any(i for i in output_types.values() if i in ("tabular", "survival"))
    any_sequence = any(i for i in output_types.values() if i in ("sequence",))

    any_array = any(i for i in output_types.values() if i in ("array", "image"))
    array_and_no_diffusion = any_array and not any_diffusion
    array_and_diffusion = any_array and any_diffusion

    if any_tabular or array_and_no_diffusion:
        fusion_class = get_fusion_class(fusion_model_type=fusion_model_type)

        fusion_callable: al_features | al_identity_features
        if fusion_model_type == "pass-through" or fusion_model_type == "attention":
            fusion_callable = pass_through_fuse
        else:
            fusion_callable = default_fuse_features

        computing_fusion_module = fusion_class(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            fusion_callable=fusion_callable,
            feature_dimensions_and_types=feature_dimensions_and_types,
        )
        fusion_modules["computed"] = computing_fusion_module

    if any_sequence or array_and_diffusion:
        model_config = fusion_identity.IdentityConfig()
        pass_through_fusion_module = fusion_identity.IdentityFusionModel(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            fusion_callable=pass_through_fuse,
            feature_dimensions_and_types=feature_dimensions_and_types,
        )
        fusion_modules["pass-through"] = cast(
            FusionModuleProtocol, pass_through_fusion_module
        )

    assert len(fusion_modules) > 0

    return fusion_modules


def _check_fusion_modules(
    output_types: dict[str, Literal["tabular", "sequence", "array"]],
    fusion_model_type: str,
) -> None:
    """
    Note we skip the 'array' here as it can be both computed and pass-through.
    """
    if not output_types:
        raise ValueError("output_types cannot be empty.")

    output_set = set(output_types.values())
    computed_set = {
        "tabular",
    }
    pass_through_set = {
        "sequence",
    }
    full_set = computed_set.union(pass_through_set).union({"array"})
    supported_fusion_models = {
        "mlp-residual",
        "mgmoe",
        "identity",
        "pass-through",
        "attention",
    }

    if not full_set:
        raise ValueError(
            f"Invalid output type(s). "
            f"Supported types are {computed_set.union(pass_through_set)}."
        )

    if fusion_model_type not in supported_fusion_models:
        raise ValueError(
            f"Invalid fusion_model_type. Supported types are {supported_fusion_models}."
        )

    if output_set == pass_through_set and fusion_model_type != "pass-through":
        raise ValueError(
            f"When using only {pass_through_set} outputs, "
            f"only pass-through is supported. "
            f"Got {fusion_model_type}. To use pass-through, "
            f"set fusion_model_type to 'pass-through'."
        )

    if output_set.issubset(computed_set) and fusion_model_type == "pass-through":
        raise ValueError(
            f"When using only {computed_set} outputs, pass-through is not supported. "
            f"Got {fusion_model_type}. Kindly set the fusion_model_type "
            "to 'mlp-residual', 'mgmoe', or 'identity'."
        )

    if (
        pass_through_set.intersection(output_set)
        and fusion_model_type != "pass-through"
    ):
        logger.warning(
            f"Note: When using {output_set} outputs, "
            f"the fusion model type {fusion_model_type} "
            f"is only applied to the {computed_set} outputs. "
            f"The fusion for {pass_through_set} outputs "
            "is handled by the respective output module themselves, "
            "and the feature representations are passed through"
            " directly to the output module."
        )


def _get_fusion_input_dimension(modules_to_fuse: "al_input_modules") -> int:
    fusion_in_dim = sum(i.num_out_features for i in modules_to_fuse.values())
    return fusion_in_dim


def get_fusion_class(
    fusion_model_type: str,
) -> type[FusionModuleProtocol]:
    if fusion_model_type == "mgmoe":
        return cast(type[FusionModuleProtocol], fusion_mgmoe.MGMoEModel)
    if fusion_model_type == "mlp-residual":
        return cast(type[FusionModuleProtocol], fusion_default.MLPResidualFusionModule)
    if fusion_model_type in ("identity", "pass-through"):
        return cast(type[FusionModuleProtocol], fusion_identity.IdentityFusionModel)
    if fusion_model_type == "attention":
        return cast(type[FusionModuleProtocol], fusion_attention.AttentionFusionModule)
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")
