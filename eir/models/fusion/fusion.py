from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Literal, NewType, Type, cast

import torch
from torch import nn

from eir.models.fusion import fusion_default, fusion_identity, fusion_mgmoe
from eir.models.layers.mlp_layers import ResidualMLPConfig
from eir.models.meta.meta_utils import FusionModuleProtocol, al_fusion_modules
from eir.utils.logging import get_logger

al_fusion_model = Literal["pass-through", "mlp-residual", "identity", "mgmoe"]
al_fusion_model_configs = (
    ResidualMLPConfig | fusion_identity.IdentityConfig | fusion_mgmoe.MGMoEModelConfig
)

ComputedType = NewType("ComputedType", torch.Tensor)
PassThroughType = NewType("PassThroughType", Dict[str, torch.Tensor])
al_fused_features = dict[str, ComputedType | PassThroughType | torch.Tensor]

if TYPE_CHECKING:
    from eir.models.meta.meta_utils import al_input_modules

logger = get_logger(name=__name__)


def pass_through_fuse(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return features


def get_fusion_modules(
    fusion_model_type: str,
    model_config: al_fusion_model_configs,
    modules_to_fuse: "al_input_modules",
    out_feature_per_feature_extractor: Dict[str, int],
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
    any_tabular = any(i for i in output_types.values() if i in ("tabular",))
    any_sequence = any(i for i in output_types.values() if i in ("sequence",))

    any_array = any(i for i in output_types.values() if i in ("array", "image"))
    array_and_no_diffusion = any_array and not any_diffusion
    array_and_diffusion = any_array and any_diffusion

    if any_tabular or array_and_no_diffusion:
        fusion_class = get_fusion_class(fusion_model_type=fusion_model_type)
        computing_fusion_module = fusion_class(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            out_feature_per_feature_extractor=out_feature_per_feature_extractor,
        )
        fusion_modules["computed"] = computing_fusion_module

    if any_sequence or array_and_diffusion:
        model_config = fusion_identity.IdentityConfig()
        pass_through_fusion_module = fusion_identity.IdentityFusionModel(
            model_config=model_config,
            fusion_in_dim=fusion_in_dim,
            fusion_callable=pass_through_fuse,
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

    elif output_set.issubset(computed_set) and fusion_model_type == "pass-through":
        raise ValueError(
            f"When using only {computed_set} outputs, pass-through is not supported. "
            f"Got {fusion_model_type}. Kindly set the fusion_model_type "
            "to 'mlp-residual', 'mgmoe', or 'identity'."
        )

    elif (
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
) -> Type[nn.Module] | Callable:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "mlp-residual":
        return fusion_default.MLPResidualFusionModule
    elif fusion_model_type == "identity":
        return fusion_identity.IdentityFusionModel
    elif fusion_model_type == "pass-through":
        return partial(
            fusion_identity.IdentityFusionModel, fusion_callable=pass_through_fuse
        )
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")
