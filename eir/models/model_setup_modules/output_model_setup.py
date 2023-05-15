from typing import Type

from eir.models.output.output_module_setup import OutputModuleConfig
from eir.models.output.linear import LinearOutputModelConfig, LinearOutputModule
from eir.models.output.mlp_residual import (
    ResidualMLPOutputModelConfig,
    ResidualMLPOutputModule,
)
from eir.setup.output_setup import al_num_outputs_per_target

al_output_module_init_configs = ResidualMLPOutputModelConfig | LinearOutputModelConfig
al_output_module_classes = Type[ResidualMLPOutputModule] | Type[LinearOutputModule]
al_output_modules = ResidualMLPOutputModule | LinearOutputModule


def get_tabular_output_module_from_model_config(
    output_model_config: OutputModuleConfig,
    input_dimension: int,
    num_outputs_per_target: "al_num_outputs_per_target",
    device: str,
) -> al_output_modules:
    class_map = _get_output_module_type_class_map()

    output_module_type = output_model_config.model_type
    cur_output_module_class = class_map[output_module_type]

    output_module = cur_output_module_class(
        model_config=output_model_config.model_init_config,
        input_dimension=input_dimension,
        num_outputs_per_target=num_outputs_per_target,
    )

    output_module = output_module.to(device=device)

    return output_module


def _get_output_module_type_class_map() -> dict[str, al_output_module_classes]:
    mapping = {
        "mlp_residual": ResidualMLPOutputModule,
        "linear": LinearOutputModule,
    }

    return mapping
