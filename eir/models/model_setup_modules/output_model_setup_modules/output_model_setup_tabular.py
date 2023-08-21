import torch

from eir.models.model_setup_modules.output_model_setup_modules import al_output_modules
from eir.models.output.tabular.linear import (
    LinearOutputModule,
    LinearOutputModuleConfig,
)
from eir.models.output.tabular.mlp_residual import (
    ResidualMLPOutputModule,
    ResidualMLPOutputModuleConfig,
)
from eir.models.output.tabular.tabular_output_modules import TabularOutputModuleConfig
from eir.setup.output_setup_modules.tabular_output_setup import (
    al_num_outputs_per_target,
)


def get_tabular_output_module_from_model_config(
    output_model_config: TabularOutputModuleConfig,
    input_dimension: int,
    num_outputs_per_target: "al_num_outputs_per_target",
    device: str,
) -> al_output_modules:
    output_module_type = output_model_config.model_type
    model_init_config = output_model_config.model_init_config

    output_module: al_output_modules
    match output_module_type:
        case "mlp_residual":
            assert isinstance(model_init_config, ResidualMLPOutputModuleConfig)
            output_module = ResidualMLPOutputModule(
                model_config=model_init_config,
                input_dimension=input_dimension,
                num_outputs_per_target=num_outputs_per_target,
            )
        case "linear":
            assert isinstance(model_init_config, LinearOutputModuleConfig)
            output_module = LinearOutputModule(
                model_config=model_init_config,
                input_dimension=input_dimension,
                num_outputs_per_target=num_outputs_per_target,
            )
        case _:
            raise ValueError(f"Invalid output module type: {output_module_type}")

    torch_device = torch.device(device=device)
    output_module = output_module.to(device=torch_device)

    return output_module
