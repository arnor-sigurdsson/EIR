from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from eir.models.models_utils import calculate_module_dict_outputs

if TYPE_CHECKING:
    from eir.setup.output_setup_modules.tabular_output_setup import (
        al_num_outputs_per_target,
    )


@dataclass
class LinearOutputModuleConfig:
    pass


class LinearOutputModule(nn.Module):
    def __init__(
        self,
        model_config: LinearOutputModuleConfig,
        input_dimension: int,
        num_outputs_per_target: "al_num_outputs_per_target",
    ):
        super().__init__()

        self.model_config = model_config
        self.input_dimension = input_dimension
        self.num_outputs_per_target = num_outputs_per_target

        self.multi_task_branches = _get_linear_multi_task_branches(
            input_dimension=input_dimension,
            num_outputs_per_target=num_outputs_per_target,
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        output_modules_out = calculate_module_dict_outputs(
            input_=inputs, module_dict=self.multi_task_branches
        )

        return output_modules_out


def _get_linear_multi_task_branches(
    input_dimension: int,
    num_outputs_per_target: "al_num_outputs_per_target",
) -> nn.ModuleDict:
    multi_task_branches = nn.ModuleDict(
        {
            target: nn.Linear(
                in_features=input_dimension,
                out_features=num_outputs,
            )
            for target, num_outputs in num_outputs_per_target.items()
        }
    )

    return multi_task_branches
