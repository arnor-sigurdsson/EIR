from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

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
        self.total_outputs = sum(num_outputs_per_target.values())
        self.linear_layer = nn.Linear(
            in_features=input_dimension,
            out_features=self.total_outputs,
        )

        sorted_targets = sorted(num_outputs_per_target.items())
        target_names, target_sizes = zip(*sorted_targets)

        self.target_names = target_names
        self.target_sizes = list(target_sizes)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.linear_layer(inputs)
        split_outputs = torch.split(outputs, self.target_sizes, dim=1)
        return {
            target: output for target, output in zip(self.target_names, split_outputs)
        }


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
