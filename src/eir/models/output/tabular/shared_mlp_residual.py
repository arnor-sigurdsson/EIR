from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_utils import create_multi_task_blocks_with_first_adaptor_block

if TYPE_CHECKING:
    from eir.setup.output_setup_modules.tabular_output_setup import (
        al_num_outputs_per_target,
    )


@dataclass
class SharedResidualMLPOutputModuleConfig(ResidualMLPConfig):
    """
    :param layers:
        Number of residual MLP residual blocks to use in the output module.

    :param fc_task_dim:
        Number of hidden nodes in each MLP residual block.

    :param rb_do:
        Dropout in each MLP residual block.

    :param fc_do:
        Dropout before final layer.

    :param stochastic_depth_p:
        Stochastic depth probability (probability of dropping input)
        for each residual block.
    """


class SharedResidualMLPOutputModule(nn.Module):
    def __init__(
        self,
        model_config: SharedResidualMLPOutputModuleConfig,
        input_dimension: int,
        num_outputs_per_target: "al_num_outputs_per_target",
    ):
        super().__init__()

        self.model_config = model_config
        self.input_dimension = input_dimension
        self.num_outputs_per_target = num_outputs_per_target

        self.total_outputs = sum(self.num_outputs_per_target.values())
        sorted_targets = sorted(num_outputs_per_target.items())
        target_names, target_sizes = zip(*sorted_targets)

        self.target_names = target_names
        self.target_sizes = list(target_sizes)

        task_resblocks_kwargs: dict[str, float | int | bool] = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }

        shared_branch_dict = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=("shared",),
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=task_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": self.input_dimension},
        )

        assert len(shared_branch_dict) == 1
        shared_branch_module = shared_branch_dict["shared"]

        final_block = MLPResidualBlock(
            in_features=self.model_config.fc_task_dim,
            out_features=self.total_outputs,
            dropout_p=self.model_config.rb_do,
            stochastic_depth_p=self.model_config.stochastic_depth_p,
            full_preactivation=False,
        )

        self.shared_branch = nn.Sequential(shared_branch_module, final_block)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:

        shared_out_tensor = self.shared_branch(inputs)

        split_outputs = torch.split(shared_out_tensor, self.target_sizes, dim=1)

        return {
            target: output for target, output in zip(self.target_names, split_outputs)
        }
