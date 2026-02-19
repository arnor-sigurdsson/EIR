from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    create_multi_task_blocks_with_first_adaptor_block,
)

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

    :param num_experts:
        If set, splits the shared branch into this many expert sub-branches
        (each with fc_task_dim // num_experts width). Each target learns a
        static gating weight over the experts. If None, uses a single shared
        branch (original behavior).
    """

    num_experts: int | None = None


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
        target_names, target_sizes = zip(*sorted_targets, strict=False)

        self.target_names = target_names
        self.target_sizes = list(target_sizes)

        num_experts = self.model_config.num_experts
        if num_experts is None:
            self._build_shared(input_dimension=input_dimension)
        else:
            self._build_expert(
                input_dimension=input_dimension,
                num_experts=num_experts,
            )

        self.output_identity = nn.Identity()

    def _build_shared(self, input_dimension: int) -> None:
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
            first_layer_kwargs_overload={"in_features": input_dimension},
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

    def _build_expert(self, input_dimension: int, num_experts: int) -> None:
        fc_task_dim = self.model_config.fc_task_dim
        if fc_task_dim % num_experts != 0:
            raise ValueError(
                f"fc_task_dim ({fc_task_dim}) must be divisible by "
                f"num_experts ({num_experts})."
            )

        expert_dim = fc_task_dim // num_experts
        expert_names = tuple(f"expert_{i}" for i in range(num_experts))

        expert_resblocks_kwargs: dict[str, float | int | bool] = {
            "in_features": expert_dim,
            "out_features": expert_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }

        self.expert_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=expert_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=expert_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": input_dimension},
        )

        num_targets = len(self.target_names)
        self.expert_gates = nn.Parameter(
            torch.zeros(num_targets, num_experts),
        )

        self.target_final_layers = nn.ModuleDict(
            {
                name: MLPResidualBlock(
                    in_features=expert_dim,
                    out_features=size,
                    dropout_p=self.model_config.rb_do,
                    stochastic_depth_p=self.model_config.stochastic_depth_p,
                    full_preactivation=False,
                )
                for name, size in zip(self.target_names, self.target_sizes, strict=True)
            }
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model_config.num_experts is None:
            return self._forward_shared(inputs=inputs)
        return self._forward_expert(inputs=inputs)

    def _forward_shared(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_out_tensor = self.shared_branch(inputs)
        final_out_tensor = self.output_identity(shared_out_tensor)

        split_outputs = torch.split(final_out_tensor, self.target_sizes, dim=1)

        return dict(zip(self.target_names, split_outputs, strict=False))

    def _forward_expert(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        expert_outputs = calculate_module_dict_outputs(
            input_=inputs,
            module_dict=self.expert_branches,
        )

        stacked = torch.stack(list(expert_outputs.values()), dim=1)
        gate_weights = torch.softmax(self.expert_gates, dim=1)

        outputs = {}
        for i, name in enumerate(self.target_names):
            weights = gate_weights[i].unsqueeze(0).unsqueeze(-1)
            mixed = (weights * stacked).sum(dim=1)
            mixed = self.output_identity(mixed)
            outputs[name] = self.target_final_layers[name](mixed)

        return outputs
