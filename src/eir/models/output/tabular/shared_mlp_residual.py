import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.layers.mlp_layers import (
    BatchedMLPResidualBlock,
    MLPResidualBlock,
    ResidualMLPConfig,
)
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

    :param expert_groups:
        If set, enables batched per-group output computation. Maps group names
        to the list of target column names belonging to that group. Each group
        gets independent MLP weights, but all groups are computed in parallel
        via batched matrix multiplication. Group names must match the keys
        produced by the fusion module (e.g. expert branch names from the
        input module).
    """

    expert_groups: dict[str, list[str]] | None = None


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

        if model_config.expert_groups is not None:
            self._build_batched(
                input_dimension=input_dimension,
                expert_groups=model_config.expert_groups,
            )
        else:
            self._build_shared(input_dimension=input_dimension)

        # these are used as TB entry points
        self.input_identity = nn.Identity()
        self.output_identity = nn.Identity()

        if model_config.expert_groups is not None:
            for group_name in self._batched_group_names:
                self.add_module(f"{group_name}_output", nn.Identity())

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
            out_features=self.model_config.fc_task_dim,
            dropout_p=self.model_config.rb_do,
            stochastic_depth_p=self.model_config.stochastic_depth_p,
            full_preactivation=False,
        )

        final_norm = nn.RMSNorm(self.model_config.fc_task_dim)
        final_proj = nn.Linear(self.model_config.fc_task_dim, self.total_outputs)

        self.shared_branch = nn.Sequential(
            shared_branch_module, final_block, final_norm, final_proj
        )

    def _build_batched(
        self,
        input_dimension: int,
        expert_groups: dict[str, list[str]],
    ) -> None:
        fc_task_dim = self.model_config.fc_task_dim
        n_groups = len(expert_groups)

        self._batched_group_names = sorted(expert_groups.keys())
        self._batched_group_targets: list[list[str]] = [
            sorted(expert_groups[name]) for name in self._batched_group_names
        ]

        grouped_targets = {
            t for targets in self._batched_group_targets for t in targets
        }
        missing = set(self.num_outputs_per_target.keys()) - grouped_targets
        if missing:
            raise ValueError(
                f"expert_groups does not cover all output targets. "
                f"Missing targets ({len(missing)}): "
                f"{sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''}. "
                f"Either add these targets to an expert group or "
                f"remove them from the output columns."
            )
        self._batched_group_output_sizes: list[list[int]] = [
            [self.num_outputs_per_target[t] for t in targets]
            for targets in self._batched_group_targets
        ]
        self._batched_group_total_outputs: list[int] = [
            sum(sizes) for sizes in self._batched_group_output_sizes
        ]

        num_blocks = self.model_config.layers[0]

        batched_blocks: list[BatchedMLPResidualBlock] = []
        batched_blocks.append(
            BatchedMLPResidualBlock(
                n_groups=n_groups,
                in_features=input_dimension,
                out_features=fc_task_dim,
                dropout_p=self.model_config.rb_do,
                stochastic_depth_p=self.model_config.stochastic_depth_p,
                full_preactivation=True,
            )
        )

        for _ in range(num_blocks - 1):
            batched_blocks.append(
                BatchedMLPResidualBlock(
                    n_groups=n_groups,
                    in_features=fc_task_dim,
                    out_features=fc_task_dim,
                    dropout_p=self.model_config.rb_do,
                    stochastic_depth_p=self.model_config.stochastic_depth_p,
                    full_preactivation=False,
                )
            )

        batched_blocks.append(
            BatchedMLPResidualBlock(
                n_groups=n_groups,
                in_features=fc_task_dim,
                out_features=fc_task_dim,
                dropout_p=self.model_config.rb_do,
                stochastic_depth_p=self.model_config.stochastic_depth_p,
                full_preactivation=False,
            )
        )

        self.batched_blocks = nn.ModuleList(batched_blocks)

        self.batched_final_norm_weight = nn.Parameter(torch.ones(n_groups, fc_task_dim))

        max_outputs = max(self._batched_group_total_outputs)
        self.batched_proj_weight = nn.Parameter(
            torch.empty(n_groups, max_outputs, fc_task_dim)
        )
        self.batched_proj_bias = nn.Parameter(torch.empty(n_groups, 1, max_outputs))
        for g in range(n_groups):
            nn.init.kaiming_uniform_(self.batched_proj_weight[g], a=math.sqrt(5))
            fan_in = fc_task_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.batched_proj_bias[g], -bound, bound)

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.model_config.expert_groups is not None:
            assert isinstance(inputs, dict)
            return self._forward_batched(inputs=inputs)

        assert isinstance(inputs, torch.Tensor)
        return self._forward_shared(inputs=inputs)

    def _forward_shared(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = self.input_identity(inputs)
        shared_out_tensor = self.shared_branch(inputs)
        final_out_tensor = self.output_identity(shared_out_tensor)

        split_outputs = torch.split(final_out_tensor, self.target_sizes, dim=1)

        return dict(zip(self.target_names, split_outputs, strict=False))

    def _forward_batched(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # (N_groups, B, D)
        stacked = torch.stack(
            [inputs[name] for name in self._batched_group_names], dim=0
        )

        # input_identity sees (B, N_groups * D) as an entry point, later we can
        # make this group aware, but then also e.g. needs TB updates
        n, b, d = stacked.shape
        stacked = (
            self.input_identity(stacked.permute(1, 0, 2).reshape(b, n * d))
            .reshape(b, n, d)
            .permute(1, 0, 2)
        )

        for block in self.batched_blocks:
            stacked = block(stacked)

        # per-group RMSNorm
        variance = stacked.pow(2).mean(dim=-1, keepdim=True)
        normed = (
            stacked
            * torch.rsqrt(variance + 1e-8)
            * self.batched_final_norm_weight.unsqueeze(1)
        )

        # batched projection: (N, B, D) @ (N, D, max_out) -> (N, B, max_out)
        projected = (
            torch.bmm(normed, self.batched_proj_weight.transpose(1, 2))
            + self.batched_proj_bias
        )

        # same as input_identity,  sees (B, N_groups * D)
        n_out, b_out, max_out = projected.shape
        projected = (
            self.output_identity(
                projected.permute(1, 0, 2).reshape(b_out, n_out * max_out)
            )
            .reshape(b_out, n_out, max_out)
            .permute(1, 0, 2)
        )

        results: dict[str, torch.Tensor] = {}
        for i, (targets, sizes) in enumerate(
            zip(
                self._batched_group_targets,
                self._batched_group_output_sizes,
                strict=False,
            )
        ):
            group_out = projected[i, :, : sum(sizes)]
            cur_expert_name = self._batched_group_names[i]
            group_out = getattr(self, f"{cur_expert_name}_output")(group_out)
            split = torch.split(group_out, sizes, dim=1)

            for target_name, target_tensor in zip(targets, split, strict=False):
                results[target_name] = target_tensor

        return results
