import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.fusion.fusion_default import al_features, default_fuse_features
from eir.models.layers.mlp_layers import MLPResidualBlock
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    create_multi_task_blocks_with_first_adaptor_block,
)

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo


@dataclass
class MGMoEModelConfig:
    """
    Note that this module by default uses sum fusion with input projection.


    :param layers:
        A sequence of two int values controlling the number of residual MLP blocks in
        the network. The first item (i.e. ``layers[0]``) refers to the number of blocks
        in the expert branches. The second item (i.e. ``layers[1]``) refers to the
        number of blocks in the predictor branches.

    :param fc_task_dim:
       Number of hidden nodes in all residual blocks (both expert and predictor) of
       the network.

    :param mg_num_experts:
        Number of multi gate experts to use.

    :param rb_do:
        Dropout in all MLP residual blocks (both expert and predictor).

    :param fc_do:
        Dropout before the last FC layer.

    :param stochastic_depth_p:
        Probability of dropping input.
    """

    layers: Sequence[int] = field(default_factory=lambda: [1, 1])
    fc_task_dim: int = 64

    mg_num_experts: int = 8

    rb_do: float = 0.00
    fc_do: float = 0.00

    stochastic_depth_p: float = 0.00


class MGMoEModel(nn.Module):
    def __init__(
        self,
        model_config: MGMoEModelConfig,
        fusion_in_dim: int,
        output_group_names: Sequence[str],
        fusion_callable: al_features = default_fuse_features,
        feature_dimensions_and_types: dict[str, "FeatureExtractorInfo"] | None = None,
        **kwargs,
    ):
        super().__init__()

        if not output_group_names:
            raise ValueError("output_group_names must be non-empty.")

        self.model_config = model_config
        self.fusion_in_dim = fusion_in_dim
        self.fusion_callable = fusion_callable

        self.num_experts = self.model_config.mg_num_experts
        self.use_sum_fusion = feature_dimensions_and_types is not None

        if self.use_sum_fusion:
            assert feature_dimensions_and_types is not None
            self.expert_projections = nn.ModuleDict()
            for expert_idx in range(self.num_experts):
                expert_projs = nn.ModuleDict()
                for name, info in feature_dimensions_and_types.items():
                    output_dim = info.output_dimension
                    expert_projs[name] = nn.Sequential(
                        nn.RMSNorm(normalized_shape=output_dim),
                        nn.Linear(
                            in_features=output_dim,
                            out_features=self.model_config.fc_task_dim,
                        ),
                        nn.GELU(),
                    )
                self.expert_projections[f"expert_{expert_idx}"] = expert_projs

            expert_in_dim = self.model_config.fc_task_dim
        else:
            expert_in_dim = fusion_in_dim

        expert_names = tuple(f"expert_{i}" for i in range(self.num_experts))
        layer_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }
        self.expert_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=expert_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=layer_kwargs,
            first_layer_kwargs_overload={
                "full_preactivation": True,
                "in_features": expert_in_dim,
            },
        )

        self.gate_logits = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(self.num_experts))
                for name in output_group_names
            }
        )

    @property
    def per_output_group(self) -> bool:
        return True

    @property
    def num_out_features(self) -> int:
        return self.model_config.fc_task_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def _fuse_for_expert(
        self,
        inputs: dict[str, torch.Tensor],
        expert_name: str,
    ) -> torch.Tensor:
        expert_projs_module = self.expert_projections[expert_name]
        assert isinstance(expert_projs_module, nn.ModuleDict)
        projected = []
        for name, tensor in inputs.items():
            if name not in expert_projs_module:
                continue
            flattened = tensor.flatten(start_dim=1)
            proj = expert_projs_module[name](flattened)
            projected.append(proj)

        if not projected:
            raise ValueError(
                f"No valid modalities found in inputs for {expert_name}. "
                f"Received: {list(inputs.keys())}, "
                f"Expected: {list(expert_projs_module.keys())}"
            )

        fused = torch.stack(projected, dim=0).sum(dim=0)

        num_modalities = len(projected)
        if num_modalities > 1:
            fused = fused / math.sqrt(num_modalities)

        return fused

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.use_sum_fusion:
            expert_out_list = []
            for expert_name, expert_branch in self.expert_branches.items():
                expert_input = self._fuse_for_expert(
                    inputs=inputs,
                    expert_name=expert_name,
                )
                cur_expert_output = expert_branch(expert_input)
                expert_out_list.append(cur_expert_output)

            stacked_expert_outputs = torch.stack(expert_out_list, dim=2)
        else:
            fused_features = self.fusion_callable(inputs)
            expert_outputs = calculate_module_dict_outputs(
                input_=fused_features,
                module_dict=self.expert_branches,
            )
            stacked_expert_outputs = torch.stack(list(expert_outputs.values()), dim=2)

        final_out = {}
        for group_name, logits in self.gate_logits.items():
            attention = torch.softmax(logits, dim=0)
            weighted = attention * stacked_expert_outputs
            final_out[group_name] = weighted.sum(dim=2)

        return final_out
