import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from eir.models.fusion.fusion_default import al_features, default_fuse_features
from eir.models.layers.mlp_layers import MLPResidualBlock
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    construct_multi_branches,
    create_multi_task_blocks_with_first_adaptor_block,
    initialize_modules_from_spec,
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
            self.input_projections = nn.ModuleDict()
            for name, info in feature_dimensions_and_types.items():
                output_dim = info.output_dimension
                self.input_projections[name] = nn.Sequential(
                    nn.RMSNorm(normalized_shape=output_dim),
                    nn.Linear(
                        in_features=output_dim,
                        out_features=self.model_config.fc_task_dim,
                    ),
                    nn.GELU(),
                )
            expert_in_dim = self.model_config.fc_task_dim
        else:
            expert_in_dim = fusion_in_dim

        gate_spec = self.get_gate_spec(
            in_features=expert_in_dim, out_features=self.num_experts
        )

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

        self.gates = construct_multi_branches(
            branch_names=tuple(output_group_names),
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": gate_spec},
        )

        self._init_weights()

    @staticmethod
    def get_gate_spec(in_features: int, out_features: int):
        spec = OrderedDict(
            {
                "gate_fc": (
                    nn.Linear,
                    {
                        "in_features": in_features,
                        "out_features": out_features,
                        "bias": True,
                    },
                ),
                "gate_attention": (nn.Softmax, {"dim": 1}),
            }
        )

        return spec

    def _init_weights(self):
        pass

    @property
    def per_output_group(self) -> bool:
        return True

    @property
    def num_out_features(self) -> int:
        return self.model_config.fc_task_dim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.num_out_features,)

    def _fuse_inputs(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_sum_fusion:
            return self.fusion_callable(inputs)

        projected = []
        for name, tensor in inputs.items():
            if name not in self.input_projections:
                continue
            flattened = tensor.flatten(start_dim=1)
            proj = self.input_projections[name](flattened)
            projected.append(proj)

        if not projected:
            raise ValueError(
                f"No valid modalities found in inputs. "
                f"Received: {list(inputs.keys())}, "
                f"Expected: {list(self.input_projections.keys())}"
            )

        fused = torch.stack(projected, dim=0).sum(dim=0)

        num_modalities = len(projected)
        if num_modalities > 1:
            fused = fused / math.sqrt(num_modalities)

        return fused

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused_features = self._fuse_inputs(inputs=inputs)

        expert_outputs = calculate_module_dict_outputs(
            input_=fused_features,
            module_dict=self.expert_branches,
        )
        stacked_expert_outputs = torch.stack(list(expert_outputs.values()), dim=2)

        gate_attentions = calculate_module_dict_outputs(
            input_=fused_features,
            module_dict=self.gates,
        )

        final_out = {}
        for group_name, attention in gate_attentions.items():
            weighted = attention.unsqueeze(1) * stacked_expert_outputs
            final_out[group_name] = weighted.sum(dim=2)

        return final_out
