from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Sequence

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
    pass


@dataclass
class MGMoEModelConfig:
    """
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
        fusion_callable: al_features = default_fuse_features,
        **kwargs,
    ):
        super().__init__()

        self.model_config = model_config
        self.fusion_in_dim = fusion_in_dim
        self.fusion_callable = fusion_callable

        self.num_experts = self.model_config.mg_num_experts

        gate_spec = self.get_gate_spec(
            in_features=self.fusion_in_dim, out_features=self.num_experts
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
                "in_features": fusion_in_dim,
            },
        )

        self.gates = construct_multi_branches(
            branch_names=expert_names,
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
    def num_out_features(self) -> int:
        return self.model_config.fc_task_dim * len(self.expert_branches)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = self.fusion_callable(inputs)

        gate_attentions = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.gates
        )

        expert_outputs = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.expert_branches
        )

        final_out = {}
        stacked_expert_outputs = torch.stack(list(expert_outputs.values()), dim=2)
        for expert_name, attention in gate_attentions.items():
            weighted_expert_outputs = attention.unsqueeze(1) * stacked_expert_outputs
            weighted_expert_sum = weighted_expert_outputs.sum(dim=2)

            final_out[expert_name] = weighted_expert_sum

        return self.fusion_callable(final_out)
