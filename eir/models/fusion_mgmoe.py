from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Sequence, TYPE_CHECKING

import torch
from aislib.pytorch_modules import Swish
from torch import nn

from eir.models.fusion import default_fuse_features, al_features
from eir.models.layers import MLPResidualBlock
from eir.models.models_base import (
    construct_multi_branches,
    initialize_modules_from_spec,
    create_multi_task_blocks_with_first_adaptor_block,
    construct_blocks,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)

if TYPE_CHECKING:
    from eir.train import al_num_outputs_per_target


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
    """

    layers: Sequence[int] = field(default_factory=lambda: [1, 1])
    fc_task_dim: int = 64

    mg_num_experts: int = 8

    rb_do: float = 0.00
    fc_do: float = 0.00


class MGMoEModel(nn.Module):
    def __init__(
        self,
        model_config: MGMoEModelConfig,
        num_outputs_per_target: "al_num_outputs_per_target",
        modules_to_fuse: nn.ModuleDict,
        fusion_callable: al_features = default_fuse_features,
    ):
        super().__init__()

        self.model_config = model_config
        self.num_outputs_per_target = num_outputs_per_target
        self.modules_to_fuse = modules_to_fuse
        self.fusion_callable = fusion_callable

        self.num_experts = self.model_config.mg_num_experts

        cur_dim = sum(i.num_out_features for i in self.modules_to_fuse.values())

        self.task_names = sorted(tuple(self.num_outputs_per_target.keys()))
        gate_spec = self.get_gate_spec(
            in_features=cur_dim, out_features=self.num_experts
        )

        self.gates = construct_multi_branches(
            branch_names=self.task_names,
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": gate_spec},
        )

        expert_names = tuple(f"expert_{i}" for i in range(self.num_experts))
        layer_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "full_preactivation": False,
        }
        self.expert_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=expert_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=layer_kwargs,
            first_layer_kwargs_overload={
                "full_preactivation": True,
                "in_features": cur_dim,
            },
        )

        task_resblocks_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "full_preactivation": False,
        }
        multi_task_branches = construct_multi_branches(
            branch_names=self.task_names,
            branch_factory=construct_blocks,
            branch_factory_kwargs={
                "num_blocks": self.model_config.layers[1],
                "block_constructor": MLPResidualBlock,
                "block_kwargs": task_resblocks_kwargs,
            },
        )

        final_act_spec = self.get_final_act_spec(
            in_features=self.model_config.fc_task_dim, dropout_p=self.model_config.fc_do
        )
        final_act = construct_multi_branches(
            branch_names=self.task_names,
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": final_act_spec},
        )

        final_layer = get_final_layer(
            in_features=self.model_config.fc_task_dim,
            num_outputs_per_target=self.num_outputs_per_target,
        )

        self.multi_task_branches = merge_module_dicts(
            (multi_task_branches, final_act, final_layer)
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

    @staticmethod
    def get_final_act_spec(in_features: int, dropout_p: float):
        # TODO: Refactor, duplicated from fusion.py

        spec = OrderedDict(
            {
                "bn_final": (nn.BatchNorm1d, {"num_features": in_features}),
                "act_final": (Swish, {}),
                "do_final": (nn.Dropout, {"p": dropout_p}),
            }
        )

        return spec

    def _init_weights(self):
        pass

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        # TODO: Refactor into class for fusion models.
        out = []
        for module in self.modules_to_fuse.values():
            if hasattr(module, "l1_penalized_weights"):
                weight_flat = torch.flatten(module.l1_penalized_weights)
                out.append(weight_flat)

        return torch.cat(out)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        out = {}
        for module_name, module in self.modules_to_fuse.items():
            module_input = inputs[module_name]
            out[module] = module(module_input)

        fused_features = default_fuse_features(tuple(out.values()))

        gate_attentions = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.gates
        )

        expert_outputs = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.expert_branches
        )

        final_out = {}
        stacked_expert_outputs = torch.stack(list(expert_outputs.values()), dim=2)
        for task_name, task_attention in gate_attentions.items():
            weighted_expert_outputs = (
                task_attention.unsqueeze(1) * stacked_expert_outputs
            )
            weighted_expert_sum = weighted_expert_outputs.sum(dim=2)

            cur_task_branch = self.multi_task_branches[task_name]
            final_out[task_name] = cur_task_branch(weighted_expert_sum)

        return final_out
