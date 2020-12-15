from collections import OrderedDict
from typing import Dict

import torch
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.fusion import default_fuse_features, al_features
from snp_pred.models.layers import MLPResidualBlock
from snp_pred.models.models_base import (
    ModelBase,
    construct_multi_branches,
    initialize_modules_from_spec,
    create_multi_task_blocks_with_first_adaptor_block,
    construct_blocks,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)


# TODO: Deprecate ModelBase
class MGMoEModel(ModelBase):
    def __init__(
        self,
        modules_to_fuse: nn.ModuleDict,
        fusion_callable: al_features = default_fuse_features,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.modules_to_fuse = modules_to_fuse
        self.fusion_callable = fusion_callable

        self.num_chunks = self.cl_args.split_mlp_num_splits
        self.num_experts = self.cl_args.mg_num_experts

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
            "in_features": self.fc_task_dim,
            "out_features": self.fc_task_dim,
            "dropout_p": self.cl_args.rb_do,
            "full_preactivation": False,
        }
        self.expert_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.cl_args.layers[0],
            branch_names=expert_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=layer_kwargs,
            first_layer_kwargs_overload={
                "full_preactivation": True,
                "in_features": cur_dim,
            },
        )

        task_resblocks_kwargs = {
            "in_features": self.fc_task_dim,
            "out_features": self.fc_task_dim,
            "dropout_p": self.cl_args.rb_do,
            "full_preactivation": False,
        }
        multi_task_branches = construct_multi_branches(
            branch_names=self.task_names,
            branch_factory=construct_blocks,
            branch_factory_kwargs={
                "num_blocks": self.cl_args.layers[1],
                "block_constructor": MLPResidualBlock,
                "block_kwargs": task_resblocks_kwargs,
            },
        )

        final_act_spec = self.get_final_act_spec(
            in_features=self.fc_task_dim, dropout_p=self.cl_args.fc_do
        )
        final_act = construct_multi_branches(
            branch_names=self.task_names,
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": final_act_spec},
        )

        final_layer = get_final_layer(
            in_features=self.fc_task_dim,
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
        # TODO: Refactor, duplicated form fusion.py

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
        # TODO: Refactor into base class for fusion models
        out = []
        for module in self.modules_to_fuse.values():
            if hasattr(module, "l1_penalized_weights"):
                out.append(module.l1_penalized_weights)

        return torch.stack(out)

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
