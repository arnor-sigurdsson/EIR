from collections import OrderedDict
from typing import Dict

import torch
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.layers import SplitLinear, MLPResidualBlock
from snp_pred.models.models_base import (
    ModelBase,
    construct_multi_branches,
    initialize_modules_from_spec,
    create_blocks_with_first_adaptor_block,
    construct_blocks,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)


class MGMoEModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_chunks = self.cl_args.split_mlp_num_splits
        self.num_experts = self.cl_args.mg_num_experts

        fc_0_out_feat = self.num_chunks * self.cl_args.fc_repr_dim
        self.fc_0 = nn.Sequential(
            OrderedDict(
                {
                    "fc_0": SplitLinear(
                        in_features=self.fc_1_in_features,
                        out_feature_sets=self.cl_args.fc_repr_dim,
                        num_chunks=self.num_chunks,
                        bias=True,
                    )
                }
            )
        )

        self.task_names = sorted(tuple(self.num_classes.keys()))
        gate_spec = self.get_gate_spec(
            in_features=fc_0_out_feat + self.extra_dim, out_features=self.num_experts
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
        self.expert_branches = create_blocks_with_first_adaptor_block(
            num_blocks=self.cl_args.layers[0],
            branch_names=expert_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=layer_kwargs,
            first_layer_kwargs_overload={
                "full_preactivation": True,
                "in_features": fc_0_out_feat + self.extra_dim,
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
            in_features=self.fc_task_dim, num_classes=self.num_classes
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
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0[0].weight

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        if extra_inputs is not None:
            out = torch.cat((extra_inputs, out), dim=1)

        gate_attentions = calculate_module_dict_outputs(
            input_=out, module_dict=self.gates
        )

        expert_outputs = calculate_module_dict_outputs(
            input_=out, module_dict=self.expert_branches
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
