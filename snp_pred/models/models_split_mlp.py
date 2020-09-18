from collections import OrderedDict
from typing import Dict

import torch
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.layers import SplitLinear, MLPResidualBlock
from snp_pred.models.models_base import (
    ModelBase,
    create_blocks_with_first_adaptor_block,
    construct_multi_branches,
    initialize_modules_from_spec,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)


class SplitMLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_chunks = self.cl_args.split_mlp_num_splits
        self.fc_0 = nn.Sequential(
            OrderedDict(
                {
                    "fc_0": SplitLinear(
                        in_features=self.fc_1_in_features,
                        out_feature_sets=self.cl_args.fc_repr_dim,
                        num_chunks=num_chunks,
                        bias=False,
                    )
                }
            )
        )

        in_feat = num_chunks * self.cl_args.fc_repr_dim

        task_names = tuple(self.num_classes.keys())
        task_resblocks_kwargs = {
            "in_features": self.fc_task_dim,
            "out_features": self.fc_task_dim,
            "dropout_p": self.cl_args.rb_do,
            "full_preactivation": False,
        }
        multi_task_branches = create_blocks_with_first_adaptor_block(
            num_blocks=self.cl_args.layers[0],
            branch_names=task_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=task_resblocks_kwargs,
            first_layer_kwargs_overload={
                "full_preactivation": True,
                "in_features": in_feat + self.extra_dim,
            },
        )

        final_act_spec = self.get_final_act_spec(
            in_features=self.fc_task_dim, dropout_p=self.cl_args.fc_do
        )
        final_act = construct_multi_branches(
            branch_names=task_names,
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
    def get_final_act_spec(in_features: int, dropout_p: float):

        spec = OrderedDict(
            {
                "bn_final": (nn.BatchNorm1d, {"num_features": in_features}),
                "act_final": (Swish, {}),
                "do_final": (nn.Dropout, {"p": dropout_p}),
            }
        )

        return spec

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0[0].weight

    def _init_weights(self):
        pass

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        return out
