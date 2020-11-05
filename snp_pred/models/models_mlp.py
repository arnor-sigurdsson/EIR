from collections import OrderedDict
from typing import Dict

import torch
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.models_base import (
    ModelBase,
    get_basic_multi_branch_spec,
    create_multi_task_blocks_with_first_adaptor_block,
    initialize_modules_from_spec,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)


class MLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_0 = nn.Linear(
            self.fc_1_in_features, self.cl_args.fc_repr_dim, bias=False
        )

        self.fc_0_act = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(self.cl_args.fc_repr_dim),
                    "fc_1_act_1": Swish(),
                    "fc_1_do_1": nn.Dropout(p=self.cl_args.fc_do),
                }
            )
        )

        first_layer_spec = get_basic_multi_branch_spec(
            in_features=self.fc_repr_and_extra_dim,
            out_features=self.fc_task_dim,
            dropout_p=self.cl_args.fc_do,
        )
        layer_spec = get_basic_multi_branch_spec(
            in_features=self.fc_task_dim,
            out_features=self.fc_task_dim,
            dropout_p=self.cl_args.fc_do,
        )

        branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.cl_args.layers[0],
            branch_names=self.target_class_mapping.keys(),
            block_constructor=initialize_modules_from_spec,
            block_constructor_kwargs={"spec": layer_spec},
            first_layer_kwargs_overload={"spec": first_layer_spec},
        )

        final_layer = get_final_layer(
            in_features=self.fc_task_dim, num_classes=self.target_class_mapping
        )

        self.multi_task_branches = merge_module_dicts((branches, final_layer))

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    def _init_weights(self):
        pass

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)

        out = self.fc_0_act(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        return out
