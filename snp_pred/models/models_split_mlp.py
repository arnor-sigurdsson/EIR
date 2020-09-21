from collections import OrderedDict
from copy import copy
from typing import Dict, List

import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.layers import SplitLinear, MLPResidualBlock
from snp_pred.models.models_base import (
    ModelBase,
    create_multi_task_blocks_with_first_adaptor_block,
    construct_multi_branches,
    initialize_modules_from_spec,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)

logger = get_logger(__name__)


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
        multi_task_branches = create_multi_task_blocks_with_first_adaptor_block(
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


class FullySplitMLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fc_0_split_size = calc_fc_0_split_size_after_expansion(
            kernel_width=self.cl_args.kernel_width,
            expansion=self.cl_args.first_kernel_expansion,
        )
        self.fc_0 = SplitLinear(
            in_features=self.fc_1_in_features,
            out_feature_sets=2 ** self.cl_args.channel_exp_base,
            split_size=fc_0_split_size,
            bias=False,
        )

        blocks_spec = self.get_block_spec(in_features=self.fc_0.out_features)
        self.split_blocks = _generate_split_blocks(
            block_layer_spec=blocks_spec,
            in_features=self.fc_0.out_features,
            kernel_width=self.cl_args.kernel_width,
            channel_exp_base=self.cl_args.channel_exp_base,
            dropout_p=self.cl_args.rb_do,
        )

        cur_dim = self.split_blocks[-1][-1].out_features
        task_names = tuple(self.num_classes.keys())
        task_resblocks_kwargs = {
            "in_features": self.fc_task_dim,
            "out_features": self.fc_task_dim,
            "dropout_p": self.cl_args.rb_do,
            "full_preactivation": False,
        }
        multi_task_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.cl_args.layers[-1],
            branch_names=task_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=task_resblocks_kwargs,
            first_layer_kwargs_overload={
                "in_features": cur_dim + self.extra_dim,
                "full_preactivation": True,
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

    def get_block_spec(self, in_features: int) -> List[int]:
        if len(self.cl_args.layers) == 1:
            residual_blocks = _find_no_split_mlp_blocks_needed(
                in_features=in_features,
                kernel_width=self.cl_args.kernel_width,
                channel_exp_base=self.cl_args.channel_exp_base,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.cl_args.layers

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
        return self.fc_0.weight

    def _init_weights(self):
        pass

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = x.view(x.shape[0], -1)

        out = self.fc_0(out)
        out = self.split_blocks(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        return out


def calc_fc_0_split_size_after_expansion(kernel_width: int, expansion: int) -> int:
    if expansion > 0:
        return kernel_width * expansion
    elif expansion < 0:
        return abs(kernel_width // expansion)
    return kernel_width


def get_split_extractor_spec(
    in_features: int, out_feature_sets: int, split_size: int, dropout_p: float
):
    spec = OrderedDict(
        {
            "bn_1": (nn.BatchNorm1d, {"num_features": in_features}),
            "act_1": (Swish, {}),
            "do_1": (nn.Dropout, {"p": dropout_p}),
            "split_1": (
                SplitLinear,
                {
                    "in_features": in_features,
                    "out_feature_sets": out_feature_sets,
                    "split_size": split_size,
                    "bias": False,
                },
            ),
        }
    )
    return spec


def _generate_split_blocks(
    block_layer_spec: List[int],
    in_features: int,
    kernel_width: int,
    channel_exp_base: int,
    dropout_p: float,
) -> nn.Sequential:

    block_layer_spec_copy = copy(block_layer_spec)

    first_spec = get_split_extractor_spec(
        in_features=in_features,
        split_size=kernel_width,
        out_feature_sets=2 ** channel_exp_base,
        dropout_p=dropout_p,
    )
    first_block = initialize_modules_from_spec(spec=first_spec)

    block_modules = [first_block]

    for cur_layer_index, block_dim in enumerate(block_layer_spec_copy):
        for block in range(block_dim):

            cur_out_feature_sets = 2 ** (channel_exp_base + cur_layer_index)
            cur_kernel_width = kernel_width
            while cur_out_feature_sets >= cur_kernel_width:
                cur_kernel_width *= 2

            cur_size = block_modules[-1][-1].out_features

            cur_spec = get_split_extractor_spec(
                in_features=cur_size,
                split_size=cur_kernel_width,
                out_feature_sets=cur_out_feature_sets,
                dropout_p=dropout_p,
            )
            cur_block = initialize_modules_from_spec(spec=cur_spec)

            block_modules.append(cur_block)

    return nn.Sequential(*block_modules)


def _find_no_split_mlp_blocks_needed(
    in_features: int, kernel_width: int, channel_exp_base: int, cutoff: int = 16384
):
    """
    Need to add in same rule here for expanding kernel width.
    """

    blocks = [0] * 4

    # account for first layer
    cur_size = _calc_out_features_after_split_layer(
        in_features=in_features,
        split_size=kernel_width,
        out_feature_sets=(2 ** channel_exp_base),
    )

    while cur_size >= cutoff:
        cur_no_blocks = sum(blocks)
        cur_index = cur_no_blocks // 2

        if cur_no_blocks >= 8:
            cur_index = 2

        blocks[cur_index] += 1

        cur_exponent = len([i for i in blocks if i != 0])
        cur_out_feature_sets = 2 ** (channel_exp_base + cur_exponent)

        cur_kernel_width = kernel_width
        while cur_out_feature_sets >= cur_kernel_width:
            cur_kernel_width *= 2

        cur_size = _calc_out_features_after_split_layer(
            in_features=cur_size,
            split_size=cur_kernel_width,
            out_feature_sets=cur_out_feature_sets,
        )

    return blocks


def _calc_out_features_after_split_layer(
    in_features: int, split_size: int, out_feature_sets: int
) -> int:
    return (in_features // split_size) * out_feature_sets
