from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from typing import Dict, List, Callable, Sequence
from functools import partial

import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.layers import SplitLinear, MLPResidualBlock, SplitMLPResidualBlock
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
        out = flatten_h_w_fortran(x=x)

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

        fc_0_split_size = calc_value_after_expansion(
            base=self.cl_args.kernel_width,
            expansion=self.cl_args.first_kernel_expansion,
        )
        fc_0_channel_exponent = calc_value_after_expansion(
            base=self.cl_args.channel_exp_base,
            expansion=self.cl_args.first_channel_expansion,
        )
        self.fc_0 = SplitLinear(
            in_features=self.fc_1_in_features,
            out_feature_sets=2 ** fc_0_channel_exponent,
            split_size=fc_0_split_size,
            bias=False,
        )

        split_parameter_spec = SplitParameterSpec(
            in_features=self.fc_0.out_features,
            kernel_width=self.cl_args.kernel_width,
            channel_exp_base=self.cl_args.channel_exp_base,
            dropout_p=self.cl_args.rb_do,
            cutoff=1024 * 16,
        )
        self.split_blocks = _get_split_blocks(
            split_parameter_spec=split_parameter_spec,
            block_layer_spec=self.cl_args.layers,
        )

        cur_dim = self.split_blocks[-1].out_features
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
            first_layer_kwargs_overload={"in_features": cur_dim + self.extra_dim},
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
        return self.fc_0.weight

    def _init_weights(self):
        pass

    def forward(
        self, x: torch.Tensor, extra_inputs: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        out = flatten_h_w_fortran(x=x)

        out = self.fc_0(out)
        out = self.split_blocks(out)

        if extra_inputs is not None:
            out_extra = self.fc_extra(extra_inputs)
            out = torch.cat((out_extra, out), dim=1)

        out = calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        return out


def flatten_h_w_fortran(x: torch.Tensor) -> torch.Tensor:
    """
    This is needed when e.g. flattening one-hot inputs, and we want to make sure the
    first part of the flattened tensor is the first column, i.e. first one-hot element.
    """
    column_order_flattened = x.transpose(2, 3).flatten(1)
    return column_order_flattened


def calc_value_after_expansion(base: int, expansion: int, min_value: int = 0) -> int:
    if expansion > 0:
        return base * expansion
    elif expansion < 0:
        abs_expansion = abs(expansion)
        return max(min_value, base // abs_expansion)
    return base


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


@dataclass
class SplitParameterSpec:
    in_features: int
    kernel_width: int
    channel_exp_base: int
    dropout_p: float
    cutoff: int


def _get_split_blocks(
    split_parameter_spec: SplitParameterSpec,
    block_layer_spec: Sequence[int],
) -> nn.Sequential:
    factory = _get_split_block_factory(block_layer_spec=block_layer_spec)

    blocks = factory(split_parameter_spec)

    return blocks


def _get_split_block_factory(
    block_layer_spec: Sequence[int],
) -> Callable[[SplitParameterSpec], nn.Sequential]:
    if len(block_layer_spec) == 1:
        return generate_split_resblocks_auto

    auto_factory = partial(
        _generate_split_blocks_from_spec, block_layer_spec=block_layer_spec[:-1]
    )

    return auto_factory


def _generate_split_blocks_from_spec(
    split_parameter_spec: SplitParameterSpec,
    block_layer_spec: List[int],
) -> nn.Sequential:

    s = split_parameter_spec
    block_layer_spec_copy = copy(block_layer_spec)

    first_block = SplitMLPResidualBlock(
        in_features=s.in_features,
        split_size=s.kernel_width,
        out_feature_sets=2 ** s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=True,
    )

    block_modules = [first_block]
    block_layer_spec_copy[0] -= 1

    for cur_layer_index, block_dim in enumerate(block_layer_spec_copy):
        for block in range(block_dim):

            cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_layer_index)
            cur_kernel_width = s.kernel_width
            while cur_out_feature_sets >= cur_kernel_width:
                cur_kernel_width *= 2

            cur_size = block_modules[-1].out_features

            cur_block = SplitMLPResidualBlock(
                in_features=cur_size,
                split_size=cur_kernel_width,
                out_feature_sets=cur_out_feature_sets,
                dropout_p=s.dropout_p,
            )

            block_modules.append(cur_block)

    return nn.Sequential(*block_modules)


def generate_split_resblocks_auto(split_parameter_spec: SplitParameterSpec):
    """
    TODO:   Create some over-engineered abstraction for this and
            `_generate_split_blocks_from_spec` if feeling bored.
    """

    s = split_parameter_spec

    first_block = SplitMLPResidualBlock(
        in_features=s.in_features,
        split_size=s.kernel_width,
        out_feature_sets=2 ** s.channel_exp_base,
        dropout_p=s.dropout_p,
        full_preactivation=True,
    )

    block_modules = [first_block]

    while True:
        cur_no_blocks = len(block_modules)
        cur_index = cur_no_blocks // 2

        cur_out_feature_sets = 2 ** (s.channel_exp_base + cur_index)
        cur_kernel_width = s.kernel_width
        while cur_out_feature_sets >= cur_kernel_width:
            cur_kernel_width *= 2

        cur_size = block_modules[-1].out_features
        if cur_size <= s.cutoff:
            break

        cur_block = SplitMLPResidualBlock(
            in_features=cur_size,
            split_size=cur_kernel_width,
            out_feature_sets=cur_out_feature_sets,
            dropout_p=s.dropout_p,
        )

        block_modules.append(cur_block)

    logger.info(
        "No SplitLinear residual blocks specified in CL arguments. Created %d "
        "blocks with final output dimension of %d.",
        len(block_modules),
        cur_size,
    )
    return nn.Sequential(*block_modules)