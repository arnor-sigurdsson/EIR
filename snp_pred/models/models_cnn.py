from argparse import Namespace
from collections import OrderedDict
from typing import List, Dict, Tuple, TYPE_CHECKING

import torch
from aislib import pytorch_utils
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from snp_pred.models.layers import FirstCNNBlock, SelfAttention, CNNResidualBlock
from snp_pred.models.models_base import (
    ModelBase,
    calculate_module_dict_outputs,
    assert_module_dict_uniqueness,
)

if TYPE_CHECKING:
    from snp_pred.train import al_num_outputs_per_target


logger = get_logger(__name__)


class CNNModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(*_make_conv_layers(self.resblocks, self.cl_args))

        self.data_size_after_conv = pytorch_utils.calc_size_after_conv_sequence(
            self.cl_args.target_width, self.conv
        )
        self.no_out_channels = self.conv[-1].out_channels

        self.fc_1 = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(self.fc_1_in_features),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(
                        self.fc_1_in_features, self.cl_args.fc_repr_dim, bias=False
                    ),
                }
            )
        )

        self.multi_task_branches = _get_cnn_multi_task_branches(
            num_outputs_per_target=self.num_outputs_per_target,
            fc_task_dim=self.fc_task_dim,
            fc_repr_and_extra_dim=self.fc_repr_and_extra_dim,
            fc_do=self.cl_args.fc_do,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.no_out_channels * self.data_size_after_conv

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.conv[0].conv_1.weight

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Swish slope is roughly 0.5 around 0
                nn.init.kaiming_normal_(m.weight, a=0.5, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def resblocks(self) -> List[int]:
        if not self.cl_args.layers:
            residual_blocks = find_no_cnn_resblocks_needed(
                self.cl_args.target_width,
                self.cl_args.down_stride,
                self.cl_args.first_stride_expansion,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.cl_args.layers

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        genotype = inputs["genotype"]

        out = self.conv(genotype)
        out = out.view(out.shape[0], -1)

        out = self.fc_1(out)

        tabular = inputs.get("tabular")
        if tabular is not None:
            out_tabular = self.fc_extra(tabular)
            out = torch.cat((out_tabular, out), dim=1)

        out = calculate_module_dict_outputs(
            input_=out, module_dict=self.multi_task_branches
        )

        return out


def _get_cnn_multi_task_branches(
    fc_repr_and_extra_dim: int,
    fc_task_dim: int,
    fc_do: float,
    num_outputs_per_target: "al_num_outputs_per_target",
) -> nn.ModuleDict:
    """
    TODO: Remove this in favor of branch factories as used in other modesl
    """

    module_dict = {}
    for key, num_outputs_per_target in num_outputs_per_target.items():
        branch_layers = OrderedDict(
            {
                "fc_2_bn_1": nn.BatchNorm1d(fc_repr_and_extra_dim),
                "fc_2_act_1": Swish(),
                "fc_2_linear_1": nn.Linear(
                    fc_repr_and_extra_dim, fc_task_dim, bias=False
                ),
                "fc_2_do_1": nn.Dropout(p=fc_do),
                "fc_3_bn_1": nn.BatchNorm1d(fc_task_dim),
                "fc_3_act_1": Swish(),
                "fc_3_do_1": nn.Dropout(p=fc_do),
            }
        )

        task_layer_branch = nn.Sequential(
            OrderedDict(
                **branch_layers,
                **{"fc_3_final": nn.Linear(fc_task_dim, num_outputs_per_target)}
            )
        )

        module_dict[key] = task_layer_branch

    assert_module_dict_uniqueness(module_dict)
    return nn.ModuleDict(module_dict)


def _make_conv_layers(
    residual_blocks: List[int], cl_args: Namespace
) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param cl_args: Experiment hyperparameters / configuration needed for the
    convolution setup.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    ca = cl_args

    down_stride_w = ca.down_stride

    first_conv_channels = 2 ** ca.channel_exp_base * ca.first_channel_expansion
    first_conv_kernel = ca.kernel_width * ca.first_kernel_expansion
    first_conv_stride = down_stride_w * ca.first_stride_expansion

    first_kernel, first_pad = pytorch_utils.calc_conv_params_needed(
        input_width=ca.target_width,
        kernel_size=first_conv_kernel,
        stride=first_conv_stride,
        dilation=1,
    )

    conv_blocks = [
        FirstCNNBlock(
            in_channels=1,
            out_channels=first_conv_channels,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=first_conv_stride,
            dilation=1,
            rb_do=ca.rb_do,
        )
    ]

    sa_added = False
    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_layer, cur_width = _get_conv_resblock(
                conv_blocks=conv_blocks,
                layer_arch_idx=layer_arch_idx,
                down_stride=down_stride_w,
                cl_args=ca,
            )

            if cl_args.sa and cur_width < 1024 and not sa_added:
                attention_channels = conv_blocks[-1].out_channels
                conv_blocks.append(SelfAttention(attention_channels))
                sa_added = True

            conv_blocks.append(cur_layer)

    return conv_blocks


def _get_conv_resblock(
    conv_blocks: List[nn.Module],
    layer_arch_idx: int,
    down_stride: int,
    cl_args: Namespace,
) -> Tuple[CNNResidualBlock, int]:
    ca = cl_args

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width = pytorch_utils.calc_size_after_conv_sequence(
        input_width=ca.target_width, conv_sequence=cur_conv
    )

    cur_kern, cur_padd = pytorch_utils.calc_conv_params_needed(
        input_width=cur_width,
        kernel_size=ca.kernel_width,
        stride=down_stride,
        dilation=1,
    )

    cur_block_number = (
        len([i for i in conv_blocks if isinstance(i, CNNResidualBlock)]) + 1
    )
    cur_dilation_factor = _get_cur_dilation(
        dilation_factor=cl_args.dilation_factor,
        width=cur_width,
        block_number=cur_block_number,
    )

    cur_in_channels = conv_blocks[-1].out_channels
    cur_out_channels = 2 ** (ca.channel_exp_base + layer_arch_idx)

    cur_layer = CNNResidualBlock(
        in_channels=cur_in_channels,
        out_channels=cur_out_channels,
        conv_1_kernel_w=cur_kern,
        conv_1_padding=cur_padd,
        down_stride_w=down_stride,
        dilation=cur_dilation_factor,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=ca.rb_do,
    )

    return cur_layer, cur_width


def _get_cur_dilation(dilation_factor: int, width: int, block_number: int):
    """
    Note that block_number refers to the number of residual blocks (not first block
    or self attention).
    """
    dilation = dilation_factor ** block_number

    while dilation >= width:
        dilation = dilation // dilation_factor

    return dilation


def find_no_cnn_resblocks_needed(
    width: int, stride: int, first_stride_expansion: int
) -> List[int]:
    """
    Used in order to calculate / set up residual blocks specifications as a list
    automatically when they are not passed in as CL args, based on the minimum
    width after the resblock convolutions.

    We have 2 resblocks per channel depth until we have a total of 8 blocks,
    then the rest is put in the third depth index (following resnet convention).

    That is with a base channel depth of 32, we have these depths in the list:
    [32, 64, 128, 256].

    Examples
    ------
    3 blocks --> [2, 1]
    7 blocks --> [2, 2, 2, 1]
    10 blocks --> [2, 2, 4, 2]
    """

    min_size = 8 * stride
    # account for first conv
    cur_width = width // (stride * first_stride_expansion)

    resblocks = [0] * 4
    while cur_width >= min_size:
        cur_no_blocks = sum(resblocks)

        if cur_no_blocks >= 8:
            resblocks[2] += 1
        else:
            cur_index = cur_no_blocks // 2
            resblocks[cur_index] += 1

        cur_width = cur_width // stride

    return [i for i in resblocks if i != 0]
