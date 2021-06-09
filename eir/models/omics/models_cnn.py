from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING, Union

import torch
from aislib import pytorch_utils
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from eir.models.layers import FirstCNNBlock, SelfAttention, CNNResidualBlock

if TYPE_CHECKING:
    from eir.train import DataDimensions

logger = get_logger(__name__)


@dataclass
class CNNModelConfig:

    layers: Union[None, List[int]]

    fc_repr_dim: int

    down_stride: int
    first_stride_expansion: int

    channel_exp_base: int
    first_channel_expansion: int

    kernel_width: int
    first_kernel_expansion: int
    dilation_factor: int

    data_dimensions: "DataDimensions"

    rb_do: float

    sa: bool = False


class CNNModel(nn.Module):
    def __init__(self, model_config: CNNModelConfig):
        # TODO: Make work for heights, this means modifying stuff in layers.py
        super().__init__()

        self.model_config = model_config
        self.conv = nn.Sequential(*_make_conv_layers(self.resblocks, self.model_config))

        self.data_size_after_conv = pytorch_utils.calc_size_after_conv_sequence(
            self.model_config.data_dimensions.width, self.conv
        )
        self.no_out_channels = self.conv[-1].out_channels

        self.fc = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_bn_1": nn.BatchNorm1d(self.fc_1_in_features),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(
                        self.fc_1_in_features, self.model_config.fc_repr_dim, bias=False
                    ),
                }
            )
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.no_out_channels * self.data_size_after_conv

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.conv[0].conv_1.weight

    @property
    def num_out_features(self) -> int:
        return self.model_config.fc_repr_dim

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
        if not self.model_config.layers:
            residual_blocks = find_no_cnn_resblocks_needed(
                self.model_config.data_dimensions.width,
                self.model_config.down_stride,
                self.model_config.first_stride_expansion,
            )
            logger.info(
                "No residual blocks specified in CL args, using input "
                "%s based on width approximation calculation.",
                residual_blocks,
            )
            return residual_blocks
        return self.model_config.layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        out = self.conv(input)
        out = out.view(out.shape[0], -1)

        out = self.fc(out)

        return out


def _make_conv_layers(
    residual_blocks: List[int], cnn_model_configuration: CNNModelConfig
) -> List[nn.Module]:
    """
    Used to set up the convolutional layers for the model. Based on the passed in
    residual blocks, we want to set up the actual blocks with all the relevant
    convolution parameters.

    We start with a base channel number of 2**5 == 32.

    Also inserts a self-attention layer in just before the last residual block.

    :param residual_blocks: List of ints, where each int indicates number of blocks w.
    that channel dimension.
    :param cnn_model_configuration: Experiment hyperparameters / configuration needed
    for the convolution setup.
    :return: A list of `nn.Module` objects to be passed to `nn.Sequential`.
    """
    mc = cnn_model_configuration

    down_stride_w = mc.down_stride

    first_conv_channels = 2 ** mc.channel_exp_base * mc.first_channel_expansion
    first_conv_kernel = mc.kernel_width * mc.first_kernel_expansion
    first_conv_stride = down_stride_w * mc.first_stride_expansion

    first_kernel, first_pad = pytorch_utils.calc_conv_params_needed(
        input_width=mc.data_dimensions.width,
        kernel_size=first_conv_kernel,
        stride=first_conv_stride,
        dilation=1,
    )

    conv_blocks = [
        FirstCNNBlock(
            in_channels=cnn_model_configuration.data_dimensions.channels,
            out_channels=first_conv_channels,
            conv_1_kernel_h=cnn_model_configuration.data_dimensions.height,
            conv_1_kernel_w=first_kernel,
            conv_1_padding=first_pad,
            down_stride_w=first_conv_stride,
            dilation=1,
            rb_do=mc.rb_do,
        )
    ]

    sa_added = False
    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_layer, cur_width = _get_conv_resblock(
                conv_blocks=conv_blocks,
                layer_arch_idx=layer_arch_idx,
                down_stride=down_stride_w,
                cnn_config=cnn_model_configuration,
            )

            if mc.sa and cur_width < 1024 and not sa_added:
                attention_channels = conv_blocks[-1].out_channels
                conv_blocks.append(SelfAttention(attention_channels))
                sa_added = True

            conv_blocks.append(cur_layer)

    return conv_blocks


def _get_conv_resblock(
    conv_blocks: List[nn.Module],
    layer_arch_idx: int,
    down_stride: int,
    cnn_config: CNNModelConfig,
) -> Tuple[CNNResidualBlock, int]:
    mc = cnn_config

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width = pytorch_utils.calc_size_after_conv_sequence(
        input_width=mc.data_dimensions.width, conv_sequence=cur_conv
    )

    cur_kern, cur_padd = pytorch_utils.calc_conv_params_needed(
        input_width=cur_width,
        kernel_size=mc.kernel_width,
        stride=down_stride,
        dilation=1,
    )

    cur_block_number = (
        len([i for i in conv_blocks if isinstance(i, CNNResidualBlock)]) + 1
    )
    cur_dilation_factor = _get_cur_dilation(
        dilation_factor=mc.dilation_factor,
        width=cur_width,
        block_number=cur_block_number,
    )

    cur_in_channels = conv_blocks[-1].out_channels
    cur_out_channels = 2 ** (mc.channel_exp_base + layer_arch_idx)

    cur_layer = CNNResidualBlock(
        in_channels=cur_in_channels,
        out_channels=cur_out_channels,
        conv_1_kernel_h=1,
        conv_1_kernel_w=cur_kern,
        conv_1_padding=cur_padd,
        down_stride_w=down_stride,
        dilation=cur_dilation_factor,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=mc.rb_do,
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
