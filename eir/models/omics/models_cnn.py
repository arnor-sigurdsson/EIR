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
    from eir.setup.input_setup import DataDimensions

logger = get_logger(__name__)


@dataclass
class CNNModelConfig:
    """
    Note that when using the automatic network setup, channels will get increased as
    the input gets propagated through the network while the width gets reduced due to
    stride.

    :param layers:
        Controls the number of layers in the model. If set to ``None``, the model will
        automatically set up the number of layers until a certain width (stride * 8)
        is met. Future work includes adding a parameter to control the target width.

    :param fc_repr_dim:
        Output dimension of the last FC layer in the network which accepts the outputs
        from the convolutional layer.

    :param down_stride:
        Down stride of the convolutional layers.

    :param first_stride_expansion:
        Factor to extend the first layer stride. This value can both be positive or
        negative. For example in the case of ``down_stride=12``, setting
        ``first_stride_expansion=2`` means that the first layer will have a stride of
        24, whereas other layers will have a stride of 12. When using a negative value,
        divides the first stride by the value instead of multiplying.

    :param channel_exp_base:
        Which power of 2 to use in order to set the number of channels in the network.
        For example, setting ``channel_exp_base=3`` means that 2**3=8 channels will be
        used.

    :param first_channel_expansion:
        Factor to extend the first layer channels. This value can both be positive or
        negative. For example in the case of ``channel_exp_base=3`` (i.e. 8 channels),
        setting ``first_channel_expansion=2`` means that the first layer will have 16
        channels, whereas other layers will have a channel of 8 as base.
        When using a negative value, divides the first channel by the value instead
        of multiplying.

    :param kernel_width:
        Base kernel width of the convolutions. Differently from the LCL model
        configurations, this number refers to the actual columns in the unflattened
        input. So assuming an omics input, setting kernel_width=2 means 2 SNPs covered
        at a time.

    :param first_kernel_expansion:
        Factor to extend the first kernel. This value can both be positive or negative.
        For example in the case of ``kernel_width=12``, setting
        ``first_kernel_expansion=2`` means that the first kernel will have a width of
        24, whereas other kernels will have a width of 12. When using a negative value,
        divides the first kernel by the value instead of multiplying.

    :param dilation_factor:
        Base dilation factor of the convolutions in the network.

    :param rb_do:
        Dropout in the convolutional residual blocks.

    :param sa:
        Whether to add a self-attention layer to the network after a width of 1024
        has been reached.

    :param l1:
        L1 regularization to apply to the first layer.
    """

    layers: Union[None, List[int]] = None

    fc_repr_dim: int = 32

    down_stride: int = 4
    first_stride_expansion: int = 1

    channel_exp_base: int = 2
    first_channel_expansion: int = 1

    kernel_width: int = 12
    first_kernel_expansion: int = 1
    dilation_factor: int = 1

    rb_do: float = 0.00

    sa: bool = False
    l1: float = 0.00


class CNNModel(nn.Module):
    def __init__(self, model_config: CNNModelConfig, data_dimensions: "DataDimensions"):
        # TODO: Make work for heights, this means modifying stuff in layers.py
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions
        self.conv = nn.Sequential(
            *_make_conv_layers(
                residual_blocks=self.resblocks,
                cnn_model_configuration=self.model_config,
                data_dimensions=self.data_dimensions,
            )
        )

        self.data_size_after_conv = pytorch_utils.calc_size_after_conv_sequence(
            self.data_dimensions.width, self.conv
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
                self.data_dimensions.width,
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
    residual_blocks: List[int],
    cnn_model_configuration: CNNModelConfig,
    data_dimensions: "DataDimensions",
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
    :return: A list of ``nn.Module`` objects to be passed to ``nn.Sequential``.
    """
    mc = cnn_model_configuration

    down_stride_w = mc.down_stride

    first_conv_channels = 2 ** mc.channel_exp_base * mc.first_channel_expansion
    first_conv_kernel = mc.kernel_width * mc.first_kernel_expansion
    first_conv_stride = down_stride_w * mc.first_stride_expansion

    first_kernel, first_pad = pytorch_utils.calc_conv_params_needed(
        input_width=data_dimensions.width,
        kernel_size=first_conv_kernel,
        stride=first_conv_stride,
        dilation=1,
    )

    conv_blocks = [
        FirstCNNBlock(
            in_channels=data_dimensions.channels,
            out_channels=first_conv_channels,
            conv_1_kernel_h=data_dimensions.height,
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
                data_dimensions=data_dimensions,
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
    data_dimensions: "DataDimensions",
) -> Tuple[CNNResidualBlock, int]:
    mc = cnn_config

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width = pytorch_utils.calc_size_after_conv_sequence(
        input_width=data_dimensions.width, conv_sequence=cur_conv
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
