from collections import OrderedDict
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Union

import torch
from aislib import pytorch_utils
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from eir.models.layers import FirstCNNBlock, CNNResidualBlock, ConvAttentionBlock

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

logger = get_logger(__name__)


@dataclass
class CNNModelConfig:
    """

    :param layers:
        A list that controls the number of layers and channels in the model.
        Each element in  the list represents a layer group with a
        specified number of layers and channels.
        Specifically, the first element in the list refers
        to the number of layers with the  number of channels
        exactly as specified by the `channel_exp_base` parameter.

        The subsequent elements in the list correspond
        to an increased number of channels, doubling with each step.
        For instance, if `channel_exp_base=3` (i.e., 2**3=8 channels),
        and the `layers` list is [5, 3, 2], the model would be constructed as follows:
        - First case: 5 layers with 8 channels
        - Second case: 3 layers with 16 channels (doubling from the previous case)
        - Third case: 2 layers with 32 channels (doubling from the previous case)

        The model currently supports a maximum of 4 elements in the list.

        If set to `None`, the model will automatically set up
        the number of layer groups until
        a certain width and height (stride * 8 for both) are met.
        In this automatic setup,
        channels will be increased as the input gets propagated
        through the network, while the
        width/height get reduced due to stride.

        Future work includes adding a parameter to control the target width and height.

    :param fc_repr_dim:
        Output dimension of the last FC layer in the network which accepts the outputs
        from the convolutional layer.

    :param channel_exp_base:
        Which power of 2 to use in order to set the number of channels in the network.
        For example, setting ``channel_exp_base=3`` means that 2**3=8 channels will be
        used.

    :param first_channel_expansion:
        Factor to extend the first layer channels.

    :param kernel_width:
        Base kernel width of the convolutions.

    :param first_kernel_expansion_width:
        Factor to extend the first kernel's width.

    :param down_stride_width:
        Down stride of the convolutional layers along the width.

    :param first_stride_expansion_width:
        Factor to extend the first layer stride along the width.

    :param dilation_factor_width:
        Base dilation factor of the convolutions along the width in the network.

    :param kernel_height:
        Base kernel height of the convolutions.

    :param first_kernel_expansion_height:
        Factor to extend the first kernel's height.

    :param down_stride_height:
        Down stride of the convolutional layers along the height.

    :param first_stride_expansion_height:
        Factor to extend the first layer stride along the height.

    :param dilation_factor_height:
        Base dilation factor of the convolutions along the height in the network.

    :param cutoff:
        If the *resulting* dimension of width * height of adding a successive block
        is less than this value, will stop adding residual blocks to the
        model in the automated case (i.e., if the layers argument is not specified).

    :param rb_do:
        Dropout in the convolutional residual blocks.

    :param stochastic_depth_p:
        Probability of dropping input.

    :param attention_inclusion_cutoff:
        If the dimension of width * height is less than this value, attention will be
        included in the model across channels and width * height after that point.

    :param l1:
        L1 regularization to apply to the first layer.
    """

    layers: Union[None, List[int]] = None

    fc_repr_dim: int = 32

    channel_exp_base: int = 2
    first_channel_expansion: int = 1

    kernel_width: int = 12
    first_kernel_expansion_width: int = 1
    down_stride_width: int = 4
    first_stride_expansion_width: int = 1
    dilation_factor_width: int = 1

    kernel_height: int = 4
    first_kernel_expansion_height: int = 1
    down_stride_height: int = 1
    first_stride_expansion_height: int = 1
    dilation_factor_height: int = 1

    cutoff: int = 32

    rb_do: float = 0.00

    stochastic_depth_p: float = 0.00

    attention_inclusion_cutoff: int = 0

    l1: float = 0.00


def _validate_cnn_config(model_config: CNNModelConfig) -> None:
    mc = model_config

    if mc.down_stride_width > mc.kernel_width:
        raise ValueError(
            f"Down stride width {mc.down_stride_width}"
            f" is greater than kernel width {mc.kernel_width}."
            f"This is currently not supported."
        )

    if mc.down_stride_height > mc.kernel_height:
        raise ValueError(
            f"Down stride height {mc.down_stride_height} "
            f" is greater than kernel height {mc.kernel_height}."
            f"This is currently not supported."
        )

    first_stride_w = mc.down_stride_width * mc.first_stride_expansion_width
    first_kernel_w = mc.kernel_width * mc.first_kernel_expansion_width

    first_stride_h = mc.down_stride_height * mc.first_stride_expansion_height
    first_kernel_h = mc.kernel_height * mc.first_kernel_expansion_height

    if first_stride_w > first_kernel_w:
        raise ValueError(
            f"Effective down stride width {first_stride_w}"
            f" (down stride width {mc.down_stride_width}"
            f" times first stride expansion width {mc.first_stride_expansion_width})"
            f" is greater than effective kernel width {first_kernel_w}"
            f" (kernel width {mc.kernel_width}"
            f" times first kernel expansion width {mc.first_kernel_expansion_width})."
            f"This is currently not supported."
        )

    if first_stride_h > first_kernel_h:
        raise ValueError(
            f"Effective down stride height {first_stride_h} "
            f" (down stride height {mc.down_stride_height} "
            f"times first stride expansion height {mc.first_stride_expansion_height})"
            f" is greater than effective kernel height {first_kernel_h}"
            f" (kernel height {mc.kernel_height}"
            f" times first kernel expansion height {mc.first_kernel_expansion_height})."
            f"This is currently not supported."
        )

    positive_int_params = [
        "fc_repr_dim",
        "first_channel_expansion",
        "kernel_width",
        "first_kernel_expansion_width",
        "down_stride_width",
        "first_stride_expansion_width",
        "dilation_factor_width",
        "kernel_height",
        "first_kernel_expansion_height",
        "down_stride_height",
        "first_stride_expansion_height",
        "dilation_factor_height",
        "cutoff",
    ]
    for param in positive_int_params:
        if getattr(mc, param) <= 0:
            raise ValueError(f"{param} must be a positive integer.")

    float_params = ["rb_do", "stochastic_depth_p"]
    for param in float_params:
        value = getattr(mc, param)
        if not (0 <= value <= 1):
            raise ValueError(f"{param} must be in the range [0, 1].")

    if mc.l1 < 0:
        raise ValueError("l1 must be non-negative.")

    if mc.layers is not None:
        if not all(isinstance(layer, int) and layer > 0 for layer in mc.layers):
            raise ValueError(
                "layers must be a list of positive integers "
                "(or None for automatic setup)."
            )


class CNNModel(nn.Module):
    def __init__(
        self,
        model_config: CNNModelConfig,
        data_dimensions: "DataDimensions",
    ):
        super().__init__()

        _validate_cnn_config(model_config=model_config)

        self.model_config = model_config
        self.data_dimensions = data_dimensions

        self.pos_representation = GeneralPositionalEmbedding(
            embedding_dim=self.data_dimensions.height,
            max_length=self.data_dimensions.width,
            dropout=0.0,
        )

        self.conv = nn.Sequential(
            *_make_conv_layers(
                residual_blocks=self.residual_blocks,
                cnn_model_configuration=self.model_config,
                data_dimensions=self.data_dimensions,
            )
        )

        size_func = pytorch_utils.calc_size_after_conv_sequence
        size_after_conv_w, size_after_conv_h = size_func(
            input_width=self.data_dimensions.width,
            input_height=self.data_dimensions.height,
            conv_sequence=self.conv,
        )
        self.data_size_after_conv = size_after_conv_w * size_after_conv_h

        self.no_out_channels = self.conv[-1].out_channels

        self.fc = nn.Sequential(
            OrderedDict(
                {
                    "fc_1_norm_1": nn.LayerNorm(normalized_shape=self.fc_1_in_features),
                    "fc_1_act_1": Swish(),
                    "fc_1_linear_1": nn.Linear(
                        in_features=self.fc_1_in_features,
                        out_features=self.model_config.fc_repr_dim,
                        bias=True,
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
                nn.init.kaiming_normal_(tensor=m.weight, a=0.5, mode="fan_out")

    @property
    def residual_blocks(self) -> List[int]:
        mc = self.model_config
        if not mc.layers:
            stride_w = mc.down_stride_width
            stride_h = mc.down_stride_height

            if stride_w < 2 and stride_h < 2:
                raise ValueError(
                    "At least one of the strides must be greater than 1, "
                    "when automatic residual block calculation is used"
                    "for CNN model creation. Got strides: "
                    f"down_stride_width: {stride_w} "
                    f"and down_stride_height: {stride_h}."
                )

            residual_blocks = auto_find_no_cnn_residual_blocks_needed(
                input_size_w=self.data_dimensions.width,
                kernel_w=mc.kernel_width,
                first_kernel_expansion_w=mc.first_kernel_expansion_width,
                stride_w=mc.down_stride_width,
                first_stride_expansion_w=mc.first_stride_expansion_width,
                dilation_w=mc.dilation_factor_width,
                input_size_h=self.data_dimensions.height,
                kernel_h=mc.kernel_height,
                first_kernel_expansion_h=mc.first_kernel_expansion_height,
                stride_h=mc.down_stride_height,
                first_stride_expansion_h=mc.first_stride_expansion_height,
                dilation_h=mc.dilation_factor_height,
                cutoff=mc.cutoff,
            )

            logger.debug(
                "No residual blocks specified in CL args, using input "
                "%s based on size approximation calculation.",
                residual_blocks,
            )
            return residual_blocks

        return self.model_config.layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.pos_representation(input)
        out = self.conv(out)
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

    first_conv_channels = 2**mc.channel_exp_base * mc.first_channel_expansion

    down_stride_w = mc.down_stride_width
    first_conv_kernel_w = mc.kernel_width * mc.first_kernel_expansion_width
    first_conv_stride_w = down_stride_w * mc.first_stride_expansion_width

    first_kernel_w, first_pad_w = pytorch_utils.calc_conv_params_needed(
        input_size=data_dimensions.width,
        kernel_size=first_conv_kernel_w,
        stride=first_conv_stride_w,
        dilation=1,
    )

    down_stride_h = mc.down_stride_height
    first_conv_kernel_h = mc.kernel_height * mc.first_kernel_expansion_height
    first_conv_stride_h = mc.down_stride_height * mc.first_stride_expansion_height

    first_kernel_h, first_pad_h = pytorch_utils.calc_conv_params_needed(
        input_size=data_dimensions.height,
        kernel_size=first_conv_kernel_h,
        stride=first_conv_stride_h,
        dilation=1,
    )

    conv_blocks = [
        FirstCNNBlock(
            in_channels=data_dimensions.channels,
            out_channels=first_conv_channels,
            conv_1_kernel_h=first_kernel_h,
            conv_1_kernel_w=first_kernel_w,
            conv_1_padding_w=first_pad_w,
            conv_1_padding_h=first_pad_h,
            down_stride_w=first_conv_stride_w,
            dilation_w=1,
            dilation_h=1,
            rb_do=mc.rb_do,
        )
    ]

    for layer_arch_idx, layer_arch_layers in enumerate(residual_blocks):
        for layer in range(layer_arch_layers):
            cur_layer = _get_conv_residual_block(
                conv_blocks=conv_blocks,
                layer_arch_idx=layer_arch_idx,
                down_stride_w=down_stride_w,
                down_stride_h=down_stride_h,
                cnn_config=cnn_model_configuration,
                data_dimensions=data_dimensions,
            )

            conv_blocks.append(cur_layer)

            cur_width, cur_height = pytorch_utils.calc_size_after_conv_sequence(
                input_width=data_dimensions.width,
                input_height=data_dimensions.height,
                conv_sequence=nn.Sequential(*conv_blocks),
            )

            if cur_height * cur_width <= mc.attention_inclusion_cutoff:
                cur_attention_block = ConvAttentionBlock(
                    channels=cur_layer.out_channels,
                    width=cur_width,
                    height=cur_height,
                )
                conv_blocks.append(cur_attention_block)

    return conv_blocks


def _get_conv_residual_block(
    conv_blocks: List[nn.Module],
    layer_arch_idx: int,
    down_stride_w: int,
    down_stride_h: int,
    cnn_config: CNNModelConfig,
    data_dimensions: "DataDimensions",
) -> CNNResidualBlock:
    mc = cnn_config

    cur_conv = nn.Sequential(*conv_blocks)
    cur_width, cur_height = pytorch_utils.calc_size_after_conv_sequence(
        input_width=data_dimensions.width,
        input_height=data_dimensions.height,
        conv_sequence=cur_conv,
    )

    cur_block_number = (
        len([i for i in conv_blocks if isinstance(i, CNNResidualBlock)]) + 1
    )
    cur_dilation_factor_w = _get_cur_dilation(
        dilation_factor=mc.dilation_factor_width,
        size=cur_width,
        block_number=cur_block_number,
    )

    cur_dilation_factor_h = _get_cur_dilation(
        dilation_factor=mc.dilation_factor_height,
        size=cur_height,
        block_number=cur_block_number,
    )

    cur_kernel_w, cur_padding_w = pytorch_utils.calc_conv_params_needed(
        input_size=cur_width,
        kernel_size=mc.kernel_width,
        stride=down_stride_w,
        dilation=cur_dilation_factor_w,
    )

    cur_kernel_h, cur_padding_h = pytorch_utils.calc_conv_params_needed(
        input_size=cur_height,
        kernel_size=mc.kernel_height,
        stride=down_stride_h,
        dilation=cur_dilation_factor_h,
    )

    cur_in_channels = conv_blocks[-1].out_channels
    cur_out_channels = 2 ** (mc.channel_exp_base + layer_arch_idx)

    cur_layer = CNNResidualBlock(
        in_channels=cur_in_channels,
        out_channels=cur_out_channels,
        conv_1_kernel_w=cur_kernel_w,
        conv_1_padding_w=cur_padding_w,
        conv_1_kernel_h=cur_kernel_h,
        conv_1_padding_h=cur_padding_h,
        down_stride_w=down_stride_w,
        dilation_w=cur_dilation_factor_w,
        dilation_h=cur_dilation_factor_h,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=mc.rb_do,
        stochastic_depth_p=mc.stochastic_depth_p,
    )

    return cur_layer


def _get_cur_dilation(dilation_factor: int, size: int, block_number: int):
    """
    Note that block_number refers to the number of residual blocks (not first block
    or self attention).
    """

    if size == 1:
        return 1

    dilation = dilation_factor**block_number

    while dilation >= size:
        dilation = dilation // dilation_factor

    return dilation


def auto_find_no_cnn_residual_blocks_needed(
    input_size_w: int,
    kernel_w: int,
    first_kernel_expansion_w: int,
    stride_w: int,
    first_stride_expansion_w: int,
    dilation_w: int,
    input_size_h: int,
    kernel_h: int,
    first_kernel_expansion_h: int,
    stride_h: int,
    first_stride_expansion_h: int,
    dilation_h: int,
    cutoff: int,
) -> List[int]:
    """
    Used in order to calculate / set up residual blocks specifications as a list
    automatically when they are not passed in as CL args, based on the minimum
    size after the residual block convolutions.

    We have 2 residual_blocks per channel depth until we have a total of 8 blocks,
    then the rest is put in the third depth index (following resnet convention).

    That is with a base channel depth of 32, we have these depths in the list:
    [32, 64, 128, 256].

    Examples
    ------
    3 blocks --> [2, 1]
    7 blocks --> [2, 2, 2, 1]
    10 blocks --> [2, 2, 4, 2]
    """

    if (stride_w == 1 and input_size_w > cutoff) or (
        stride_h == 1 and input_size_h > cutoff
    ):
        err_dim = "width" if stride_w == 1 else "height"
        logger.warning(
            f"With stride=1, the {err_dim} size "
            f"({input_size_w if err_dim == 'width' else input_size_h}) "
            f"cannot be larger than the cutoff ({cutoff}). "
            f"This would result in an infinite loop."
            f"Will stop when no more reduction is possible,"
            f"despite not reaching the cutoff."
        )

    k_size_w_first, padding_w_first = pytorch_utils.calc_conv_params_needed(
        input_size=input_size_w,
        kernel_size=kernel_w * first_kernel_expansion_w,
        stride=stride_w * first_stride_expansion_w,
        dilation=dilation_w,
    )

    k_size_h_first, padding_h_first = pytorch_utils.calc_conv_params_needed(
        input_size=input_size_h,
        kernel_size=kernel_h * first_kernel_expansion_h,
        stride=stride_h * first_stride_expansion_h,
        dilation=dilation_h,
    )

    cur_size_w = pytorch_utils.conv_output_formula(
        input_size=input_size_w,
        kernel_size=k_size_w_first,
        stride=stride_w * first_stride_expansion_w,
        dilation=dilation_w,
        padding=padding_w_first,
    )
    cur_size_h = pytorch_utils.conv_output_formula(
        input_size=input_size_h,
        kernel_size=k_size_h_first,
        stride=stride_h * first_stride_expansion_h,
        dilation=dilation_h,
        padding=padding_h_first,
    )

    residual_blocks = [0] * 4
    while True:
        cur_kernel_w, cur_padding_w = pytorch_utils.calc_conv_params_needed(
            input_size=cur_size_w,
            kernel_size=kernel_w,
            stride=stride_w,
            dilation=dilation_w,
        )

        cur_kernel_h, cur_padding_h = pytorch_utils.calc_conv_params_needed(
            input_size=cur_size_h,
            kernel_size=kernel_h,
            stride=stride_h,
            dilation=dilation_h,
        )

        cur_size_w_next = pytorch_utils.conv_output_formula(
            input_size=cur_size_w,
            kernel_size=cur_kernel_w,
            stride=stride_w,
            dilation=dilation_w,
            padding=cur_padding_w,
        )

        cur_size_h_next = pytorch_utils.conv_output_formula(
            input_size=cur_size_h,
            kernel_size=cur_kernel_h,
            stride=stride_h,
            dilation=dilation_h,
            padding=cur_padding_h,
        )

        cannot_reduce_more = (
            cur_size_w == cur_size_w_next and cur_size_h == cur_size_h_next
        )
        w_kernel_larger = cur_kernel_w > cur_size_w
        h_kernel_larger = cur_kernel_h > cur_size_h
        kernel_too_large = w_kernel_larger or h_kernel_larger
        if (
            cur_size_w_next * cur_size_h_next < cutoff
            or cannot_reduce_more
            or kernel_too_large
        ):
            if cannot_reduce_more:
                logger.warning(
                    f"Could not reduce size more, "
                    f"despite not reaching the cutoff ({cutoff})."
                )

            break
        else:
            cur_no_blocks = sum(residual_blocks)

            if cur_no_blocks >= 8:
                residual_blocks[2] += 1
            else:
                cur_index = cur_no_blocks // 2
                residual_blocks[cur_index] += 1

            cur_size_w, cur_size_h = cur_size_w_next, cur_size_h_next

    return [i for i in residual_blocks if i != 0]


class GeneralPositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = torch.nn.Parameter(
            data=torch.zeros(1, self.embedding_dim, self.max_length),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.embedding
        return self.dropout(x)
