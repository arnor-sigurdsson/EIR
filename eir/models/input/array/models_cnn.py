from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

import torch
from sympy import Symbol
from sympy.solvers import solve
from torch import nn

from eir.models.layers.cnn_layers import (
    CNNResidualBlock,
    ConvAttentionBlock,
    DownSamplingResidualBlock,
    FirstCNNBlock,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

logger = get_logger(__name__)


@dataclass
class CNNModelConfig:
    """
    :param layers:
        A list that controls the number of layers and channels in the model.
        Each element in the list represents a layer group with a specified number of
        layers and channels. Specifically,

        - The first element in the list refers to the number of layers with the
          number of channels exactly as specified by the ``channel_exp_base`` parameter.

        - The subsequent elements in the list correspond to an increased number
          of channels, doubling with each step. For instance, if ``channel_exp_base=3``
          (i.e., ``2**3=8`` channels), and the ``layers`` list is ``[5, 3, 2]``,
          the model would be constructed as follows,

            - First case: 5 layers with 8 channels
            - Second case: 3 layers with 16 channels (doubling from the previous case)
            - Third case: 2 layers with 32 channels (doubling from the previous case)

        - The model currently supports a maximum of 4 elements in the list.

        - If set to ``None``, the model will automatically set up the number of
          layer groups until a certain width and height (``stride * 8`` for both)
          are met. In this automatic setup, channels will be increased as the input
          gets propagated through the network, while the width/height get reduced
          due to stride.

        Future work includes adding a parameter to control the target width and height.

    :param num_output_features:
        Output dimension of the last FC layer in the network which accepts the outputs
        from the convolutional layer. If set to 0, the output will be passed through
        directly to the fusion module.

    :param channel_exp_base:
        Which power of 2 to use in order to set the number of channels in the network.
        For example, setting ``channel_exp_base=3`` means that 2**3=8 channels will be
        used.

    :param first_channel_expansion:
        Factor to extend the first layer channels.

    :param kernel_width:
        Base kernel width of the convolutions.

    :param first_kernel_expansion_width:
        Factor to extend the first kernel's width. The result of the multiplication
        will be rounded to the nearest integer.

    :param down_stride_width:
        Down stride of the convolutional layers along the width.

    :param first_stride_expansion_width:
        Factor to extend the first layer stride along the width. The result of the
        multiplication will be rounded to the nearest integer.

    :param dilation_factor_width:
        Base dilation factor of the convolutions along the width in the network.

    :param kernel_height:
        Base kernel height of the convolutions.

    :param first_kernel_expansion_height:
        Factor to extend the first kernel's height. The result of the multiplication
        will be rounded to the nearest integer.

    :param down_stride_height:
        Down stride of the convolutional layers along the height.

    :param first_stride_expansion_height:
        Factor to extend the first layer stride along the height.
        The result of the multiplication will be rounded to the nearest integer.

    :param dilation_factor_height:
        Base dilation factor of the convolutions along the height in the network.

    :param allow_first_conv_size_reduction:
        If set to False, will not allow the first convolutional layer to reduce the
        size of the input. Setting this is true if you want to ensure that the first
        convolutional layer reduces the size of the input, for example when the input
        is very large, and we want to compress it early.

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
        included in the model across channels and width * height as embedding dimension
        after that point (with the channels representing the length of the sequence).

    :param l1:
        L1 regularization to apply to the first layer.
    """

    layers: Union[None, List[int]] = None

    num_output_features: int = 0

    channel_exp_base: int = 2
    first_channel_expansion: int = 1

    kernel_width: int = 12
    first_kernel_expansion_width: float = 1.0
    down_stride_width: int = 4
    first_stride_expansion_width: float = 1.0
    dilation_factor_width: int = 1

    kernel_height: int = 4
    first_kernel_expansion_height: float = 1.0
    down_stride_height: int = 1
    first_stride_expansion_height: float = 1.0
    dilation_factor_height: int = 1

    allow_first_conv_size_reduction: bool = True

    down_sample_every_n_blocks: Optional[int] = 2

    cutoff: int = 32

    rb_do: float = 0.00

    stochastic_depth_p: float = 0.00

    attention_inclusion_cutoff: int = 256

    l1: float = 0.00


def _validate_cnn_config(model_config: CNNModelConfig) -> None:
    mc = model_config

    if mc.down_stride_width > mc.kernel_width:
        raise ValueError(
            f"Down stride width {mc.down_stride_width}"
            f" is greater than kernel width {mc.kernel_width}. "
            f"This is currently not supported."
        )

    if mc.down_stride_height > mc.kernel_height:
        raise ValueError(
            f"Down stride height {mc.down_stride_height} "
            f" is greater than kernel height {mc.kernel_height}."
            f"This is currently not supported."
        )

    first_stride_w = int(round(mc.down_stride_width * mc.first_stride_expansion_width))
    first_kernel_w = int(round(mc.kernel_width * mc.first_kernel_expansion_width))

    first_stride_h = int(
        round(mc.down_stride_height * mc.first_stride_expansion_height)
    )
    first_kernel_h = int(round(mc.kernel_height * mc.first_kernel_expansion_height))

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

        size_func = calc_size_after_conv_sequence
        size_after_conv_w, size_after_conv_h = size_func(
            input_width=self.data_dimensions.width,
            input_height=self.data_dimensions.height,
            conv_sequence=self.conv,
        )
        self.data_size_after_conv = size_after_conv_w * size_after_conv_h

        self.no_out_channels = self.conv[-1].out_channels

        self.output_shape = (self.no_out_channels, size_after_conv_h, size_after_conv_w)

        self.final_layer = (
            nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(
                    self.no_out_channels * self.data_size_after_conv,
                    model_config.num_output_features,
                ),
            )
            if model_config.num_output_features > 0
            else nn.Identity()
        )

        self._init_weights()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.conv[0].conv_1.weight

    @property
    def num_out_features(self) -> int:
        if self.model_config.num_output_features > 0:
            return self.model_config.num_output_features
        else:
            return self.no_out_channels * self.data_size_after_conv

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
            down_every = mc.down_sample_every_n_blocks

            if stride_w < 2 and stride_h < 2 and not down_every:
                raise ValueError(
                    "At least one of the strides must be greater than 1, "
                    "or down_sample_every_n_blocks must be set "
                    "when automatic residual block calculation is used "
                    "for CNN model creation. Got: "
                    f"down_stride_width: {stride_w}, "
                    f"down_stride_height: {stride_h}, "
                    f" and down_sample_every_n_blocks: {down_every}."
                )

            residual_blocks = auto_find_no_cnn_residual_blocks_needed(
                input_size_w=self.data_dimensions.width,
                kernel_w=mc.kernel_width,
                first_kernel_expansion_w=mc.first_kernel_expansion_width,
                stride_w=mc.down_stride_width,
                first_stride_expansion_w=mc.first_stride_expansion_width,
                dilation_w=mc.dilation_factor_width,
                down_sample_every_n_blocks=mc.down_sample_every_n_blocks,
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

        assert isinstance(mc.layers, list)
        return mc.layers

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.pos_representation(input)
        out = self.conv(out)
        out = self.final_layer(out)

        return out


def _make_conv_layers(
    residual_blocks: List[int],
    cnn_model_configuration: CNNModelConfig,
    data_dimensions: "DataDimensions",
) -> List[nn.Module]:
    mc = cnn_model_configuration

    first_conv_channels = 2**mc.channel_exp_base * mc.first_channel_expansion

    down_stride_w = mc.down_stride_width
    first_conv_kernel_w = int(round(mc.kernel_width * mc.first_kernel_expansion_width))
    first_conv_stride_w = int(round(down_stride_w * mc.first_stride_expansion_width))

    conv_param_suggestion_w = calc_conv_params_needed(
        input_size=data_dimensions.width,
        kernel_size=first_conv_kernel_w,
        stride=first_conv_stride_w,
        dilation=mc.dilation_factor_width,
        allow_non_stride_reduction=mc.allow_first_conv_size_reduction,
    )

    down_stride_h = mc.down_stride_height
    first_conv_kernel_h = int(
        round(mc.kernel_height * mc.first_kernel_expansion_height)
    )
    first_conv_stride_h = int(
        round(mc.down_stride_height * mc.first_stride_expansion_height)
    )

    conv_param_suggestion_h = calc_conv_params_needed(
        input_size=data_dimensions.height,
        kernel_size=first_conv_kernel_h,
        stride=first_conv_stride_h,
        dilation=mc.dilation_factor_height,
        allow_non_stride_reduction=mc.allow_first_conv_size_reduction,
    )

    conv_blocks: list[FirstCNNBlock | CNNResidualBlock | ConvAttentionBlock | nn.Module]
    conv_blocks = [
        FirstCNNBlock(
            in_channels=data_dimensions.channels,
            out_channels=first_conv_channels,
            conv_1_kernel_h=conv_param_suggestion_h.kernel_size,
            conv_1_kernel_w=conv_param_suggestion_w.kernel_size,
            conv_1_padding_w=conv_param_suggestion_w.padding,
            conv_1_padding_h=conv_param_suggestion_h.padding,
            down_stride_w=first_conv_stride_w,
            down_stride_h=first_conv_stride_h,
            dilation_w=conv_param_suggestion_w.dilation,
            dilation_h=conv_param_suggestion_h.dilation,
            rb_do=mc.rb_do,
        )
    ]

    first_width, first_height = calc_size_after_conv_sequence(
        input_width=data_dimensions.width,
        input_height=data_dimensions.height,
        conv_sequence=nn.Sequential(*conv_blocks),
    )
    do_add_attention = _do_add_attention(
        width=first_width,
        height=first_height,
        attention_inclusion_cutoff=mc.attention_inclusion_cutoff,
    )

    if do_add_attention:
        last_block = conv_blocks[-1]
        assert isinstance(last_block, (CNNResidualBlock, FirstCNNBlock))
        cur_attention_block = ConvAttentionBlock(
            channels=last_block.out_channels,
            width=first_width,
            height=first_height,
        )
        conv_blocks.append(cur_attention_block)

    down_every = mc.down_sample_every_n_blocks
    for block_arch_idx, block_arch_blocks in enumerate(residual_blocks):
        for layer in range(block_arch_blocks):
            cur_block = _get_conv_residual_block(
                conv_blocks=conv_blocks,
                layer_arch_idx=block_arch_idx,
                down_stride_w=down_stride_w,
                down_stride_h=down_stride_h,
                cnn_config=cnn_model_configuration,
                data_dimensions=data_dimensions,
            )

            conv_blocks.append(cur_block)

            cur_width, cur_height = calc_size_after_conv_sequence(
                input_width=data_dimensions.width,
                input_height=data_dimensions.height,
                conv_sequence=nn.Sequential(*conv_blocks),
            )

            n_blocks = len([i for i in conv_blocks if isinstance(i, CNNResidualBlock)])
            if down_every and n_blocks % down_every == 0:
                down_block = DownSamplingResidualBlock(
                    in_channels=conv_blocks[-1].out_channels,
                    in_width=cur_width,
                    in_height=cur_height,
                )
                conv_blocks.append(down_block)

                cur_width, cur_height = calc_size_after_conv_sequence(
                    input_width=data_dimensions.width,
                    input_height=data_dimensions.height,
                    conv_sequence=nn.Sequential(*conv_blocks),
                )

            do_add_attention = _do_add_attention(
                width=cur_width,
                height=cur_height,
                attention_inclusion_cutoff=mc.attention_inclusion_cutoff,
            )
            if do_add_attention:
                cur_attention_block = ConvAttentionBlock(
                    channels=cur_block.out_channels,
                    width=cur_width,
                    height=cur_height,
                )
                conv_blocks.append(cur_attention_block)

    return conv_blocks


def _do_add_attention(
    width: int,
    height: int,
    attention_inclusion_cutoff: int,
) -> bool:
    if attention_inclusion_cutoff == 0:
        return False

    if width * height <= attention_inclusion_cutoff:
        return True

    return False


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
    cur_width, cur_height = calc_size_after_conv_sequence(
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
        kernel_size=mc.kernel_width,
    )
    conv_param_suggestion_w = calc_conv_params_needed(
        input_size=cur_width,
        kernel_size=mc.kernel_width,
        stride=down_stride_w,
        dilation=cur_dilation_factor_w,
    )

    cur_dilation_factor_h = _get_cur_dilation(
        dilation_factor=mc.dilation_factor_height,
        size=cur_height,
        block_number=cur_block_number,
        kernel_size=mc.kernel_height,
    )
    conv_param_suggestion_h = calc_conv_params_needed(
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
        conv_1_kernel_w=conv_param_suggestion_w.kernel_size,
        conv_1_padding_w=conv_param_suggestion_w.padding,
        down_stride_w=conv_param_suggestion_w.stride,
        dilation_w=conv_param_suggestion_w.dilation,
        conv_1_kernel_h=conv_param_suggestion_h.kernel_size,
        conv_1_padding_h=conv_param_suggestion_h.padding,
        down_stride_h=conv_param_suggestion_h.stride,
        dilation_h=conv_param_suggestion_h.dilation,
        full_preact=True if len(conv_blocks) == 1 else False,
        rb_do=mc.rb_do,
        stochastic_depth_p=mc.stochastic_depth_p,
    )

    return cur_layer


def _get_cur_dilation(
    dilation_factor: int, size: int, block_number: int, kernel_size: int
):
    """
    Note that block_number refers to the number of full residual blocks
    (excluding the first one).
    """
    if size == 1 or kernel_size == 1:
        return 1

    dilation = dilation_factor**block_number

    max_dilation = max(1, (size - 1) // (kernel_size - 1))
    while dilation > max_dilation:
        dilation = dilation // dilation_factor

    return dilation


def auto_find_no_cnn_residual_blocks_needed(
    input_size_w: int,
    kernel_w: int,
    first_kernel_expansion_w: float,
    stride_w: int,
    first_stride_expansion_w: float,
    dilation_w: int,
    down_sample_every_n_blocks: Optional[int],
    input_size_h: int,
    kernel_h: int,
    first_kernel_expansion_h: float,
    stride_h: int,
    first_stride_expansion_h: float,
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

    if (
        (stride_w == 1 and input_size_w > cutoff)
        or (stride_h == 1 and input_size_h > cutoff)
    ) and not down_sample_every_n_blocks:
        err_dim = "width" if stride_w == 1 else "height"
        logger.warning(
            f"With stride=1, the {err_dim} size "
            f"({input_size_w if err_dim == 'width' else input_size_h}) "
            f"cannot be larger than the cutoff ({cutoff}). "
            f"This would result in an infinite loop."
            f"Will stop when no more reduction is possible,"
            f"despite not reaching the cutoff."
        )

    first_kernel_w = int(round(kernel_w * first_kernel_expansion_w))
    first_stride_w = int(round(stride_w * first_stride_expansion_w))

    conv_param_suggestion_w_first = calc_conv_params_needed(
        input_size=input_size_w,
        kernel_size=first_kernel_w,
        stride=first_stride_w,
        dilation=dilation_w,
    )

    first_kernel_h = int(round(kernel_h * first_kernel_expansion_h))
    first_stride_h = int(round(stride_h * first_stride_expansion_h))

    conv_param_suggestion_h_first = calc_conv_params_needed(
        input_size=input_size_h,
        kernel_size=first_kernel_h,
        stride=first_stride_h,
        dilation=dilation_h,
    )

    cur_size_w = conv_output_formula(
        input_size=input_size_w,
        kernel_size=conv_param_suggestion_w_first.kernel_size,
        stride=first_stride_w,
        dilation=dilation_w,
        padding=conv_param_suggestion_w_first.padding,
    )
    cur_size_h = conv_output_formula(
        input_size=input_size_h,
        kernel_size=conv_param_suggestion_h_first.kernel_size,
        stride=first_stride_h,
        dilation=dilation_h,
        padding=conv_param_suggestion_h_first.padding,
    )

    residual_blocks = [0] * 4
    down_every = down_sample_every_n_blocks
    while True:
        if down_every is not None and sum(residual_blocks) % down_every == 0:
            cur_size_w = max(1, cur_size_w // 2)
            cur_size_h = max(1, cur_size_h // 2)

        conv_param_suggestion_w = calc_conv_params_needed(
            input_size=cur_size_w,
            kernel_size=kernel_w,
            stride=stride_w,
            dilation=dilation_w,
        )
        cur_size_w_next = conv_output_formula(
            input_size=cur_size_w,
            kernel_size=conv_param_suggestion_w.kernel_size,
            stride=conv_param_suggestion_w.stride,
            dilation=conv_param_suggestion_w.dilation,
            padding=conv_param_suggestion_w.padding,
        )

        conv_param_suggestion_h = calc_conv_params_needed(
            input_size=cur_size_h,
            kernel_size=kernel_h,
            stride=stride_h,
            dilation=dilation_h,
        )
        cur_size_h_next = conv_output_formula(
            input_size=cur_size_h,
            kernel_size=conv_param_suggestion_h.kernel_size,
            stride=conv_param_suggestion_h.stride,
            dilation=conv_param_suggestion_h.dilation,
            padding=conv_param_suggestion_h.padding,
        )

        cannot_reduce_more = (
            cur_size_w == cur_size_w_next and cur_size_h == cur_size_h_next
        )

        w_kernel_larger = conv_param_suggestion_w.kernel_size > cur_size_w
        h_kernel_larger = conv_param_suggestion_h.kernel_size > cur_size_h
        kernel_too_large = w_kernel_larger or h_kernel_larger
        if (
            cur_size_w_next * cur_size_h_next < cutoff
            or (cannot_reduce_more and down_every is None)
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


def calc_size_after_conv_sequence(
    input_width: int,
    input_height: int,
    conv_sequence: nn.Sequential,
) -> Tuple[int, int]:
    current_width = input_width
    current_height = input_height
    for block_index, block in enumerate(conv_sequence):
        conv_operations = [i for i in vars(block)["_modules"] if i.find("conv") != -1]

        for operation_index, operation in enumerate(conv_operations):
            conv_layer = vars(block)["_modules"][operation]

            padded_height = current_height + 2 * conv_layer.padding[0]
            padded_width = current_width + 2 * conv_layer.padding[1]
            if any(
                k > s
                for k, s in zip(conv_layer.kernel_size, (padded_height, padded_width))
            ):
                raise ValueError(
                    f"Kernel size of layer "
                    f"{block_index}.{operation_index} ({operation}) "
                    f"exceeds padded input size in one or more dimensions. "
                    f"Original input size (hxw): {current_height}x{current_width} "
                    f"(this is likely the source of the problem, especially if "
                    f"error layer is conv_1). "
                    f"Padded input size became (hxw): {padded_height}x{padded_width}. "
                    f"Kernel size: {conv_layer.kernel_size}. "
                    "Please adjust the kernel size to ensure the it "
                    "does not exceed the padded input size for each dimension."
                )

            new_width = _calc_layer_output_size_for_axis(
                size=current_width, layer=conv_layer, axis=1
            )
            new_height = _calc_layer_output_size_for_axis(
                size=current_height, layer=conv_layer, axis=0
            )

            if int(new_width) == 0 or int(new_height) == 0:
                kernel_size = conv_layer.kernel_size
                stride = conv_layer.stride
                padding = conv_layer.padding
                dilation = conv_layer.dilation

                raise ValueError(
                    f"Calculated size after convolution sequence is 0 for layer "
                    f"{block_index}.{operation_index} ({operation}). "
                    f"Input size (hxw): {current_height}x{current_width}. "
                    f"Convolution parameters: kernel size = {kernel_size}, "
                    f"stride = {stride}, padding = {padding}, dilation = {dilation}. "
                    "Please adjust these parameters to ensure they are appropriate "
                    "for the input size."
                )

            current_width, current_height = new_width, new_height

    return int(current_width), int(current_height)


def _calc_layer_output_size_for_axis(
    size: int, layer: nn.Conv1d | nn.Conv2d | nn.Conv3d, axis: int
):
    kernel_size = layer.kernel_size[axis]
    padding = layer.padding[axis]
    assert isinstance(padding, int)
    stride = layer.stride[axis]
    dilation = layer.dilation[axis]

    output_size = conv_output_formula(
        input_size=size,
        padding=padding,
        dilation=dilation,
        kernel_size=kernel_size,
        stride=stride,
    )

    return output_size


@dataclass()
class ConvParamSuggestion:
    kernel_size: int
    target_size: int
    stride: int
    dilation: int
    padding: int


def calc_conv_params_needed(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    allow_non_stride_reduction: bool = False,
) -> "ConvParamSuggestion":
    if input_size < 0:
        raise ValueError("Got negative size for input width: %d", input_size)

    if stride == 1 and not allow_non_stride_reduction:
        target_size = input_size
    else:
        target_size = conv_output_formula(
            input_size=input_size,
            padding=0,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
        )

    target_size = max(1, target_size)

    param_suggestions = _get_conv_param_suggestion_iterator(
        target_size=target_size,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    solutions = []

    for s in param_suggestions:
        padding = solve_for_padding(
            input_size=input_size,
            target_size=s.target_size,
            dilation=s.dilation,
            stride=s.stride,
            kernel_size=s.kernel_size,
        )

        if padding is not None:
            assert isinstance(padding, int)
            cur_solution = ConvParamSuggestion(
                kernel_size=s.kernel_size,
                target_size=s.target_size,
                stride=s.stride,
                dilation=s.dilation,
                padding=padding,
            )
            solutions.append(cur_solution)

    if len(solutions) == 0:
        raise AssertionError(
            f"Could not find a solution for padding with the supplied conv "
            f"parameters: input size = {input_size}, kernel size = {kernel_size}, "
            f"stride = {stride}, dilation = {dilation}. "
        )

    best_solution = _choose_best_solution(
        solutions=solutions,
        input_target_size=target_size,
        input_kernel_size=kernel_size,
        input_stride=stride,
        input_dilation=dilation,
    )

    return best_solution


def _choose_best_solution(
    solutions: List[ConvParamSuggestion],
    input_target_size: int,
    input_kernel_size: int,
    input_stride: int,
    input_dilation: int,
) -> ConvParamSuggestion:
    def _calculate_distance(
        solution: ConvParamSuggestion,
    ) -> Tuple[int, Tuple[int, ...]]:
        """
        We have the second returned value as a tuple so that if >1 solutions
        have the same distance, we get a consistent ordering.

        We have *10 there to prioritize maintaining the target size
        over other parameters.
        """
        kernel_size_diff = abs(solution.kernel_size - input_kernel_size)
        stride_diff = abs(solution.stride - input_stride)
        dilation_diff = abs(solution.dilation - input_dilation)
        padding_diff = abs(solution.padding)
        target_size_diff = abs(solution.target_size - input_target_size) * 10

        distance = (
            kernel_size_diff
            + stride_diff
            + dilation_diff
            + padding_diff
            + target_size_diff
        )

        values_tuple = (
            solution.kernel_size,
            solution.stride,
            solution.dilation,
            solution.padding,
            solution.target_size,
        )

        return distance, values_tuple

    sorted_solutions = sorted(solutions, key=_calculate_distance)

    return sorted_solutions[0]


def _get_conv_param_suggestion_iterator(
    target_size: int, kernel_size: int, stride: int, dilation: int
) -> Iterator[ConvParamSuggestion]:
    def _get_range(base: int) -> List[int]:
        return [sug for sug in [base, base + 1, base - 1] if sug > 0]

    for k_size in _get_range(base=kernel_size):
        for t_size in _get_range(base=target_size):
            for s_size in _get_range(base=stride):
                for d_size in _get_range(base=dilation):
                    yield ConvParamSuggestion(
                        kernel_size=k_size,
                        target_size=t_size,
                        stride=s_size,
                        dilation=d_size,
                        padding=0,
                    )


def conv_output_formula(
    input_size: int, padding: int, dilation: int, kernel_size: int, stride: int
) -> int:
    out_size = (
        input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    ) // stride + 1
    return out_size


def solve_for_padding(
    input_size: int, target_size: int, dilation: int, stride: int, kernel_size: int
) -> Union[int, None]:
    p = Symbol("p", integer=True, nonnegative=True)
    padding = solve(
        ((input_size + (2 * p) - dilation * (kernel_size - 1) - 1) / stride + 1)
        - target_size,
        p,
    )

    if len(padding) > 0:
        return int(padding[0])

    return None
