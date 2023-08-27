import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import torch.nn as nn
from aislib.pytorch_modules import Swish

from eir.models.layers.cnn_layers import ConvAttentionBlock, SEBlock, StochasticDepth

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions


@dataclass
class CNNUpscaleModelConfig:
    """
    :param channel_exp_base:
          Which power of 2 to use in order to set the number of channels in the network.
          For example, setting ``channel_exp_base=3`` means that 2**3=8 channels will be
          used.

      :param rb_do:
          Dropout in the convolutional residual blocks.

      :param stochastic_depth_p:
          Probability of dropping input.

      :param attention_inclusion_cutoff:
          If the dimension of width * height is less than this value, attention will be
          included in the model across channels and width * height as embedding
          dimension after that point
          (with the channels representing the length of the sequence).

      :param allow_pooling:
          Whether to allow adaptive average pooling in the model to match the target
          dimensions.
    """

    channel_exp_base: int
    rb_do: float = 0.1
    stochastic_depth_p: float = 0.1
    attention_inclusion_cutoff: int = 0
    allow_pooling: bool = True


class CNNUpscaleResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        stride: tuple[int, int],
        rb_do: float = 0.1,
        stochastic_depth_p: float = 0.1,
    ):
        super(CNNUpscaleResidualBlock, self).__init__()

        self.stochastic_depth_p = stochastic_depth_p

        self.norm_1 = nn.LayerNorm([in_channels, in_height, in_width])

        output_padding: tuple[int, int] = (stride[0] - 1, stride[1] - 1)

        self.conv_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=output_padding,
            bias=True,
        )

        self.rb_do = nn.Dropout2d(rb_do)
        self.act_1 = Swish()

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.upsample_identity = (
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=output_padding,
                bias=True,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.stochastic_depth = StochasticDepth(
            p=self.stochastic_depth_p,
            mode="batch",
        )

        self.se_block = SEBlock(
            channels=out_channels,
            reduction=16,
        )

    def forward(self, x: Any) -> Any:
        out = self.norm_1(x)

        identity = self.upsample_identity(x)

        out = self.conv_1(out)

        out = self.act_1(out)
        out = self.rb_do(out)
        out = self.conv_2(out)

        channel_recalibrations = self.se_block(out)
        out = out * channel_recalibrations

        out = self.stochastic_depth(out)

        out = out + identity

        return out


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


def setup_blocks(
    out_channels: int,
    initial_height: int,
    initial_width: int,
    target_height: int,
    target_width: int,
    attention_inclusion_cutoff: int,
    allow_pooling: bool = True,
) -> Tuple[nn.Sequential, int, int, int]:
    blocks = nn.Sequential()
    current_height, current_width = initial_height, initial_width
    in_channels = 1
    reduce_counter = 0
    block_counter = 0

    if _do_add_attention(
        width=current_width,
        height=current_height,
        attention_inclusion_cutoff=attention_inclusion_cutoff,
    ):
        cur_attention_block = ConvAttentionBlock(
            channels=in_channels,
            width=current_width,
            height=current_height,
        )
        blocks.add_module(
            name=f"block_{len(blocks)}",
            module=cur_attention_block,
        )

    while current_height < target_height or current_width < target_width:
        stride_height = 2 if current_height * 2 <= target_height else 1
        stride_width = 2 if current_width * 2 <= target_width else 1

        if stride_height == 1 and stride_width == 1:
            break

        stride = (stride_height, stride_width)

        block_counter += 1
        if block_counter % 2 == 0 and reduce_counter < 4:
            out_channels = max(out_channels // 2, 1)
            reduce_counter += 1

        blocks.add_module(
            name=f"block_{len(blocks)}",
            module=CNNUpscaleResidualBlock(
                in_channels=in_channels,
                in_height=current_height,
                in_width=current_width,
                out_channels=out_channels,
                stride=stride,
            ),
        )

        in_channels = out_channels

        current_height *= stride[0]
        current_width *= stride[1]

        if _do_add_attention(
            width=current_width,
            height=current_height,
            attention_inclusion_cutoff=attention_inclusion_cutoff,
        ):
            cur_attention_block = ConvAttentionBlock(
                channels=out_channels,
                width=current_width,
                height=current_height,
            )
            blocks.add_module(
                name=f"block_{len(blocks)}",
                module=cur_attention_block,
            )

    not_matching = current_height != target_height or current_width != target_width
    if allow_pooling and not_matching:
        blocks.add_module(
            name="pooling",
            module=nn.AdaptiveAvgPool2d(
                output_size=(target_height, target_width),
            ),
        )
        current_height = target_height
        current_width = target_width

    return blocks, in_channels, current_height, current_width


class CNNUpscaleModel(nn.Module):
    def __init__(
        self,
        model_config: CNNUpscaleModelConfig,
        data_dimensions: "DataDimensions",
        target_dimensions: "DataDimensions",
    ):
        super(CNNUpscaleModel, self).__init__()

        self.model_config = model_config

        input_size = data_dimensions.num_elements()

        self.target_width = target_dimensions.width
        self.target_height = target_dimensions.height
        self.target_channels = target_dimensions.channels

        ratio = math.sqrt(self.target_height * self.target_width / input_size)
        initial_height = int(self.target_height / ratio)
        initial_width = int(self.target_width / ratio)

        self.initial_layer = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=initial_height * initial_width,
            ),
            nn.ReLU(),
            nn.Unflatten(
                dim=1,
                unflattened_size=(1, initial_height, initial_width),
            ),
        )

        (
            self.blocks,
            self.block_channels,
            self.final_height,
            self.final_width,
        ) = setup_blocks(
            initial_height=initial_height,
            initial_width=initial_width,
            target_height=self.target_height,
            target_width=self.target_width,
            out_channels=2**self.model_config.channel_exp_base,
            allow_pooling=self.model_config.allow_pooling,
            attention_inclusion_cutoff=self.model_config.attention_inclusion_cutoff,
        )

        self.final_layer = nn.Conv2d(
            in_channels=self.block_channels,
            out_channels=self.target_channels,
            kernel_size=1,
        )

    @property
    def num_out_features(self) -> int:
        return self.target_channels * self.final_height * self.final_width

    def forward(self, x: Any) -> Any:
        x = self.initial_layer(x)
        x = self.blocks(x)
        x = self.final_layer(x)
        return x
