import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn

from eir.models.fusion.fusion_attention import UniDirectionalCrossAttention
from eir.models.layers.cnn_layers import (
    ConvAttentionBlock,
    SEBlock,
    StochasticDepth,
    UpSamplingResidualBlock,
)
from eir.models.layers.norm_layers import GRN
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo
    from eir.setup.input_setup_modules.common import DataDimensions


logger = get_logger(name=__name__)


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

    :param num_ca_blocks:
          Number of cross-attention blocks to include in the model when fusing
          with other feature extractor outputs.

    :param up_every_n_blocks:
        If set, the model will upsample every n blocks. If not set, the model will
        use the down_every_n_blocks parameter in linked feature extractor, if
        that is the case and the pass-through upscale model is being used (
        the default for diffusion models, or if pass-through fusion model is used).
        Otherwise, will default to 2.

    :param n_final_extra_blocks:
        Number of extra blocks to add at the end of the model after upsampling
        has reached the target size.
    """

    channel_exp_base: int
    rb_do: float = 0.1
    stochastic_depth_p: float = 0.1
    attention_inclusion_cutoff: int = 256
    allow_pooling: bool = True
    num_ca_blocks: int = 1
    up_every_n_blocks: Optional[int] = None
    n_final_extra_blocks: int = 1


class CNNUpscaleResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        out_channels: int,
        stride: tuple[int, int],
        rb_do: float = 0.0,
        stochastic_depth_p: float = 0.0,
    ):
        super(CNNUpscaleResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.stochastic_depth_p = stochastic_depth_p

        self.conv_ds = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=in_channels,
        )

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)

        self.conv_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )

        self.rb_do = nn.Dropout2d(rb_do)
        self.act_1 = nn.GELU()

        self.grn = GRN(in_channels=out_channels)

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.upsample_identity = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
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

        out = self.conv_ds(x)

        out = self.norm_1(out)

        identity = self.upsample_identity(x)

        out = self.conv_1(out)

        out = self.act_1(out)
        out = self.grn(out)
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
    in_channels: int,
    out_channels: int,
    initial_height: int,
    initial_width: int,
    target_height: int,
    target_width: int,
    attention_inclusion_cutoff: int,
    up_sample_every_n_blocks: int,
    n_final_extra_blocks: int,
    allow_pooling: bool = True,
) -> Tuple[nn.Sequential, int, int, int]:
    blocks = nn.Sequential()
    current_height = initial_height
    current_width = initial_width
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

    n_blocks = 0
    while current_height < target_height or current_width < target_width:
        stride_height = 1
        stride_width = 1

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
        n_blocks += 1

        in_channels = out_channels

        up_every = up_sample_every_n_blocks
        if up_every and block_counter % up_every == 0:

            do_height = current_height < target_height
            do_width = current_width < target_width

            up_sampling_block = UpSamplingResidualBlock(
                in_channels=in_channels,
                in_height=current_height,
                in_width=current_width,
                upsample_height=do_height,
                upsample_width=do_width,
            )
            blocks.add_module(
                name=f"block_{len(blocks)}_upsampling",
                module=up_sampling_block,
            )

            current_height = current_height * 2 if do_height else current_height
            current_width = current_width * 2 if do_width else current_width

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

    for _ in range(n_final_extra_blocks):
        blocks.add_module(
            name=f"block_{len(blocks)}",
            module=CNNUpscaleResidualBlock(
                in_channels=in_channels,
                in_height=current_height,
                in_width=current_width,
                out_channels=out_channels,
                stride=(1, 1),
            ),
        )
        in_channels = out_channels

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

        up_every_n_blocks = model_config.up_every_n_blocks
        if not up_every_n_blocks:
            logger.debug("Using default up_every_n_blocks=2.")
            up_every_n_blocks = 2

        ratio = math.sqrt(self.target_height * self.target_width / input_size)
        initial_height = int(self.target_height / ratio)
        initial_width = int(self.target_width / ratio)

        self.initial_layer = nn.Sequential(
            nn.Linear(
                in_features=input_size,
                out_features=initial_height * initial_width,
            ),
            nn.GELU(),
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
            in_channels=1,
            target_height=self.target_height,
            target_width=self.target_width,
            out_channels=2**self.model_config.channel_exp_base,
            allow_pooling=self.model_config.allow_pooling,
            attention_inclusion_cutoff=self.model_config.attention_inclusion_cutoff,
            up_sample_every_n_blocks=up_every_n_blocks,
            n_final_extra_blocks=model_config.n_final_extra_blocks,
        )

        self.final_layer = nn.Conv2d(
            in_channels=self.block_channels,
            out_channels=self.target_channels,
            kernel_size=1,
        )

    @property
    def num_out_features(self) -> int:
        return self.target_channels * self.final_height * self.final_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        x = self.blocks(x)
        x = self.final_layer(x)
        return x


class CNNPassThroughUpscaleModel(nn.Module):
    def __init__(
        self,
        model_config: "CNNUpscaleModelConfig",
        feature_extractor_infos: dict[str, "FeatureExtractorInfo"],
        target_dimensions: "DataDimensions",
        output_name: str,
        diffusion_time_steps: Optional[int] = None,
    ) -> None:
        super(CNNPassThroughUpscaleModel, self).__init__()

        self.model_config = model_config
        self.feature_extractor_infos = feature_extractor_infos
        self.output_name = output_name

        if output_name not in feature_extractor_infos:
            raise ValueError(
                f"When using CNNPassThroughUpscaleModel, the output_name "
                f"'{output_name}' must be included as an input module, as the "
                f"passthrough model is intended to be linked with a feature "
                f"extractor."
            )

        cur_fei = feature_extractor_infos[output_name]
        cur_shape = cur_fei.output_shape
        assert cur_shape is not None

        up_every_n_blocks = model_config.up_every_n_blocks
        if not up_every_n_blocks:
            up_every_n_blocks = cur_fei.extras.get("down_every_n_blocks")
            if up_every_n_blocks:
                logger.debug(
                    "Using down_every_n_blocks=%s from linked feature extractor.",
                    up_every_n_blocks,
                )
            else:
                logger.debug("Using default up_every_n_blocks=2.")
                up_every_n_blocks = 2

        initial_channels = cur_shape[0]
        self.initial_height = cur_shape[1]
        self.initial_width = cur_shape[2]

        self.target_width = target_dimensions.width
        self.target_height = target_dimensions.height
        self.target_channels = target_dimensions.channels
        self.target_size = self.target_height * self.target_width
        self.diffusion_time_steps = diffusion_time_steps

        self.ca_layers: nn.ModuleDict = nn.ModuleDict()
        for name, fei in feature_extractor_infos.items():
            if name != output_name:
                cur_blocks = nn.ModuleList()

                for _ in range(self.model_config.num_ca_blocks):
                    cur_blocks.append(
                        CrossAttentionArrayOutBlock(
                            input_channels=initial_channels,
                            input_height=self.initial_height,
                            input_width=self.initial_width,
                            context_num_elements=fei.output_dimension,
                            context_dimension=fei.output_shape,
                        )
                    )

                self.ca_layers[name] = cur_blocks

        self.timestep_mixing_layer: nn.Identity | TimeStepMixingBlock = nn.Identity()
        if self.diffusion_time_steps is not None:
            self.timestep_mixing_layer = TimeStepMixingBlock(
                input_channels=initial_channels,
                input_height=self.initial_height,
                input_width=self.initial_width,
                n_time_steps=self.diffusion_time_steps,
            )

        (
            self.blocks,
            self.block_channels,
            self.final_height,
            self.final_width,
        ) = setup_blocks(
            in_channels=initial_channels,
            initial_height=self.initial_height,
            initial_width=self.initial_width,
            target_height=self.target_height,
            target_width=self.target_width,
            out_channels=2**self.model_config.channel_exp_base,
            allow_pooling=self.model_config.allow_pooling,
            attention_inclusion_cutoff=self.model_config.attention_inclusion_cutoff,
            up_sample_every_n_blocks=up_every_n_blocks,
            n_final_extra_blocks=model_config.n_final_extra_blocks,
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.block_channels,
                out_channels=self.target_channels,
                kernel_size=1,
            ),
        )

    @property
    def num_out_features(self) -> int:
        return self.target_channels * self.final_height * self.final_width

    def timestep_embeddings(self, t: torch.Tensor) -> torch.Tensor:
        return self.timestep_mixing_layer.time_embedding(t)

    def forward(self, input: dict[str, torch.Tensor]) -> torch.Tensor:
        out = input[self.output_name]

        if len(out.shape) == 2:
            out = out.unflatten(
                dim=1,
                sizes=(1, self.initial_height, self.initial_width),
            )

        if self.diffusion_time_steps is not None:
            t_emb = input[f"__extras_{self.output_name}"]
            out = self.timestep_mixing_layer(input=out, t_emb=t_emb)

        for name, input_tensor in input.items():
            if name == self.output_name or name.startswith("__extras_"):
                continue

            cur_cross_attention: nn.ModuleList = self.ca_layers[name]
            for block in cur_cross_attention:
                out = block(input=out, context=input_tensor)

        out = self.blocks(out)
        out = self.final_layer(out)
        return out


class TimeStepMixingBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        n_time_steps: int,
    ):
        super(TimeStepMixingBlock, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        embedding_dim = input_height * input_width

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=input_channels)
        self.act_1 = nn.GELU()

        self.time_embedding = nn.Embedding(
            num_embeddings=n_time_steps,
            embedding_dim=embedding_dim,
        )

        self.cross_attention = UniDirectionalCrossAttention(
            dim=input_height * input_width,
            dim_head=input_height * input_width,
            context_dim=embedding_dim,
            heads=1,
        )

    def forward(self, input: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        identity = input

        out = self.norm_1(input)

        # (B, C, H, W) -> (B, C, H * W) -> (B, seq, emb_dim)
        out = out.view(
            out.shape[0],
            self.input_channels,
            self.input_height * self.input_width,
        )

        # (B, emb_dim) -> (B, 1, emb_dim)
        t_emb = t_emb.unsqueeze(1)

        out = self.cross_attention(x=out, context=t_emb)

        out = self.act_1(out)

        out = out.view(
            out.shape[0],
            self.input_channels,
            self.input_height,
            self.input_width,
        )

        out = out + identity

        return out


class CrossAttentionArrayOutBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        context_num_elements: int,
        context_dimension: Optional[tuple[int, ...]],
    ):
        super(CrossAttentionArrayOutBlock, self).__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.context_dim = context_dimension

        self.conv_ds = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=input_channels,
        )

        self.norm_1 = nn.GroupNorm(num_groups=1, num_channels=input_channels)
        self.act_1 = nn.GELU()

        if self.context_dim is None:
            self.context_channels = 1
            self.context_height = 1
            self.context_width = context_num_elements
        else:
            self.context_channels = self.context_dim[0]
            self.context_height = self.context_dim[1]
            self.context_width = self.context_dim[2]

        context_emb_dim = self.context_height * self.context_width

        self.cross_attention = UniDirectionalCrossAttention(
            dim=input_height * input_width,
            dim_head=input_height * input_width,
            context_dim=context_emb_dim,
            heads=1,
        )

        self.conv_1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, input: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        identity = input

        out = self.conv_ds(input)
        out = self.norm_1(out)

        # (B, C, H, W) -> (B, C, H * W) -> (B, seq, emb_dim)
        out = out.view(
            out.shape[0],
            self.input_channels,
            self.input_height * self.input_width,
        )

        # (B, C, H, W) -> (B, C, H * W) -> (B, seq, emb_dim)
        context = context.view(
            context.shape[0],
            self.context_channels,
            self.context_height * self.context_width,
        )

        out = self.cross_attention(x=out, context=context)
        out = self.act_1(out)

        out = out.view(
            out.shape[0],
            self.input_channels,
            self.input_height,
            self.input_width,
        )

        out = self.conv_1(out)

        out = out + identity

        return out
