from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal

import torch
from torch import nn

from eir.models.fusion.fusion_attention import MetaSequenceProjection
from eir.models.input.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
    parse_dim_feedforward,
)
from eir.models.layers.attention_layers import Transformer
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo
    from eir.setup.output_setup_modules.sequence_output_setup import (
        ComputedSequenceOutputInfo,
    )

al_sequence_output_models = Literal["sequence"]

logger = get_logger(name=__name__)


@dataclass
class TransformerSequenceOutputModuleConfig(
    BasicTransformerFeatureExtractorModelConfig
):
    pass


@dataclass
class SequenceOutputModuleConfig:
    """
    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param model_type:
         Which type of image model to use.

    :param embedding_dim:
        Which dimension to use for the embeddings. If ``None``, will automatically set
        this value based on the number of tokens and attention heads.

    :param position:
        Whether to encode the token position or use learnable position embeddings.

    :param position_dropout:
        Dropout for the positional encoding / embedding.

    """

    model_init_config: TransformerSequenceOutputModuleConfig
    model_type: Literal["sequence"] = "sequence"

    embedding_dim: int = 64

    position: Literal["encode", "embed"] = "encode"
    position_dropout: float = 0.10

    projection_layer_type: Literal["auto", "lcl", "lcl_residual", "linear"] = "auto"


class SequenceOutputModule(nn.Module):
    def __init__(
        self,
        output_object: "ComputedSequenceOutputInfo",
        output_name: str,
        feature_dimensionalities_and_types: Dict[str, "FeatureExtractorInfo"],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_tokens = len(output_object.vocab)
        self.input_dimensions = feature_dimensionalities_and_types
        self.embedding_dim = output_object.embedding_dim
        self.max_length = output_object.computed_max_length
        self.output_name = output_name
        self.output_model_config = output_object.output_config.model_config
        assert isinstance(self.output_model_config, SequenceOutputModuleConfig)
        self.output_model_init_config = self.output_model_config.model_init_config
        assert isinstance(
            self.output_model_init_config, TransformerSequenceOutputModuleConfig
        )

        dim_feed_forward = parse_dim_feedforward(
            dim_feedforward=self.output_model_init_config.dim_feedforward,
            embedding_dim=self.embedding_dim,
        )

        mask = torch.triu(
            torch.ones(self.max_length, self.max_length) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("mask", mask)

        self.output_transformer = Transformer(
            d_model=self.embedding_dim,
            nhead=self.output_model_init_config.num_heads,
            num_layers=self.output_model_init_config.num_layers,
            dim_feedforward=dim_feed_forward,
            dropout=self.output_model_init_config.dropout,
            norm_first=True,
        )

        self.match_projections = nn.ModuleDict()
        for input_name, feature_extractor_info in self.input_dimensions.items():
            if input_name == self.output_name:
                continue

            match feature_extractor_info.input_type:
                case "sequence":
                    in_embed = feature_extractor_info.input_dimension.width
                case _:
                    in_embed = feature_extractor_info.output_dimension

            in_elements = feature_extractor_info.output_dimension
            cur_projection = MetaSequenceProjection(
                in_total_num_elements=in_elements,
                in_embedding_dim=in_embed,
                target_embedding_dim=self.embedding_dim,
                target_max_length=self.max_length,
                projection_layer_type=self.output_model_config.projection_layer_type,
            )

            self.match_projections[input_name] = cur_projection

        self.head = nn.Linear(self.embedding_dim, self.num_tokens)

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = input[self.output_name]
        out = out.reshape(out.shape[0], self.max_length, self.embedding_dim)

        for input_name, input_tensor in input.items():
            if input_name == self.output_name:
                continue

            cur_projection = self.match_projections[input_name]
            projected = cur_projection(input_tensor=input_tensor, target_tensor=out)
            out = out + projected

        out = self.output_transformer(out, mask=self.mask)

        out = self.head(out)

        return {self.output_name: out}
