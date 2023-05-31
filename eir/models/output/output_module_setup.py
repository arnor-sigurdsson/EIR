from dataclasses import dataclass
from typing import Union, Literal

from eir.models.output.linear import LinearOutputModuleConfig
from eir.models.output.mlp_residual import ResidualMLPOutputModuleConfig
from eir.models.output.sequence.sequence_output_modules import (
    TransformerSequenceOutputModuleConfig,
)


@dataclass
class TabularOutputModuleConfig:
    """
    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param model_type:
         Which type of image model to use."""

    model_init_config: Union[ResidualMLPOutputModuleConfig, LinearOutputModuleConfig]
    model_type: Literal["mlp_residual", "linear"] = "mlp_residual"


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
