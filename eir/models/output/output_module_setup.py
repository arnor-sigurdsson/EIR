from dataclasses import dataclass
from typing import Union, Literal

from eir.models.output.linear import LinearOutputModuleConfig
from eir.models.output.mlp_residual import ResidualMLPOutputModulelConfig
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

    model_init_config: Union[ResidualMLPOutputModulelConfig, LinearOutputModuleConfig]
    model_type: Literal["mlp_residual", "linear"] = "mlp_residual"


@dataclass
class SequenceOutputModuleConfig:
    model_init_config: Union[TransformerSequenceOutputModuleConfig]
    model_type: Literal["sequence"] = "sequence"
