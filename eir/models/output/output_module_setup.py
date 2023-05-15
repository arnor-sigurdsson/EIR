from dataclasses import dataclass
from typing import Union, Literal

from eir.models.output.mlp_residual import ResidualMLPOutputModelConfig
from eir.models.output.linear import LinearOutputModelConfig


@dataclass
class OutputModuleConfig:
    """
    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param model_type:
         Which type of image model to use."""

    model_init_config: Union[ResidualMLPOutputModelConfig, LinearOutputModelConfig]
    model_type: Literal["mlp_residual", "linear"] = "mlp_residual"
