from dataclasses import dataclass
from typing import Literal, Union

from eir.models.output.tabular.linear import LinearOutputModuleConfig
from eir.models.output.tabular.mlp_residual import ResidualMLPOutputModuleConfig


@dataclass
class TabularOutputModuleConfig:
    """
    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param model_type:
         Which type of image model to use."""

    model_init_config: Union[ResidualMLPOutputModuleConfig, LinearOutputModuleConfig]
    model_type: Literal["mlp_residual", "linear"] = "mlp_residual"
