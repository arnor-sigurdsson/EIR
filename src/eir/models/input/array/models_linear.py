from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions


@dataclass
class LinearModelConfig:
    """
    :param fc_repr_dim:
        Number of output nodes in the first and only hidden layer.
    :param l1:
        L1 regularisation to apply to the first layer.
    """

    fc_repr_dim: int = 32
    l1: float = 0.0


class LinearModel(nn.Module):
    def __init__(
        self, model_config: LinearModelConfig, data_dimensions: "DataDimensions"
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions

        self.fc_0 = nn.Linear(
            self.fc_1_in_features,
            self.model_config.fc_repr_dim,
            bias=True,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    @property
    def num_out_features(self) -> int:
        return self.model_config.fc_repr_dim

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.view(input.shape[0], -1)

        out = self.fc_0(out)

        return out
