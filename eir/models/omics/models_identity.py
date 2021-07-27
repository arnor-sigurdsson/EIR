from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import torch
from torch import nn

from eir.models.omics.models_locally_connected import flatten_h_w_fortran

if TYPE_CHECKING:
    from eir.setup.input_setup import DataDimensions


@dataclass
class IdentityModelConfig:
    flatten: bool = True
    flatten_shape: str = "c"


class IdentityModel(nn.Module):
    def __init__(
        self, model_config: IdentityModelConfig, data_dimensions: "DataDimensions"
    ):
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions

        self.process_func = get_identity_reshape_func(
            flatten=self.model_config.flatten,
            flatten_shape=self.model_config.flatten_shape,
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.data_dimensions.num_elements()

    @property
    def num_out_features(self) -> int:
        return self.fc_1_in_features

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.process_func(input)
        return out


def get_identity_reshape_func(
    flatten: bool, flatten_shape: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    if not flatten:
        return lambda x: x

    if flatten_shape == "fortran":
        return flatten_h_w_fortran
    else:
        return lambda x: x.view(x.shape[0], -1)
