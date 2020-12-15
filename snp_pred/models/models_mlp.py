import torch
from torch import nn

from snp_pred.models.models_base import (
    ModelBase,
)


class MLPModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_0 = nn.Linear(
            self.fc_1_in_features, self.cl_args.fc_repr_dim, bias=False
        )

        self._init_weights()

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_0.weight

    @property
    def num_out_features(self) -> int:
        return self.cl_args.fc_repr_dim

    def _init_weights(self):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.view(input.shape[0], -1)

        out = self.fc_0(out)

        return out
