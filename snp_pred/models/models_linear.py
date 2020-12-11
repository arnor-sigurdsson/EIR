from typing import Callable, Dict

import torch
from aislib.misc_utils import get_logger
from torch import nn

from snp_pred.models.models_base import ModelBase

logger = get_logger(__name__)


class LinearModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        total_in_dim = self.fc_1_in_features + self.extra_dim
        self.fc_1 = nn.Linear(total_in_dim, 1)
        self.act = self._get_act()

        self.output_parser = self._get_output_parser()

    @property
    def fc_1_in_features(self) -> int:
        return self.cl_args.target_width * 4

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_1.weight

    def _get_act(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.cl_args.target_cat_columns:
            logger.info(
                "Using logistic regression model on categorical column: %s.",
                self.cl_args.target_cat_columns,
            )
            return nn.Sigmoid()

        # no activation for linear regression
        elif self.cl_args.target_con_columns:
            logger.info(
                "Using linear regression model on continuous column: %s.",
                self.cl_args.target_con_columns,
            )
            return lambda x: x

        raise ValueError()

    def _get_output_parser(self) -> Callable[[torch.Tensor], Dict[str, torch.Tensor]]:
        def _parse_categorical(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            # we create a 2D output from 1D for compatibility with visualization funcs
            out = torch.cat(((1 - out[:, 0]).unsqueeze(1), out), 1)
            out = {self.cl_args.target_cat_columns[0]: out}
            return out

        def _parse_continuous(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = {self.cl_args.target_con_columns[0]: out}
            return out

        if self.cl_args.target_cat_columns:
            return _parse_categorical
        return _parse_continuous

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        genotype = inputs["genotype"]
        out = genotype.view(genotype.shape[0], -1)

        tabular = inputs.get("tabular", None)
        if tabular is not None:
            out = torch.cat((out, tabular), dim=1)

        out = self.fc_1(out)
        out = self.act(out)

        out = self.output_parser(out)

        return out
