from dataclasses import dataclass
from typing import Callable, Dict, TYPE_CHECKING, Sequence

import torch
from aislib.misc_utils import get_logger
from torch import nn

if TYPE_CHECKING:
    from eir.train import DataDimensions

logger = get_logger(__name__)


@dataclass
class LinearModelConfig:
    input_name: str
    data_dimensions: "DataDimensions"
    target_cat_columns: Sequence[str]
    target_con_columns: Sequence[str]


class LinearModel(nn.Module):
    def __init__(self, model_config: LinearModelConfig):
        super().__init__()

        self.model_config = model_config

        self.fc_1 = nn.Linear(self.fc_1_in_features, 1)
        self.act = self._get_act()

        self.output_parser = self._get_output_parser()

    @property
    def fc_1_in_features(self) -> int:
        return self.model_config.data_dimensions.num_elements()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.fc_1.weight

    @property
    def num_out_features(self) -> int:
        if self.model_config.target_cat_columns:
            return 2
        return 1

    def _get_act(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.model_config.target_cat_columns:
            logger.info(
                "Using logistic regression model on categorical column: %s.",
                self.model_config.target_cat_columns,
            )
            return nn.Sigmoid()

        # no activation for linear regression
        elif self.model_config.target_con_columns:
            logger.info(
                "Using linear regression model on continuous column: %s.",
                self.model_config.target_con_columns,
            )
            return lambda x: x

        raise ValueError()

    def _get_output_parser(self) -> Callable[[torch.Tensor], Dict[str, torch.Tensor]]:
        def _parse_categorical(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            # we create a 2D output from 1D for compatibility with visualization funcs
            out = torch.cat(((1 - out[:, 0]).unsqueeze(1), out), 1)
            out = {self.model_config.target_cat_columns[0]: out}
            return out

        def _parse_continuous(out: torch.Tensor) -> Dict[str, torch.Tensor]:
            out = {self.model_config.target_con_columns[0]: out}
            return out

        if self.model_config.target_cat_columns:
            return _parse_categorical
        return _parse_continuous

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        genotype = inputs[self.model_config.input_name]
        out = genotype.view(genotype.shape[0], -1)

        tabular = inputs.get("tabular_cl_args", None)
        if tabular is not None:
            out = torch.cat((out, tabular), dim=1)

        out = self.fc_1(out)
        out = self.act(out)

        out = self.output_parser(out)

        return out
