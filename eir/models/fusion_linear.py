import dataclasses
from typing import Dict, TYPE_CHECKING

import torch
from torch import nn

from eir.models.fusion import default_fuse_features, al_features
from eir.models.models_base import (
    get_final_layer,
    calculate_module_dict_outputs,
)

if TYPE_CHECKING:
    from eir.train import al_num_outputs_per_target


@dataclasses.dataclass
class LinearFusionModelConfig:
    """
    :param l1:
        L1 regularisation to apply to the first and only layer in the model.
    """

    l1: float = 0.0


class LinearFusionModel(nn.Module):
    def __init__(
        self,
        model_config: LinearFusionModelConfig,
        num_outputs_per_target: "al_num_outputs_per_target",
        modules_to_fuse: nn.ModuleDict,
        fusion_callable: al_features = default_fuse_features,
    ):
        super().__init__()

        self.model_config = model_config
        self.num_outputs_per_target = num_outputs_per_target
        self.modules_to_fuse = modules_to_fuse
        self.fusion_callable = fusion_callable

        cur_dim = sum(i.num_out_features for i in self.modules_to_fuse.values())

        self.multi_task_branches = get_final_layer(
            in_features=cur_dim, num_outputs_per_target=self.num_outputs_per_target
        )

        self._init_weights()

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        l1_parameters = []

        for module in self.multi_task_branches.values():
            weight_flat = torch.flatten(module.fc_final.weight)
            l1_parameters.append(weight_flat)

        return torch.cat(l1_parameters)

    def _init_weights(self):
        pass

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        out = {}
        for module_name, module_input in inputs.items():
            module = self.modules_to_fuse[module_name]
            out[module] = module(module_input)

        fused_features = default_fuse_features(tuple(out.values()))

        out = calculate_module_dict_outputs(
            input_=fused_features, module_dict=self.multi_task_branches
        )

        return out
