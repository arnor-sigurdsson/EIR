from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Callable, Sequence, List, TYPE_CHECKING

import torch
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import Swish
from torch import nn

from eir.models.layers import MLPResidualBlock
from eir.models.models_base import (
    create_multi_task_blocks_with_first_adaptor_block,
    construct_multi_branches,
    initialize_modules_from_spec,
    get_final_layer,
    merge_module_dicts,
    calculate_module_dict_outputs,
)

if TYPE_CHECKING:
    from eir.train import al_num_outputs_per_target

al_features = Callable[[Sequence[torch.Tensor]], torch.Tensor]


logger = get_logger(__name__)


def default_fuse_features(features: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple(features), dim=1)


@dataclass
class FusionModelConfig:
    """
    :param layers:
        Number of residual MLP layers to use in for each output predictor after fusing.

    :param fc_task_dim:
        Number of hidden nodes in each MLP residual block.

    :param rb_do:
        Dropout in each MLP residual block.

    :param fc_do:
        Dropout before last FC layer.
    """

    layers: List[int] = field(default_factory=lambda: [2])

    fc_task_dim: int = 32

    rb_do: float = 0.00
    fc_do: float = 0.00


class FusionModel(nn.Module):
    def __init__(
        self,
        model_config: FusionModelConfig,
        num_outputs_per_target: "al_num_outputs_per_target",
        modules_to_fuse: nn.ModuleDict,
        fusion_callable: al_features = default_fuse_features,
    ):
        super().__init__()

        self.model_config = model_config
        self.num_outputs_per_target = num_outputs_per_target
        self.modules_to_fuse = modules_to_fuse
        self.fusion_callable = fusion_callable

        task_names = tuple(self.num_outputs_per_target.keys())

        task_resblocks_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "full_preactivation": False,
        }

        cur_dim = sum(i.num_out_features for i in self.modules_to_fuse.values())

        multi_task_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=task_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=task_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": cur_dim},
        )

        final_act_spec = self.get_final_act_spec(
            in_features=self.model_config.fc_task_dim, dropout_p=self.model_config.fc_do
        )
        final_act = construct_multi_branches(
            branch_names=task_names,
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": final_act_spec},
        )

        final_layer = get_final_layer(
            in_features=self.model_config.fc_task_dim,
            num_outputs_per_target=self.num_outputs_per_target,
        )

        self.multi_task_branches = merge_module_dicts(
            (multi_task_branches, final_act, final_layer)
        )

        self._init_weights()

    @staticmethod
    def get_final_act_spec(in_features: int, dropout_p: float):

        spec = OrderedDict(
            {
                "bn_final": (nn.BatchNorm1d, {"num_features": in_features}),
                "act_final": (Swish, {}),
                "do_final": (nn.Dropout, {"p": dropout_p}),
            }
        )

        return spec

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        out = []
        for module in self.modules_to_fuse.values():
            if hasattr(module, "l1_penalized_weights"):
                weight_flat = torch.flatten(module.l1_penalized_weights)
                out.append(weight_flat)

        return torch.cat(out)

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
