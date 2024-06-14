from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Sequence, Union

import torch
from torch import nn

from eir.models.layers.mlp_layers import MLPResidualBlock, ResidualMLPConfig
from eir.models.models_utils import (
    calculate_module_dict_outputs,
    construct_multi_branches,
    create_multi_task_blocks_with_first_adaptor_block,
    get_final_layer,
    initialize_modules_from_spec,
    merge_module_dicts,
)

if TYPE_CHECKING:
    from eir.setup.output_setup_modules.tabular_output_setup import (
        al_num_outputs_per_target,
    )


@dataclass
class ResidualMLPOutputModuleConfig(ResidualMLPConfig):
    """
    :param layers:
        Number of residual MLP residual blocks to use in the output module.

    :param fc_task_dim:
        Number of hidden nodes in each MLP residual block.

    :param rb_do:
        Dropout in each MLP residual block.

    :param fc_do:
        Dropout before final layer.

    :param stochastic_depth_p:
        Stochastic depth probability (probability of dropping input)
        for each residual block.

    :param final_layer_type:
        Which type of final layer to use to construct tabular output prediction.
    """

    final_layer_type: Union[Literal["linear"], Literal["mlp_residual"]] = "linear"


class ResidualMLPOutputModule(nn.Module):
    def __init__(
        self,
        model_config: ResidualMLPOutputModuleConfig,
        input_dimension: int,
        num_outputs_per_target: "al_num_outputs_per_target",
    ):
        super().__init__()

        self.model_config = model_config
        self.input_dimension = input_dimension
        self.num_outputs_per_target = num_outputs_per_target

        task_names = tuple(self.num_outputs_per_target.keys())

        task_resblocks_kwargs = {
            "in_features": self.model_config.fc_task_dim,
            "out_features": self.model_config.fc_task_dim,
            "dropout_p": self.model_config.rb_do,
            "stochastic_depth_p": self.model_config.stochastic_depth_p,
            "full_preactivation": False,
        }

        multi_task_branches = create_multi_task_blocks_with_first_adaptor_block(
            num_blocks=self.model_config.layers[0],
            branch_names=task_names,
            block_constructor=MLPResidualBlock,
            block_constructor_kwargs=task_resblocks_kwargs,
            first_layer_kwargs_overload={"in_features": self.input_dimension},
        )

        final_layers = get_default_tabular_output_final_layers(
            fc_task_dim=self.model_config.fc_task_dim,
            dropout_p=self.model_config.fc_do,
            num_outputs_per_target=self.num_outputs_per_target,
            task_names=task_names,
            final_layer_type=self.model_config.final_layer_type,
        )
        self.multi_task_branches = merge_module_dicts(
            (multi_task_branches, *final_layers)
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        output_modules_out = calculate_module_dict_outputs(
            input_=inputs, module_dict=self.multi_task_branches
        )

        return output_modules_out


def get_linear_final_act_spec(in_features: int, dropout_p: float):
    spec = OrderedDict(
        {
            "norm_final": (nn.LayerNorm, {"normalized_shape": in_features}),
            "act_final": (nn.GELU, {}),
            "do_final": (nn.Dropout, {"p": dropout_p}),
        }
    )

    return spec


def get_default_tabular_output_final_layers(
    fc_task_dim: int,
    dropout_p: float,
    num_outputs_per_target: "al_num_outputs_per_target",
    task_names: Union[None, Sequence[str]],
    final_layer_type: Union[Literal["linear"], Literal["mlp_residual"]],
) -> Sequence[nn.ModuleDict]:
    final_layers = []
    if final_layer_type == "linear":
        if task_names is None:
            raise ValueError(
                "Task names are needed when using '%s' as final layer type.",
                final_layer_type,
            )

        final_act_spec = get_linear_final_act_spec(
            in_features=fc_task_dim, dropout_p=dropout_p
        )
        final_act = construct_multi_branches(
            branch_names=task_names,
            branch_factory=initialize_modules_from_spec,
            branch_factory_kwargs={"spec": final_act_spec},
        )
        final_layers.append(final_act)
        final_layer_specific_kwargs = {}

    elif final_layer_type == "mlp_residual":
        final_layer_specific_kwargs = {
            "dropout_p": dropout_p,
            "stochastic_depth_p": 0.0,
        }

    else:
        raise ValueError("Unknown final layer type: '%s'.", final_layer_type)

    final_layer = get_final_layer(
        in_features=fc_task_dim,
        num_outputs_per_target=num_outputs_per_target,
        layer_type=final_layer_type,
        layer_type_specific_kwargs=final_layer_specific_kwargs,
    )
    final_layers.append(final_layer)

    return tuple(final_layers)
