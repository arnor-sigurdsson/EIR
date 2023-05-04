from functools import partial
from typing import Dict, Union, Callable, Type, Literal, TYPE_CHECKING

import torch
from torch import nn

from eir.data_load import data_utils
from eir.setup import schemas
from eir.train_utils.metrics import calculate_prediction_losses

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict

al_cat_loss_names = Literal["CrossEntropyLoss"]
al_con_loss_names = Literal[
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
]

al_cat_losses = nn.CrossEntropyLoss
al_con_losses = (
    nn.MSELoss | nn.L1Loss | nn.SmoothL1Loss | nn.PoissonNLLLoss | nn.GaussianNLLLoss
)

al_criteria = al_con_losses | al_cat_losses

al_criteria_dict = Dict[str, Dict[str, al_criteria]]

al_losses_classes = [
    Type[nn.CrossEntropyLoss],
    Type[nn.MSELoss],
    Type[nn.L1Loss],
    Type[nn.SmoothL1Loss],
    Type[nn.PoissonNLLLoss],
    Type[nn.GaussianNLLLoss],
]


def get_criteria(outputs_as_dict: "al_output_objects_as_dict") -> al_criteria_dict:
    criteria_dict = {}
    target_columns_gen = data_utils.get_output_info_generator(
        outputs_as_dict=outputs_as_dict
    )

    for output_name, column_type, column_name in target_columns_gen:
        output_config = outputs_as_dict[output_name].output_config
        label_smoothing = _get_label_smoothing(
            output_config=output_config,
            column_type=column_type,
        )

        loss_name = _parse_loss_name(
            output_config=output_config, column_type=column_type
        )

        criterion = get_criterion(
            column_type_=column_type,
            loss_name=loss_name,
            cat_label_smoothing_=label_smoothing,
        )

        if output_name not in criteria_dict:
            criteria_dict[output_name] = {}
        criteria_dict[output_name][column_name] = criterion

    return criteria_dict


def build_loss_dict() -> dict[str, list[str]]:
    loss_dict = {
        "cat": [
            "CrossEntropyLoss",
        ],
        "con": [
            "MSELoss",
            "L1Loss",
            "SmoothL1Loss",
            "PoissonNLLLoss",
            "GaussianNLLLoss",
        ],
    }

    return loss_dict


def get_criterion(
    column_type_: str, loss_name: str, cat_label_smoothing_: float = 0.0
) -> Union[nn.CrossEntropyLoss, Callable]:
    loss_dict = build_loss_dict()

    match column_type_, loss_name:
        case "con", _:
            assert loss_name in loss_dict["con"]

            loss_module = getattr(nn, loss_name)
            return partial(_calc_con_loss, loss_func=loss_module())

        case "cat", "CrossEntropyLoss":
            return nn.CrossEntropyLoss(label_smoothing=cat_label_smoothing_)

        case "cat", _:
            assert loss_name in loss_dict["cat"]
            loss_module = getattr(nn, loss_name)
            return loss_module()


def _parse_loss_name(
    output_config: schemas.OutputConfig, column_type: str
) -> al_cat_loss_names | al_con_loss_names:
    match column_type:
        case "cat":
            return output_config.output_type_info.cat_loss_name
        case "con":
            return output_config.output_type_info.con_loss_name
        case _:
            raise ValueError(f"Unknown column type: {column_type}")


def _get_label_smoothing(
    output_config: schemas.OutputConfig,
    column_type: str,
) -> float:
    match column_type:
        case "con":
            return 0.0
        case "cat":
            return output_config.output_type_info.cat_label_smoothing
        case _:
            raise ValueError(f"Unknown column type: {column_type}")


def _calc_con_loss(input: torch.Tensor, target: torch.Tensor, loss_func: al_con_losses):
    return loss_func(input=input.squeeze(), target=target.squeeze())


def get_loss_callable(criteria: al_criteria_dict):
    single_task_loss_func = partial(calculate_prediction_losses, criteria=criteria)
    return single_task_loss_func
