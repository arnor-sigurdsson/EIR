from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Type, Union

import torch
from torch import nn

from eir.setup import schemas
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.train_utils.metrics import calculate_prediction_losses, log_empty_loss_once

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.setup.schema_modules.output_schemas_tabular import (
        al_cat_loss_names,
        al_con_loss_names,
    )

al_cat_losses = nn.CrossEntropyLoss
al_con_losses = (
    nn.MSELoss | nn.L1Loss | nn.SmoothL1Loss | nn.PoissonNLLLoss | nn.HuberLoss
)

al_criteria = al_con_losses | al_cat_losses | Callable

al_criteria_dict = Dict[str, Dict[str, al_criteria]]

al_losses = (
    nn.CrossEntropyLoss
    | nn.MSELoss
    | nn.L1Loss
    | nn.SmoothL1Loss
    | nn.PoissonNLLLoss
    | nn.HuberLoss
    | Callable
)


al_losses_classes = (
    Type[nn.CrossEntropyLoss]
    | Type[nn.MSELoss]
    | Type[nn.L1Loss]
    | Type[nn.SmoothL1Loss]
    | Type[nn.PoissonNLLLoss]
    | Type[nn.HuberLoss]
)


def get_criteria(outputs_as_dict: "al_output_objects_as_dict") -> al_criteria_dict:
    criteria_dict: al_criteria_dict = {}

    for output_name, output_object in outputs_as_dict.items():
        if output_name not in criteria_dict:
            criteria_dict[output_name] = {}

        match output_object:
            case ComputedTabularOutputInfo():
                target_col_iter = output_object.target_columns.items()
                for column_type, columns_of_type in target_col_iter:
                    for column_name in columns_of_type:
                        label_smoothing = _get_label_smoothing(
                            output_config=output_object.output_config,
                            column_type=column_type,
                        )

                        loss_name = _parse_loss_name(
                            output_config=output_object.output_config,
                            column_type=column_type,
                        )

                        criterion = get_supervised_criterion(
                            column_type_=column_type,
                            loss_name=loss_name,
                            cat_label_smoothing_=label_smoothing,
                        )

                        criteria_dict[output_name][column_name] = criterion

            case ComputedSequenceOutputInfo():
                pad_token = getattr(output_object.tokenizer, "pad_token", "<pad>")
                pad_idx = output_object.vocab[pad_token]

                criterion = partial(
                    _sequence_cat_loss,
                    cat_loss_func=nn.CrossEntropyLoss(ignore_index=pad_idx),
                )

                criteria_dict[output_name][output_name] = criterion

            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                criterion = partial(_calc_con_loss, loss_func=nn.MSELoss())
                criteria_dict[output_name][output_name] = criterion

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
            "HuberLoss",
        ],
    }

    return loss_dict


def get_supervised_criterion(
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

    raise ValueError()


def _parse_loss_name(
    output_config: schemas.OutputConfig, column_type: str
) -> Union["al_cat_loss_names", "al_con_loss_names"]:
    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, TabularOutputTypeConfig)
    match column_type:
        case "cat":
            return output_type_info.cat_loss_name
        case "con":
            return output_type_info.con_loss_name
        case _:
            raise ValueError(f"Unknown column type: {column_type}")


def _get_label_smoothing(
    output_config: schemas.OutputConfig,
    column_type: str,
) -> float:
    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, TabularOutputTypeConfig)
    match column_type:
        case "con":
            return 0.0
        case "cat":
            return output_type_info.cat_label_smoothing
        case _:
            raise ValueError(f"Unknown column type: {column_type}")


def _calc_con_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    loss_func: al_con_losses,
) -> torch.Tensor:
    match loss_func:
        case nn.PoissonNLLLoss():
            return loss_func(log_input=input.squeeze(), target=target.squeeze())
        case _:
            return loss_func(input=input.squeeze(), target=target.squeeze())


def get_loss_callable(criteria: al_criteria_dict) -> Callable:
    log_empty_callable = log_empty_loss_once()
    single_task_loss_func = partial(
        calculate_prediction_losses,
        criteria=criteria,
        log_empty_loss_callable=log_empty_callable,
    )
    return single_task_loss_func


def _sequence_cat_loss(
    input, target, cat_loss_func: nn.CrossEntropyLoss
) -> torch.Tensor:
    loss = cat_loss_func(input=input.transpose(2, 1), target=target)
    return loss
