from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Sequence, Type, Union

import torch
from torch import nn

from eir.setup import schemas
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
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

al_criteria_dict = dict[str, dict[str, al_criteria]]

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

            case ComputedSurvivalOutputInfo():
                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, schemas.SurvivalOutputTypeConfig)
                surv_loss_name = output_type_info.loss_function
                event_name = output_type_info.event_column
                criterion = get_survival_criterion(loss_name=surv_loss_name)
                criteria_dict[output_name][event_name] = criterion

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


def get_loss_callable(
    criteria: al_criteria_dict,
    survival_links: dict[str, dict[str, str]],
) -> Callable:
    log_empty_callable = log_empty_loss_once()
    single_task_loss_func = partial(
        calculate_prediction_losses,
        criteria=criteria,
        log_empty_loss_callable=log_empty_callable,
        survival_links=survival_links,
    )
    return single_task_loss_func


def build_survival_links_for_criteria(
    output_configs: Sequence[schemas.OutputConfig],
) -> dict[str, dict[str, str]]:

    survival_links = {}
    for output_config in output_configs:
        output_type_info = output_config.output_type_info
        if not isinstance(output_type_info, schemas.SurvivalOutputTypeConfig):
            continue

        output_name = output_config.output_info.output_name
        event_column = output_type_info.event_column
        duration_column = output_type_info.time_column

        survival_links[output_name] = {
            "duration": duration_column,
            "event": event_column,
        }

    return survival_links


def _sequence_cat_loss(
    input, target, cat_loss_func: nn.CrossEntropyLoss
) -> torch.Tensor:
    loss = cat_loss_func(input=input.transpose(2, 1), target=target)
    return loss


def _survival_loss(
    input: torch.Tensor,
    time: torch.Tensor,
    target: torch.Tensor,
    loss_func: Literal["NegativeLogLikelihood", "CoxPHLoss"] = "NegativeLogLikelihood",
) -> torch.Tensor:

    if loss_func == "NegativeLogLikelihood":
        return _negative_log_likelihood_loss(log_hazards=input, time=time, event=target)
    elif loss_func == "CoxPHLoss":
        return _cox_ph_loss(risk_scores=input, time=time, event=target)
    else:
        raise ValueError(f"Unsupported loss function: {loss_func}")


def _negative_log_likelihood_loss(
    log_hazards: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    # Convert log-hazards to hazards
    hazards = torch.sigmoid(log_hazards)

    # Calculate log(survival probability) for each interval
    log_surv = torch.log(1 - hazards + 1e-8)

    # Calculate cumulative log(survival probability)
    cum_log_surv = torch.cumsum(log_surv, dim=1)

    # Select the relevant timepoints
    idx = torch.arange(log_hazards.shape[0], device=log_hazards.device)
    relevant_log_surv = cum_log_surv[idx, time]
    relevant_hazards = hazards[idx, time]

    # Calculate log-likelihood
    uncensored_ll = torch.log(relevant_hazards + 1e-8) + relevant_log_surv
    censored_ll = relevant_log_surv

    # Combine censored and uncensored log-likelihoods
    log_likelihood = event * uncensored_ll + (1 - event) * censored_ll

    # Return negative average log-likelihood
    return -torch.mean(log_likelihood)


def _cox_ph_loss(
    risk_scores: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ties_method: str = "efron",
) -> torch.Tensor:
    if torch.sum(event) == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    sorted_time, indices = torch.sort(time)
    sorted_risk_scores = risk_scores[indices]
    sorted_event = event[indices]

    if ties_method == "breslow":
        log_risk = torch.logcumsumexp(sorted_risk_scores.flip(0), dim=0).flip(0)
        loss = sorted_risk_scores - log_risk
        loss = loss[sorted_event == 1]
        return -torch.nanmean(loss)

    elif ties_method == "efron":
        unique_times = torch.unique(sorted_time)

        # Get risk sets and event sets for each unique time
        log_likelihood = torch.tensor(0.0, device=risk_scores.device)

        for t in unique_times:
            # Events at this time
            events_at_t = (sorted_time == t) & (sorted_event == 1)
            if not torch.any(events_at_t):
                continue

            # Risk set (samples still at risk at time t)
            risk_set = sorted_time >= t

            # Calculate Efron's correction for ties
            n_events = events_at_t.sum()
            tied_scores = sorted_risk_scores[events_at_t]
            risk_scores_exp = torch.exp(sorted_risk_scores)
            risk_set_scores = risk_scores_exp[risk_set]
            tied_scores_exp = risk_scores_exp[events_at_t]

            for j in range(n_events):
                factor = j / n_events
                log_likelihood += tied_scores[j].squeeze() - torch.log(
                    torch.sum(risk_set_scores) - factor * torch.sum(tied_scores_exp)
                )

        n_events_total = torch.sum(event)
        return -log_likelihood / n_events_total

    else:
        raise ValueError(f"Unsupported ties method: {ties_method}")


def get_survival_criterion(
    loss_name: Literal["NegativeLogLikelihood", "CoxPHLoss"],
) -> Callable:
    return partial(
        _survival_loss,
        loss_func=loss_name,
    )
