import warnings
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Literal, Union

import torch
from torch import nn
from torchsurv.loss.cox import neg_partial_log_likelihood

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
from eir.train_utils.metrics import (
    LogEmptyLossProtocol,
    calculate_prediction_losses,
    log_empty_loss_once,
)

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

al_criteria_dict = dict[str, al_criteria]

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
    type[nn.CrossEntropyLoss]
    | type[nn.MSELoss]
    | type[nn.L1Loss]
    | type[nn.SmoothL1Loss]
    | type[nn.PoissonNLLLoss]
    | type[nn.HuberLoss]
)


def get_criteria(outputs_as_dict: "al_output_objects_as_dict") -> al_criteria_dict:
    criteria_dict: al_criteria_dict = {}
    log_empty_once = log_empty_loss_once()

    for output_name, output_object in outputs_as_dict.items():
        match output_object:
            case ComputedTabularOutputInfo():
                con_loss_name = _parse_loss_name(
                    output_config=output_object.output_config,
                    column_type="con",
                )

                criterion_con = get_supervised_criterion(
                    column_type_="con",
                    loss_name=con_loss_name,
                    reduction="none",
                )

                label_smoothing = _get_label_smoothing(
                    output_config=output_object.output_config,
                    column_type="cat",
                )
                cat_loss_name = _get_cat_loss_name(
                    output_config=output_object.output_config
                )
                criterion_cat = get_supervised_criterion(
                    column_type_="cat",
                    loss_name=cat_loss_name,
                    cat_label_smoothing_=label_smoothing,
                    reduction="none",
                )
                cat_loss_callable = (
                    loop_ce_loss
                    if cat_loss_name == "CrossEntropyLoss"
                    else vectorized_bce_loss
                )
                assert isinstance(
                    criterion_cat, nn.CrossEntropyLoss | nn.BCEWithLogitsLoss
                )

                loss_func = partial(
                    tabular_output_loss,
                    con_columns=output_object.target_columns["con"],
                    cat_columns=output_object.target_columns["cat"],
                    output_name=output_name,
                    log_empty_loss_callable=log_empty_once,
                    con_loss_func=criterion_con,
                    cat_loss_func=criterion_cat,
                    cat_loss_callable=cat_loss_callable,
                )

                criteria_dict[output_name] = loss_func

            case ComputedSequenceOutputInfo():
                pad_token = getattr(output_object.tokenizer, "pad_token", "<pad>")
                pad_idx = output_object.vocab[pad_token]

                criterion = partial(
                    _sequence_cat_loss,
                    cat_loss_func=nn.CrossEntropyLoss(ignore_index=pad_idx),
                )

                criterion_callable = _get_extract_and_call_criterion(
                    output_name=output_name,
                    loss_callable=criterion,
                )

                criteria_dict[output_name] = criterion_callable

            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                criterion = partial(
                    _calc_con_loss,
                    loss_func=nn.MSELoss(),
                    nan_handling=NaNHandling.MASK,
                )

                criterion_callable = _get_extract_and_call_criterion(
                    output_name=output_name,
                    loss_callable=criterion,
                )

                criteria_dict[output_name] = criterion_callable

            case ComputedSurvivalOutputInfo():
                output_type_info = output_object.output_config.output_type_info
                assert isinstance(output_type_info, schemas.SurvivalOutputTypeConfig)

                surv_loss_name = output_type_info.loss_function
                time_column = output_type_info.time_column

                criterion_surv = create_survival_criterion(
                    loss_name=surv_loss_name,
                    time_column=time_column,
                )
                criteria_dict[output_name] = criterion_surv

    return criteria_dict


def _get_extract_and_call_criterion(
    output_name: str,
    loss_callable: Callable,
) -> Callable:
    def _compute_loss(
        input: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        input_tensor = input[output_name]
        target_tensor = target[output_name]
        loss = loss_callable(input=input_tensor, target=target_tensor)
        return {output_name: loss}

    return _compute_loss


def tabular_output_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    con_columns: list[str],
    cat_columns: list[str],
    output_name: str,
    log_empty_loss_callable: LogEmptyLossProtocol,
    con_loss_func: Callable,
    cat_loss_func: nn.CrossEntropyLoss | nn.BCEWithLogitsLoss,
    cat_loss_callable: Callable,
) -> dict[str, torch.Tensor]:
    con_preds = {k: predictions[k] for k in con_columns}
    cat_preds = {k: predictions[k] for k in cat_columns}
    con_targets = {k: targets[k] for k in con_columns}
    cat_targets = {k: targets[k] for k in cat_columns}

    con_losses = (
        vectorized_con_loss(
            predictions=con_preds,
            targets=con_targets,
            loss_func=con_loss_func,
        )
        if con_columns
        else {}
    )

    cat_losses = (
        cat_loss_callable(
            predictions=cat_preds,
            targets=cat_targets,
            loss_func=cat_loss_func,
        )
        if cat_columns
        else {}
    )

    all_losses = {}
    for name, loss in con_losses.items():
        if torch.isnan(loss):
            log_empty_loss_callable(output_name=output_name, output_head_name=name)
        all_losses[name] = loss
    for name, loss in cat_losses.items():
        if torch.isnan(loss):
            log_empty_loss_callable(output_name=output_name, output_head_name=name)
        all_losses[name] = loss

    return all_losses


def vectorized_con_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_func: Callable | None = None,
) -> dict[str, torch.Tensor]:
    if loss_func is None:
        loss_func = nn.MSELoss(reduction="none")

    predictions_stacked = torch.stack(list(predictions.values()), dim=1).squeeze()
    targets_stacked = torch.stack(list(targets.values()), dim=1).squeeze()

    valid_mask = ~torch.isnan(targets_stacked)

    predictions_masked = predictions_stacked * valid_mask
    targets_masked = targets_stacked.clone()
    targets_masked[~valid_mask] = 0

    losses = loss_func(predictions_masked, targets_masked)
    target_means = losses.sum(dim=0) / (valid_mask.sum(dim=0) + 1e-8)

    if len(target_means.shape) == 0:
        target_means = target_means.unsqueeze(0)

    return dict(zip(predictions.keys(), target_means, strict=False))


def vectorized_bce_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_func: Callable | None = None,
) -> dict[str, torch.Tensor]:
    if loss_func is None:
        loss_func = nn.BCEWithLogitsLoss(reduction="none")

    predictions_stacked = torch.stack(list(predictions.values()), dim=1).squeeze()
    targets_stacked = torch.stack(list(targets.values()), dim=1).squeeze().float()

    valid_mask = targets_stacked != -1

    targets_masked = targets_stacked.clone()
    targets_masked[~valid_mask] = 0

    losses = loss_func(predictions_stacked, targets_masked)
    target_means = losses.sum(dim=0) / (valid_mask.sum(dim=0) + 1e-8)

    if len(target_means.shape) == 0:
        target_means = target_means.unsqueeze(0)

    return dict(zip(predictions.keys(), target_means, strict=False))


def loop_ce_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_func: Callable | None = None,
) -> dict[str, torch.Tensor]:
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss(reduction="none")

    target_losses = {}

    for name in predictions:
        valid_mask = targets[name] != -1

        pred = predictions[name]
        target = targets[name]

        masked_target = target.clone()
        masked_target[~valid_mask] = -100

        losses = loss_func(pred, masked_target)

        target_losses[name] = (
            losses[valid_mask].mean() if valid_mask.any() else losses.new_zeros(())
        )

    return target_losses


def build_loss_dict() -> dict[str, list[str]]:
    loss_dict = {
        "cat": [
            "CrossEntropyLoss",
            "BCEWithLogitsLoss",
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
    column_type_: str,
    loss_name: str,
    cat_label_smoothing_: float = 0.0,
    reduction: str = "none",
) -> nn.CrossEntropyLoss | Callable:
    loss_dict = build_loss_dict()

    match column_type_, loss_name:
        case "con", _:
            assert loss_name in loss_dict["con"]

            loss_module = getattr(nn, loss_name)
            return partial(
                _calc_con_loss,
                loss_func=loss_module(reduction=reduction),
            )

        case "cat", "CrossEntropyLoss":
            return nn.CrossEntropyLoss(
                label_smoothing=cat_label_smoothing_,
                reduction=reduction,
            )

        case "cat", _:
            assert loss_name in loss_dict["cat"]
            loss_module = getattr(nn, loss_name)
            return loss_module(reduction=reduction)

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


def _get_cat_loss_name(
    output_config: schemas.OutputConfig,
) -> "al_cat_loss_names":
    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, schemas.TabularOutputTypeConfig)
    return output_type_info.cat_loss_name


class NaNHandling(Enum):
    MASK = "mask"
    RAISE = "raise"
    NONE = "none"


def _calc_con_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    loss_func: nn.MSELoss | nn.L1Loss | nn.PoissonNLLLoss,
    nan_handling: NaNHandling = NaNHandling.NONE,
) -> torch.Tensor:
    if nan_handling != NaNHandling.NONE:
        has_nan = torch.isnan(target).any()
        if has_nan and nan_handling == NaNHandling.RAISE:
            raise ValueError(
                f"NaN values found in target tensor with NaNHandling.RAISE specified. "
                f"NaN count: {torch.isnan(target).sum().item()}"
            )
        if nan_handling == NaNHandling.MASK:
            mask = ~torch.isnan(target).squeeze()
            input_masked = input.squeeze()[mask]
            target_masked = target.squeeze()[mask]

            match loss_func:
                case nn.PoissonNLLLoss():
                    loss = loss_func(log_input=input_masked, target=target_masked)
                case _:
                    loss = loss_func(input=input_masked, target=target_masked)

            return loss

    match loss_func:
        case nn.PoissonNLLLoss():
            return loss_func(log_input=input.squeeze(), target=target.squeeze())
        case _:
            return loss_func(input=input.squeeze(), target=target.squeeze())


def get_loss_callable(
    criteria: al_criteria_dict,
) -> Callable:
    single_task_loss_func = partial(
        calculate_prediction_losses,
        criteria=criteria,
    )
    return single_task_loss_func


def _sequence_cat_loss(
    input,
    target,
    cat_loss_func: nn.CrossEntropyLoss,
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
        return _discrete_survival_nll_loss(log_hazards=input, time=time, event=target)
    if loss_func == "CoxPHLoss":
        return _cox_ph_loss(risk_scores=input, time=time, event=target)
    raise ValueError(f"Unsupported loss function: {loss_func}")


def _discrete_survival_nll_loss(
    log_hazards: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    valid_mask = (time != -1) & (event != -1)

    # Early return if no valid samples
    if not valid_mask.any():
        return log_hazards.new_zeros((), requires_grad=True)

    masked_log_hazards = log_hazards[valid_mask]
    masked_time = time[valid_mask]
    masked_event = event[valid_mask]

    # Convert log-hazards to hazards
    hazards = torch.sigmoid(masked_log_hazards)

    # Calculate log(survival probability) for each interval
    log_surv = torch.log(1 - hazards + 1e-8)

    # Calculate cumulative log(survival probability)
    cum_log_surv = torch.cumsum(log_surv, dim=1)

    # Select the relevant timepoints
    idx = torch.arange(masked_log_hazards.shape[0])
    relevant_log_surv = cum_log_surv[idx, masked_time]
    relevant_hazards = hazards[idx, masked_time]

    # Calculate log-likelihood
    uncensored_ll = torch.log(relevant_hazards + 1e-8) + relevant_log_surv
    censored_ll = relevant_log_surv

    # Combine censored and uncensored log-likelihoods
    log_likelihood = masked_event * uncensored_ll + (1 - masked_event) * censored_ll

    # Return negative average log-likelihood
    return -torch.mean(log_likelihood)


def _cox_ph_loss(
    risk_scores: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ties_method: str = "efron",
) -> torch.Tensor:
    valid_mask = (time != -1) & (event != -1)

    if not valid_mask.any():
        return risk_scores.new_zeros((), requires_grad=True)

    masked_risk_scores = risk_scores[valid_mask]
    masked_time = time[valid_mask]
    masked_event = event[valid_mask].bool()

    if torch.sum(masked_event) == 0:
        return risk_scores.new_zeros((), requires_grad=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Ties in event time detected*")
        loss = neg_partial_log_likelihood(
            log_hz=masked_risk_scores,
            event=masked_event,
            time=masked_time,
            ties_method=ties_method,
            reduction="mean",
            checks=True,
        )

    return loss


def create_survival_criterion(
    loss_name: Literal["NegativeLogLikelihood", "CoxPHLoss"],
    time_column: str,
) -> Callable:
    def survival_criterion(
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        time_tensor = targets[time_column].to(dtype=torch.long)

        losses = {}
        for name, pred in predictions.items():
            if name != time_column:
                losses[name] = _survival_loss(
                    input=pred,
                    time=time_tensor,
                    target=targets[name],
                    loss_func=loss_name,
                )
        return losses

    return survival_criterion
