import csv
from collections.abc import Callable, Generator, Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from statistics import mean
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Protocol,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from aislib.misc_utils import ensure_path_exists
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, label_binarize
from torch.linalg import vector_norm
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from eir.data_load.data_utils import get_output_info_generator
from eir.setup.schemas import (
    OutputConfig,
    SurvivalOutputTypeConfig,
    TabularOutputTypeConfig,
)
from eir.target_setup.target_label_setup import MissingTargetsInfo
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.label_setup import (  # noqa: F401
        al_label_transformers,
        al_label_transformers_object,
        al_target_columns,
    )
    from eir.models.input.omics.omics_models import al_omics_models  # noqa: F401
    from eir.models.meta.meta_utils import FeatureExtractorProtocolWithL1
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.train import Experiment, al_criteria_dict  # noqa: F401
    from eir.train_utils.step_logic import al_training_labels_target
    from eir.train_utils.train_handlers import HandlerConfig

# aliases
# output_name -> target_name -> metric name: value
al_step_metric_dict = dict[str, dict[str, dict[str, float]]]

logger = get_logger(name=__name__)


class MetricFunctionProtocol(Protocol):
    def __call__(
        self,
        outputs: np.ndarray,
        labels: np.ndarray,
        column_name: str,
        output_name: str,
    ) -> float: ...


class SurvivalMetricFunctionProtocol(Protocol):
    def __call__(
        self,
        outputs: np.ndarray,
        labels: np.ndarray,
        times: np.ndarray,
        event_name: str,
        time_name: str,
        output_name: str,
    ) -> float: ...


class AverageMetricFunctionProtocol(Protocol):
    def __call__(
        self,
        metric_dict: al_step_metric_dict,
        output_name: str,
        column_name: str,
    ) -> float: ...


al_averaging_functions_dict = dict[str, AverageMetricFunctionProtocol]
al_metric_record_dict = dict[
    str,
    Union[
        tuple["MetricRecord", ...], "al_averaging_functions_dict", "GeneralMetricInfo"
    ],
]

al_cat_metric_choices = Sequence[
    Literal[
        "mcc",
        "acc",
        "roc-auc-macro",
        "ap-macro",
        "f1-macro",
        "precision-macro",
        "recall-macro",
        "cohen-kappa",
    ]
]
al_con_metric_choices = Sequence[
    Literal[
        "r2",
        "pcc",
        "loss",
        "rmse",
        "mae",
        "mape",
        "explained-variance",
    ]
]

all_survival_metric_choices = Sequence[
    Literal[
        "c-index",
        "ibs",
        "td-cindex",
        "td-auc",
    ]
]


@dataclass()
class MetricRecord:
    name: str
    function: MetricFunctionProtocol | SurvivalMetricFunctionProtocol
    output_type: str = "supervised"
    only_val: bool = True
    minimize_goal: bool = False


def calculate_batch_metrics(
    outputs_as_dict: "al_output_objects_as_dict",
    outputs: dict[str, dict[str, torch.Tensor]],
    labels: "al_training_labels_target",
    mode: str,
    metric_record_dict: al_metric_record_dict,
) -> al_step_metric_dict:
    assert mode in ["val", "train"]

    general_info = metric_record_dict["general_metric_info"]
    assert isinstance(general_info, GeneralMetricInfo)
    if mode == "train" and general_info.all_are_val_only:
        return general_info.base_metric_structure

    target_columns_gen = get_output_info_generator(outputs_as_dict=outputs_as_dict)

    master_metric_dict: al_step_metric_dict = {}

    for output_name, output_target_type, target_name in target_columns_gen:
        cur_metric_dict: dict[str, float] = {}

        if output_name not in master_metric_dict:
            master_metric_dict[output_name] = {}

        if output_target_type == "general":
            master_metric_dict[output_name][target_name] = cur_metric_dict

        elif output_target_type in ("con", "cat"):
            cur_output_object = outputs_as_dict[output_name]
            cur_output_type = cur_output_object.output_config.output_info.output_type
            assert cur_output_type == "tabular"

            al_record = tuple[MetricRecord, ...]
            cur_records = metric_record_dict[output_target_type]
            assert isinstance(cur_records, tuple)
            cur_metric_records: al_record = cur_records

            cur_outputs = outputs[output_name][target_name]
            cur_target_labels = labels[output_name][target_name]

            filtered = filter_tabular_missing_targets(
                outputs=cur_outputs,
                target_labels=cur_target_labels,
                ids=[],
                target_type=output_target_type,
            )

            cur_outputs_np = general_torch_to_numpy(tensor=filtered.model_outputs)
            cur_labels_np = general_torch_to_numpy(tensor=filtered.target_labels)
            if output_target_type == "cat":
                cur_labels_np = cur_labels_np.astype(int)

            for metric_record in cur_metric_records:
                if metric_record.output_type != "supervised":
                    continue

                if metric_record.only_val and mode == "train":
                    continue

                cur_key = f"{output_name}_{target_name}_{metric_record.name}"
                cur_fun = cast(MetricFunctionProtocol, metric_record.function)
                cur_value = cur_fun(
                    outputs=cur_outputs_np,
                    labels=cur_labels_np,
                    column_name=target_name,
                    output_name=output_name,
                )

                cur_metric_dict[cur_key] = cur_value

            master_metric_dict[output_name][target_name] = cur_metric_dict

        elif output_target_type == "survival":
            cur_output_object = outputs_as_dict[output_name]
            cur_output_type = cur_output_object.output_config.output_info.output_type
            assert cur_output_type == "survival"

            cur_records = metric_record_dict[output_target_type]
            assert isinstance(cur_records, tuple)
            cur_metric_records = cur_records

            cur_outputs = outputs[output_name][target_name]
            cur_events = labels[output_name][target_name]

            cur_oti = cur_output_object.output_config.output_type_info
            assert isinstance(cur_oti, SurvivalOutputTypeConfig)

            cur_time_name = cur_oti.time_column
            cur_times = labels[output_name][cur_time_name]

            filtered = filter_survival_missing_targets(
                model_outputs=cur_outputs,
                events=cur_events,
                times=cur_times,
                cur_ids=[],
            )

            cur_outputs_np = general_torch_to_numpy(tensor=filtered.model_outputs)
            cur_labels_np = general_torch_to_numpy(tensor=filtered.events)
            cur_labels_np = cur_labels_np.astype(int)
            cur_times_np = general_torch_to_numpy(tensor=filtered.times)

            for metric_record in cur_metric_records:
                if metric_record.output_type != "survival":
                    continue

                if metric_record.only_val and mode == "train":
                    continue

                cur_key = f"{output_name}_{target_name}_{metric_record.name}"
                cur_fun = cast(SurvivalMetricFunctionProtocol, metric_record.function)
                cur_value = cur_fun(
                    outputs=cur_outputs_np,
                    labels=cur_labels_np,
                    times=cur_times_np,
                    event_name=target_name,
                    output_name=output_name,
                    time_name=cur_time_name,
                )

                cur_metric_dict[cur_key] = cur_value

            master_metric_dict[output_name][target_name] = cur_metric_dict
        else:
            raise NotImplementedError()

    return master_metric_dict


def add_loss_to_metrics(
    outputs_as_dict: "al_output_objects_as_dict",
    losses: dict[str, dict[str, torch.Tensor]],
    metric_dict: al_step_metric_dict,
) -> al_step_metric_dict:
    target_columns_gen = get_output_info_generator(outputs_as_dict=outputs_as_dict)
    metric_dict_copy = copy(metric_dict)

    for output_name, _column_type, column_name in target_columns_gen:
        cur_metric_dict = metric_dict_copy[output_name][column_name]
        cur_key = f"{output_name}_{column_name}_loss"
        cur_metric_dict[cur_key] = losses[output_name][column_name].item()

    return metric_dict_copy


def add_multi_task_average_metrics(
    batch_metrics_dict: al_step_metric_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    loss: float,
    performance_average_functions: Optional["al_averaging_functions_dict"],
) -> al_step_metric_dict:
    average = {
        "average": {
            "loss-average": loss,
        }
    }

    if performance_average_functions is not None:
        average_performance = average_performances_across_tasks(
            metric_dict=batch_metrics_dict,
            outputs_as_dict=outputs_as_dict,
            performance_calculation_functions=performance_average_functions,
        )
        average["average"]["perf-average"] = average_performance

    batch_metrics_dict["average"] = average

    return batch_metrics_dict


def average_performances_across_tasks(
    metric_dict: al_step_metric_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    performance_calculation_functions: "al_averaging_functions_dict",
) -> float:
    target_columns_gen = get_output_info_generator(
        outputs_as_dict=outputs_as_dict,
    )

    all_metrics = []

    for output_name, column_type, column_name in target_columns_gen:
        cur_metric_func = performance_calculation_functions[column_type]

        cur_value = cur_metric_func(
            metric_dict=metric_dict,
            output_name=output_name,
            column_name=column_name,
        )

        if not np.isnan(cur_value):
            all_metrics.append(cur_value)

    average = np.array(all_metrics).mean()

    return average


def handle_empty(default_value: float, metric_name: str | None = None):
    """
    This can happen when modelling on multiple outputs, which can vary in their
    sparsity, and by chance some outputs are empty in a batch.
    """

    def decorator(func):
        logged = False

        @wraps(func)
        def wrapper(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs):
            nonlocal logged
            if (outputs.size == 0 or labels.size == 0) and not logged:
                logger.info(
                    f"Empty inputs encountered in "
                    f"{metric_name or func.__name__}, "
                    f"returning default value: {default_value}"
                )
                logged = True
            if outputs.size == 0 or labels.size == 0:
                return default_value
            return func(outputs, labels, *args, **kwargs)

        return wrapper

    return decorator


def handle_class_mismatch(default_value: float, metric_name: str | None = None):
    """
    Decorator to handle cases where the number of unique classes in 'labels'
    does not match the number of columns in 'outputs'. This scenario can occur
    in multi-class classification tasks with sparse or imbalanced outputs.
    """

    def decorator(func):
        logged = False

        @wraps(func)
        def wrapper(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs):
            nonlocal logged
            unique_classes = np.unique(labels)
            # For binary classification as CE we have this special case
            if (
                outputs.shape[1] == 2
                and len(unique_classes) == 1
                or outputs.shape[1] == 1
            ):
                pass
            elif len(unique_classes) != outputs.shape[1]:
                if not logged:
                    logger.info(
                        f"Class mismatch encountered in "
                        f"{metric_name or func.__name__}, "
                        f"expected number of classes: {outputs.shape[1]}, "
                        f"found unique classes in labels: {len(unique_classes)}. "
                        f"Returning default value: {default_value}"
                    )
                    logged = True
                return default_value
            return func(outputs, labels, *args, **kwargs)

        return wrapper

    return decorator


@handle_empty(default_value=np.nan, metric_name="MCC")
def calc_mcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    prediction = parse_cat_pred(outputs=outputs)

    labels = labels.astype(int)

    num_classes = int(max(np.max(labels), np.max(prediction))) + 1

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf_matrix, (labels, prediction), 1)

    t_sum = conf_matrix.sum(axis=1, dtype=np.float64)
    p_sum = conf_matrix.sum(axis=0, dtype=np.float64)
    n_correct = np.sum(np.diag(conf_matrix), dtype=np.float64)
    n_samples = np.sum(conf_matrix, dtype=np.float64)

    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)

    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)


@handle_class_mismatch(default_value=np.nan, metric_name="ROC-AUC")
@handle_empty(default_value=np.nan, metric_name="ROC-AUC")
def calc_roc_auc_ovo(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    """
    TODO:   In rare scenarios, we might run into the issue of not having all labels
            represented in the labels array (i.e. labels were in train, but not in
            valid). This is not a problem for metrics like MCC / accuracy, but we
            will have to account for this here and in the AP calculation, possibly
            by ignoring columns in outputs and label_binarize outputs where the columns
            returned from label_binarize are all 0.
    """

    average = "macro"

    if outputs.shape[1] > 2:
        outputs = softmax(x=outputs, axis=1)
    elif outputs.shape[1] == 2:
        outputs = outputs[:, 1]

    roc_auc = roc_auc_score(
        y_true=labels,
        y_score=outputs,
        average=average,
        multi_class="ovo",
    )
    return roc_auc


@handle_class_mismatch(default_value=np.nan, metric_name="AP")
@handle_empty(default_value=np.nan, metric_name="AP")
def calc_average_precision(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    average = "macro"

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    elif outputs.shape[1] == 2:
        outputs = outputs[:, 1]

    average_precision = average_precision_score(
        y_true=labels,
        y_score=outputs,
        average=average,
    )

    return average_precision


def parse_cat_pred(outputs: np.ndarray) -> np.ndarray:
    # Single logit case
    if outputs.shape[1] == 1:
        pred = (outputs > 0).astype(int).squeeze()
    # Multi-class case
    else:
        pred = np.argmax(outputs, axis=1)

    return pred


@handle_empty(default_value=np.nan, metric_name="ACC")
def calc_acc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = parse_cat_pred(outputs=outputs)
    accuracy = np.mean(pred == labels)

    return accuracy


@handle_empty(default_value=np.nan, metric_name="F1-MACRO")
def calc_f1_score_macro(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    pred = parse_cat_pred(outputs=outputs)
    f1 = f1_score(y_true=labels, y_pred=pred, average="macro")
    return f1


@handle_empty(default_value=np.nan, metric_name="PRECISION-MACRO")
def calc_precision_macro(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    pred = parse_cat_pred(outputs=outputs)
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    return precision


@handle_empty(default_value=np.nan, metric_name="RECALL-MACRO")
def calc_recall_macro(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    pred = parse_cat_pred(outputs=outputs)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    return recall


@handle_empty(default_value=np.nan, metric_name="COHEN-KAPPA")
def calc_cohen_kappa(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = parse_cat_pred(outputs=outputs)
    kappa = cohen_kappa_score(y1=labels, y2=pred)
    return kappa


@handle_empty(default_value=np.nan, metric_name="PCC")
def calc_pcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    if len(outputs) < 2:
        return 0.0

    pcc = pearsonr(x=labels.squeeze(), y=outputs.squeeze())[0]
    return pcc


@handle_empty(default_value=np.nan, metric_name="R2")
def calc_r2(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    if len(outputs) < 2:
        return 0.0

    r2 = r2_score(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return r2


@handle_empty(default_value=np.nan, metric_name="RMSE")
def calc_rmse(
    outputs: np.ndarray,
    labels: np.ndarray,
    target_transformers: dict[str, dict[str, StandardScaler]],
    output_name: str,
    column_name: str,
    *args,
    **kwargs,
) -> float:
    cur_target_transformer = target_transformers[output_name][column_name]

    labels_2d = labels.reshape(-1, 1)
    outputs_2d = outputs.reshape(-1, 1)

    mean_ = cur_target_transformer.mean_
    scale_ = cur_target_transformer.scale_

    labels = (labels_2d * scale_ + mean_).squeeze()
    predictions = (outputs_2d * scale_ + mean_).squeeze()

    if np.shape(labels):
        rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    else:
        rmse = np.sqrt((labels - predictions) ** 2)

    return rmse


@handle_empty(default_value=np.nan, metric_name="MAE")
def calc_mae(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    mae = mean_absolute_error(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return mae


@handle_empty(default_value=np.nan, metric_name="MAPE")
def calc_mape(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    labels, outputs = labels.squeeze(), outputs.squeeze()
    mask = labels != 0
    return np.mean(np.abs((labels[mask] - outputs[mask]) / labels[mask])) * 100


@handle_empty(default_value=np.nan, metric_name="EXPLAINED-VARIANCE")
def calc_explained_variance(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    ev = explained_variance_score(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return ev


@handle_empty(default_value=np.nan, metric_name="C-INDEX")
def calc_survival_c_index(
    outputs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    target_transformers: dict[str, dict[str, KBinsDiscretizer | IdentityTransformer]],
    output_name: str,
    event_name: str,
    time_name: str,
    *args,
    **kwargs,
) -> float:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    target_transformer = target_transformers[output_name][time_name]
    model_type = "discrete"
    if isinstance(target_transformer, IdentityTransformer):
        model_type = "cox"

    events = labels
    times = target_transformer.inverse_transform(times.reshape(-1, 1)).flatten()

    # Convert model outputs to risk scores
    # Note: Unlike TD C-index, here we want risk scores (higher = more risk)
    if model_type == "discrete":
        hazards = sigmoid(outputs)
        survival_probs = np.cumprod(1 - hazards, axis=1)
        # Negative of survival prob = risk
        risk_scores = -survival_probs[:, -1]
    else:
        # Cox already outputs risk scores
        risk_scores = outputs.flatten()

    events_torch = torch.tensor(events, dtype=torch.bool)
    times_torch = torch.tensor(times, dtype=torch.float32)
    risk_scores_torch = torch.tensor(risk_scores, dtype=torch.float32)

    try:
        c_index = ConcordanceIndex()
        c_index_value = float(
            c_index(
                estimate=risk_scores_torch,
                event=events_torch,
                time=times_torch,
            )
        )

    except Exception as e:
        logger.error(f"C-index calculation failed: {str(e)}")
        c_index_value = np.nan

    return c_index_value


@handle_empty(default_value=np.nan, metric_name="IBS")
def calc_survival_ibs(
    outputs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    target_transformers: dict[str, dict[str, KBinsDiscretizer | IdentityTransformer]],
    output_name: str,
    event_name: str,
    time_name: str,
    *args,
    **kwargs,
) -> float:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    target_transformer = target_transformers[output_name][time_name]
    model_type = "discrete"
    if isinstance(target_transformer, IdentityTransformer):
        model_type = "cox"

    events = labels
    times = target_transformer.inverse_transform(times.reshape(-1, 1)).flatten()

    # Note: Unlike the C-index calculations, here we want survival probabilities
    if model_type == "discrete":
        hazards = sigmoid(outputs)
        survival_probs = np.cumprod(1 - hazards, axis=1)
        time_grid = target_transformer.bin_edges_[0][:-1]
        time_points = time_grid
        survival_preds = survival_probs
    else:
        risk_scores = outputs.flatten()
        # TODO: Maybe deprecate / remove * 0.99
        time_points = np.linspace(times.min(), times.max() * 0.99, 100)
        unique_times, baseline_hazard = estimate_baseline_hazard(
            times=times,
            events=events,
            risk_scores=risk_scores,
        )
        baseline_survival = np.exp(-np.cumsum(baseline_hazard))

        survival_preds = np.zeros((len(times), len(time_points)))
        for i, risk_score in enumerate(risk_scores):
            interp_baseline = np.interp(
                time_points,
                unique_times,
                baseline_survival,
                right=baseline_survival[-1],
            )
            survival_preds[i] = interp_baseline ** np.exp(risk_score)

    events_torch = torch.tensor(events, dtype=torch.bool)
    times_torch = torch.tensor(times, dtype=torch.float32)
    time_points_torch = torch.tensor(time_points, dtype=torch.float32)
    survival_probs_torch = torch.tensor(survival_preds, dtype=torch.float32)

    brier_score = BrierScore()
    try:
        brier_score(
            estimate=survival_probs_torch,
            event=events_torch,
            time=times_torch,
            new_time=time_points_torch,
        )
        brier_integral_score = brier_score.integral()
        return float(brier_integral_score)
    except Exception as e:
        logger.error(f"IBS calculation failed: {str(e)}")
        return np.nan


@handle_empty(default_value=np.nan, metric_name="TD-CINDEX")
def calc_survival_td_cindex(
    outputs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    target_transformers: dict[str, dict[str, KBinsDiscretizer | IdentityTransformer]],
    output_name: str,
    event_name: str,
    time_name: str,
    *args,
    **kwargs,
) -> float:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def select_tau(event_indicators: np.ndarray, time_values: np.ndarray) -> float:
        event_times = time_values[event_indicators.astype(bool)]
        tau_ = np.percentile(event_times, 80)
        return float(min(tau_, time_values.max()))

    target_transformer = target_transformers[output_name][time_name]
    model_type = "discrete"
    if isinstance(target_transformer, IdentityTransformer):
        model_type = "cox"

    events = labels
    times = target_transformer.inverse_transform(times.reshape(-1, 1)).flatten()

    if model_type == "discrete":
        hazards = sigmoid(outputs)
        survival_probs = np.cumprod(1 - hazards, axis=1)
        # Negative since higher survival = lower risk
        risk_scores = -survival_probs[:, -1]
    else:
        risk_scores = outputs.flatten()

    events_torch = torch.tensor(events, dtype=torch.bool)
    times_torch = torch.tensor(times, dtype=torch.float32)
    risk_scores_torch = torch.tensor(risk_scores, dtype=torch.float32)

    try:
        tau = select_tau(event_indicators=events, time_values=times)
        tau_torch = torch.tensor(tau, dtype=torch.float32)

        c_index = ConcordanceIndex()
        ipcw = get_ipcw(event=events_torch, time=times_torch)
        td_c_index = float(
            c_index(
                estimate=risk_scores_torch,
                event=events_torch,
                time=times_torch,
                tmax=tau_torch,
                weight=ipcw,
            )
        )

    except Exception as e:
        logger.error(f"Time-dependent C-index calculation failed: {str(e)}")
        td_c_index = np.nan

    return td_c_index


@handle_empty(default_value=np.nan, metric_name="TD-AUC")
def calc_survival_td_auc(
    outputs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    target_transformers: dict[str, dict[str, KBinsDiscretizer | IdentityTransformer]],
    output_name: str,
    event_name: str,
    time_name: str,
    *args,
    **kwargs,
) -> float:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    target_transformer = target_transformers[output_name][time_name]
    model_type = "discrete"
    if isinstance(target_transformer, IdentityTransformer):
        model_type = "cox"

    events = labels
    times = target_transformer.inverse_transform(times.reshape(-1, 1)).flatten()

    if model_type == "discrete":
        hazards = sigmoid(outputs)
        time_grid = target_transformer.bin_edges_[0][:-1]

        # Filter time points to be within observed time range,
        # otherwise AUC calculation fails
        time_mask = (time_grid >= times.min()) & (time_grid <= times.max())
        time_points = time_grid[time_mask]
        risk_scores = hazards[:, time_mask]
    else:
        risk_scores = outputs.flatten()
        time_points = np.linspace(times.min(), times.max() * 0.99, 100)

    # Convert to torch tensors
    events_torch = torch.tensor(events, dtype=torch.bool)
    times_torch = torch.tensor(times, dtype=torch.float32)
    time_points_torch = torch.tensor(time_points, dtype=torch.float32)
    risk_scores_torch = torch.tensor(risk_scores, dtype=torch.float32)

    try:
        auc = Auc()
        auc(
            estimate=risk_scores_torch,
            event=events_torch,
            time=times_torch,
            auc_type="cumulative",
            new_time=time_points_torch,
        )
        td_auc = float(auc.integral())

    except Exception as e:
        logger.error(f"Time-dependent AUC calculation failed: {str(e)}")
        td_auc = np.nan

    return td_auc


class LogEmptyLossProtocol(Protocol):
    def __call__(self, output_name: str, output_head_name: str) -> None: ...


def log_empty_loss_once() -> LogEmptyLossProtocol:
    logged_combinations = set()

    def log(output_name: str, output_head_name: str) -> None:
        if (output_name, output_head_name) not in logged_combinations:
            logger.info(
                f"Empty output batch encountered for {output_name},"
                f" {output_head_name}; "
                f"setting loss to NaN for this batch and future empty batches. "
                f"Empty batches will not be used for training."
            )
            logged_combinations.add((output_name, output_head_name))

    return log


def calculate_prediction_losses(
    criteria: "al_criteria_dict",
    inputs: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    We check for empty output tensors and log a warning if we encounter them.
    This can happen when modelling on multiple outputs, which can vary in their
    sparsity, and by chance some outputs are empty in a batch.
    """
    losses_dict: dict[str, dict[str, torch.Tensor]] = {}

    for output_name, _target_dict in targets.items():
        cur_inputs = inputs[output_name]
        cur_targets = targets[output_name]
        cur_criterion = criteria[output_name]
        losses_dict[output_name] = {}

        cur_losses = cur_criterion(cur_inputs, cur_targets)
        for name, loss in cur_losses.items():
            losses_dict[output_name][name] = loss

    return losses_dict


def aggregate_losses(losses_dict: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    losses_values = []
    for _output_name, targets_for_output_dict in losses_dict.items():
        for loss in targets_for_output_dict.values():
            losses_values.append(loss)

    if not losses_values:
        return torch.tensor(np.nan, requires_grad=True)

    average_loss = torch.nanmean(torch.stack(losses_values))
    return average_loss


def get_uncertainty_loss_hook(
    output_configs: Sequence[OutputConfig],
    device: str,
) -> tuple[Callable, dict[str, "UncertaintyMultiTaskLoss"]]:
    uncertainty_loss_modules: dict[str, UncertaintyMultiTaskLoss] = {}
    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            continue

        output_type_info = output_config.output_type_info
        assert isinstance(output_type_info, TabularOutputTypeConfig)

        if not output_type_info.uncertainty_weighted_mt_loss:
            continue

        logger.debug(
            f"Adding uncertainty loss for {output_config.output_info.output_name}."
        )

        cur_target_cat_columns = list(output_type_info.target_cat_columns)
        cur_target_con_columns = list(output_type_info.target_con_columns)
        loss_module = UncertaintyMultiTaskLoss(
            target_cat_columns=cur_target_cat_columns,
            target_con_columns=cur_target_con_columns,
            device=device,
        )
        uncertainty_loss_modules[output_config.output_info.output_name] = loss_module

    if len(uncertainty_loss_modules) == 0:
        raise ValueError("Expected at least one uncertainty loss module.")

    hook = partial(
        hook_add_uncertainty_loss, uncertainty_modules=uncertainty_loss_modules
    )

    return hook, uncertainty_loss_modules


def hook_add_uncertainty_loss(
    state: dict,
    uncertainty_modules: dict[str, "UncertaintyMultiTaskLoss"],
    loss_key: str = "per_target_train_losses",
    *args,
    **kwargs,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Note that we only update the relevant losses in the base dict.
    """

    base_losses_dict = state[loss_key]
    updated_losses = copy(base_losses_dict)

    for output_name, _module in uncertainty_modules.items():
        cur_module = uncertainty_modules[output_name]
        cur_loss_dict = base_losses_dict[output_name]
        cur_uncertainty_losses = cur_module(losses_dict=cur_loss_dict)
        updated_losses[output_name] = cur_uncertainty_losses

    state_updates = {loss_key: updated_losses}

    return state_updates


class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(
        self,
        target_cat_columns: list[str],
        target_con_columns: list[str],
        device: str,
    ):
        super().__init__()

        self.target_cat_columns = target_cat_columns
        self.target_con_columns = target_con_columns
        self.device = device

        self._construct_params(
            cur_target_columns=self.target_cat_columns + self.target_con_columns,
            device=self.device,
        )

    def _construct_params(self, cur_target_columns: list[str], device: str):
        for column_name in cur_target_columns:
            param = nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
            self.register_parameter(f"log_var_{column_name}", param)

    def _calc_uncertainty_loss(
        self,
        name: str,
        loss_value: torch.Tensor,
    ) -> torch.Tensor:
        log_var = getattr(self, f"log_var_{name}")
        scalar = 2.0 if name in self.target_cat_columns else 1.0

        precision = torch.exp(-log_var)
        loss = scalar * torch.sum(precision * loss_value + log_var)

        return loss

    def forward(self, losses_dict: dict) -> dict[str, torch.Tensor]:
        losses_uncertain = {}

        for target_column, loss_value_base in losses_dict.items():
            loss_value_uncertain = self._calc_uncertainty_loss(
                name=target_column,
                loss_value=loss_value_base,
            )
            losses_uncertain[target_column] = loss_value_uncertain

        return losses_uncertain


def hook_add_l1_loss(
    experiment: "Experiment",
    state: dict,
    loss_key: str = "loss",
    *args,
    **kwargs,
) -> dict:
    """
    TODO: Do the validation outside of the actual hook.
    """
    model_configs = experiment.inputs

    current_device = state[loss_key].device

    l1_loss = torch.tensor(0.0, device=current_device)
    for input_name, input_module in experiment.model.input_modules.items():
        cur_model_config = model_configs[input_name].input_config.model_config
        cur_model_init_config = cur_model_config.model_init_config

        current_l1 = getattr(cur_model_init_config, "l1", None)
        has_l1_weights = hasattr(input_module, "l1_penalized_weights")

        if current_l1 and not has_l1_weights:
            raise AttributeError(
                f"Module {input_module} for input name {input_name} does not have"
                f"l1_penalized_weights attribute."
            )

        if has_l1_weights and current_l1:
            input_module_with_l1 = cast("FeatureExtractorProtocolWithL1", input_module)
            cur_l1_loss = get_model_l1_loss(
                model=input_module_with_l1, l1_weight=current_l1
            )
            cur_l1_loss = cur_l1_loss.to(device=current_device)
            l1_loss += cur_l1_loss

    updated_loss = state[loss_key] + l1_loss
    state_updates = {loss_key: updated_loss}

    return state_updates


def get_model_l1_loss(
    model: "FeatureExtractorProtocolWithL1", l1_weight: float
) -> torch.Tensor:
    l1_loss = calc_l1_loss(
        weight_tensor=model.l1_penalized_weights, l1_weight=l1_weight
    )
    return l1_loss


def calc_l1_loss(weight_tensor: torch.Tensor, l1_weight: float):
    l1_loss = vector_norm(weight_tensor, ord=1, dim=None) * l1_weight
    return l1_loss


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: dict[str, str],
    writer_funcs: None | dict[str, dict[str, Callable]] = None,
) -> None:
    hc = handler_config
    exp = handler_config.experiment
    gc = exp.configs.global_config

    target_generator = get_output_info_generator(outputs_as_dict=exp.outputs)

    metrics_files = get_metrics_files(
        target_generator=target_generator,
        run_folder=hc.run_folder,
        train_or_val_target_prefix=f"{prefixes['metrics']}",
        detail_level=gc.evaluation_checkpoint.saved_result_detail_level,
    )

    if write_header:
        _ensure_metrics_paths_exists(metrics_files=metrics_files)

    for output_name, target_and_file_dict in metrics_files.items():
        for target_name, target_history_file in target_and_file_dict.items():
            cur_metric_dict = metrics_dict[output_name][target_name]

            if writer_funcs:
                cur_func = writer_funcs[output_name][target_name]
            else:
                cur_func = get_buffered_metrics_writer(buffer_interval=1)

            cur_func(
                filepath=target_history_file,
                metrics=cur_metric_dict,
                iteration=iteration,
                write_header=write_header,
            )


def get_metrics_files(
    target_generator: Generator[tuple[str, str, str]],
    run_folder: Path,
    train_or_val_target_prefix: str,
    detail_level: int,
) -> dict[str, dict[str, Path]]:
    assert train_or_val_target_prefix in ["validation_", "train_"]

    path_dict: dict[str, dict[str, Path]] = {}
    for output_name, _column_type, target_column in target_generator:
        if detail_level <= 1:
            continue

        if output_name not in path_dict:
            path_dict[output_name] = {}

        cur_file_name = train_or_val_target_prefix + target_column + "_history.log"
        cur_path = Path(
            run_folder,
            "results",
            output_name,
            target_column,
            cur_file_name,
        )
        path_dict[output_name][target_column] = cur_path

    average_loss_training_metrics_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix=train_or_val_target_prefix
    )
    path_dict["average"] = {"average": average_loss_training_metrics_file}

    return path_dict


def get_average_history_filepath(
    run_folder: Path, train_or_val_target_prefix: str
) -> Path:
    assert train_or_val_target_prefix in ["validation_", "train_"]
    metrics_file_path = run_folder / f"{train_or_val_target_prefix}average_history.log"
    return metrics_file_path


def _ensure_metrics_paths_exists(metrics_files: dict[str, dict[str, Path]]) -> None:
    for _output_name, target_and_file_dict in metrics_files.items():
        for path in target_and_file_dict.values():
            ensure_path_exists(path=path)


def get_buffered_metrics_writer(buffer_interval: int):
    """
    One might be tempted to use "a" mode here to append metrics to the log
    file instead of read-modify-write. However, some cloud storage platforms
    (e.g., S3) do not support appending (i.e., modifying) to files, raising
    OSError 95: Operation not supported.
    """
    buffer: list[dict[str, float]] = []

    def append_metrics_to_file(
        filepath: Path,
        metrics: dict[str, float],
        iteration: int,
        write_header: bool = False,
    ) -> None:
        nonlocal buffer

        dict_to_write = {"iteration": iteration, **metrics}
        buffer.append(dict_to_write)

        if not (iteration % buffer_interval == 0 or write_header):
            return

        fieldnames = ["iteration"] + sorted(metrics.keys())

        existing_rows = []
        if filepath.exists():
            with open(filepath) as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        all_rows = existing_rows + buffer

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        buffer.clear()

    return append_metrics_to_file


def read_metrics_history_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="iteration")

    return df


def get_metrics_dataframes(
    results_dir: Path, target_string: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_history_path = read_metrics_history_file(
        results_dir / f"train_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"validation_{target_string}_history.log"
    )

    return train_history_path, valid_history_path


def get_available_supervised_metrics() -> tuple[
    dict[str, MetricRecord], dict[str, MetricRecord]
]:
    cat_metrics: dict[str, MetricRecord] = {
        "mcc": MetricRecord(
            name="mcc",
            function=calc_mcc,
        ),
        "acc": MetricRecord(
            name="acc",
            function=calc_acc,
        ),
        "roc-auc-macro": MetricRecord(
            name="roc-auc-macro",
            function=calc_roc_auc_ovo,
        ),
        "ap-macro": MetricRecord(
            name="ap-macro",
            function=calc_average_precision,
        ),
        "f1-macro": MetricRecord(
            name="f1-macro",
            function=calc_f1_score_macro,
        ),
        "precision-macro": MetricRecord(
            name="precision-macro",
            function=calc_precision_macro,
        ),
        "recall-macro": MetricRecord(
            name="recall-macro",
            function=calc_recall_macro,
        ),
        "cohen-kappa": MetricRecord(
            name="cohen-kappa",
            function=calc_cohen_kappa,
        ),
    }

    con_metrics: dict[str, MetricRecord] = {
        "rmse": MetricRecord(
            name="rmse",
            function=partial(calc_rmse, target_transformers=None),
            minimize_goal=True,
        ),
        "r2": MetricRecord(
            name="r2",
            function=calc_r2,
        ),
        "pcc": MetricRecord(
            name="pcc",
            function=calc_pcc,
        ),
        "mae": MetricRecord(
            name="mae",
            function=calc_mae,
            minimize_goal=True,
        ),
        "mape": MetricRecord(
            name="mape",
            function=calc_mape,
            minimize_goal=True,
        ),
        "explained-variance": MetricRecord(
            name="explained-variance",
            function=calc_explained_variance,
        ),
    }

    return cat_metrics, con_metrics


def get_available_survival_metrics(
    target_transformers: dict[str, "al_label_transformers"],
) -> tuple[MetricRecord, ...]:
    survival_metrics: tuple[MetricRecord, ...] = (
        MetricRecord(
            name="c-index",
            function=partial(
                calc_survival_c_index, target_transformers=target_transformers
            ),
            output_type="survival",
            minimize_goal=False,
        ),
        MetricRecord(
            name="ibs",
            function=partial(
                calc_survival_ibs, target_transformers=target_transformers
            ),
            output_type="survival",
            minimize_goal=True,
        ),
        MetricRecord(
            name="td-cindex",
            function=partial(
                calc_survival_td_cindex, target_transformers=target_transformers
            ),
            output_type="survival",
            minimize_goal=False,
        ),
        MetricRecord(
            name="td-auc",
            function=partial(
                calc_survival_td_auc, target_transformers=target_transformers
            ),
            output_type="survival",
            minimize_goal=False,
        ),
    )

    return survival_metrics


def get_default_metrics(
    target_transformers: dict[str, "al_label_transformers"],
    cat_metrics: al_cat_metric_choices,
    con_metrics: al_con_metric_choices,
    cat_averaging_metrics: al_cat_metric_choices | None,
    con_averaging_metrics: al_con_metric_choices | None,
    output_configs: Sequence[OutputConfig],
) -> "al_metric_record_dict":
    available_cat_metrics, available_con_metrics = get_available_supervised_metrics()

    cat_metric_records = tuple(
        available_cat_metrics[metric]
        for metric in cat_metrics
        if metric in available_cat_metrics
    )

    con_metric_records = tuple(
        available_con_metrics[metric]
        for metric in con_metrics
        if metric in available_con_metrics
    )

    for metric in con_metric_records:
        if metric.name == "rmse":
            metric.function = partial(
                calc_rmse, target_transformers=target_transformers
            )

    cat_for_avg, con_for_avg = parse_averaging_metrics(
        cat_averaging_metrics=cat_averaging_metrics,
        con_averaging_metrics=con_averaging_metrics,
    )
    averaging_functions = get_performance_averaging_functions(
        cat_metric_names=cat_for_avg,
        con_metric_names=con_for_avg,
    )

    surv_metric_records = get_available_survival_metrics(
        target_transformers=target_transformers
    )

    general_metric_info = _build_general_metric_info(
        cat_metric_records=cat_metric_records,
        con_metric_records=con_metric_records,
        surv_metric_records=surv_metric_records,
        output_configs=output_configs,
    )

    default_metrics: al_metric_record_dict = {
        "cat": cat_metric_records,
        "con": con_metric_records,
        "survival": surv_metric_records,
        "averaging_functions": averaging_functions,
        "general_metric_info": general_metric_info,
    }
    return default_metrics


@dataclass
class GeneralMetricInfo:
    all_are_val_only: bool
    base_metric_structure: dict[str, dict[str, dict]]


def _build_general_metric_info(
    cat_metric_records: tuple[MetricRecord, ...],
    con_metric_records: tuple[MetricRecord, ...],
    surv_metric_records: tuple[MetricRecord, ...],
    output_configs: Sequence[OutputConfig],
) -> GeneralMetricInfo:
    """
    This function and the created object is used in in the training loop to:

        1. Determine if all metrics are validation-only metrics. If that is the case
        the metric calculation for training batches is completely skipped.
        2. If (1) is True, we create a base structure for the metric calculation
        which is used in other parts of the code (e.g. for running average metrics).
        The structure is bootstrapped here, even though it is never filled with metrics
        as the general output_name -> target name nested dict structure is e.g.
        used when adding losses to the state dict in the training loop.
    """
    all_cat_are_val = all(metric.only_val for metric in cat_metric_records)
    all_con_are_val = all(metric.only_val for metric in con_metric_records)
    all_surv_are_val = all(metric.only_val for metric in surv_metric_records)
    all_are_val_only = all([all_cat_are_val, all_con_are_val, all_surv_are_val])

    if all_are_val_only:
        logger.debug(
            "All metrics are validation-only metrics, will skip all metric "
            "calculation for training batches."
        )

    base_structure: dict[str, Any] = {}
    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        base_structure[output_name] = {}

        oti = output_config.output_type_info
        match oti:
            case TabularOutputTypeConfig():
                targets = list(oti.target_cat_columns) + list(oti.target_con_columns)
            case SurvivalOutputTypeConfig():
                targets = [oti.event_column]
            case _:
                targets = [output_name]

        for target in targets:
            base_structure[output_name][target] = {}

    return GeneralMetricInfo(
        all_are_val_only=all_are_val_only,
        base_metric_structure=base_structure,
    )


def parse_averaging_metrics(
    cat_averaging_metrics: al_cat_metric_choices | None,
    con_averaging_metrics: al_con_metric_choices | None,
) -> tuple[al_cat_metric_choices, al_con_metric_choices]:
    cat_parsed, con_parsed = _get_default_averaging_metrics()

    if cat_averaging_metrics:
        _validate_metrics(
            passed_in_metrics=cat_averaging_metrics,
            expected_metrics=[
                "loss",
                "acc",
                "mcc",
                "roc-auc-macro",
                "ap-macro",
            ],
            target_type="categorical",
        )
        cat_parsed = cat_averaging_metrics
    if con_averaging_metrics:
        _validate_metrics(
            passed_in_metrics=con_averaging_metrics,
            expected_metrics=[
                "loss",
                "rmse",
                "pcc",
                "r2",
            ],
            target_type="continuous",
        )
        con_parsed = con_averaging_metrics

    return cat_parsed, con_parsed


def _validate_metrics(
    passed_in_metrics: Sequence[str], expected_metrics: Sequence[str], target_type: str
) -> None:
    for metric in passed_in_metrics:
        if metric not in expected_metrics:
            raise ValueError(
                f"Metric {metric} not found in expected metrics {expected_metrics} for "
                f" {target_type} targets."
            )


def _get_default_averaging_metrics() -> tuple[
    al_cat_metric_choices, al_con_metric_choices
]:
    cat_names: list[Literal["mcc", "roc-auc-macro", "ap-macro", "acc"]]
    con_names: list[Literal["loss", "pcc", "r2"]]

    cat_names = ["mcc", "roc-auc-macro", "ap-macro"]
    con_names = ["loss", "pcc", "r2"]

    return cat_names, con_names


def get_performance_averaging_functions(
    cat_metric_names: al_cat_metric_choices,
    con_metric_names: al_con_metric_choices,
) -> al_averaging_functions_dict:
    """
    Note that we have the mean(values) else 0.0 to account for some values not being
    computed on the training batches, e.g. ROC-AUC, due some metrics possibly
    raising errors if e.g. there are only negative labels in a batch.
    """

    parsed_con_names = []
    for metric in con_metric_names:
        if metric in ["rmse", "mae", "mape"]:
            logger.warning(
                "Using metric %s for performance averaging, which is affected by "
                "the scale of the target. This can lead to the metric dominating "
                "the performance calculation. Consider using a different metric "
                "for performance averaging.",
            )

        if metric in ["loss", "mae", "rmse", "mape"]:
            parsed_con_names.append(f"1.0-{metric.upper()}")
        else:
            parsed_con_names.append(metric.upper())

    logger.info(
        "Tabular output performance averaging functions across tasks set to averages "
        "of %s for categorical targets and %s for continuous targets. These "
        "values are used to determine overall performance (using the validation set), "
        "which is used to control factors such as early stopping and LR scheduling. "
        "Other output cases (e.g. sequence generation, image generation) "
        "use 1.0-LOSS by default.",
        [i.upper() for i in cat_metric_names],
        parsed_con_names,
    )

    def _calc_cat_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_names: al_cat_metric_choices | all_survival_metric_choices,
    ) -> float:
        values = []
        for metric_name in metric_names:
            combined_key = f"{output_name}_{column_name}_{metric_name}"
            value = metric_dict[output_name][column_name].get(combined_key, None)

            if value is None:
                continue

            values.append(float(value))

        return mean(values) if values else 0.0

    def _calc_con_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_names: al_con_metric_choices,
    ) -> float:
        values = []
        for metric_name in metric_names:
            combined_key = f"{output_name}_{column_name}_{metric_name}"
            value = metric_dict[output_name][column_name].get(combined_key, None)

            if value is None:
                continue

            if metric_name in ["loss", "rmse", "mae", "mape"]:
                value = 1.0 - value

            values.append(float(value))

        return mean(values) if values else 0.0

    surv_metric_names: all_survival_metric_choices = ["c-index"]

    performance_averaging_functions: al_averaging_functions_dict = {
        "cat": partial(_calc_cat_averaging_value, metric_names=cat_metric_names),
        "con": partial(_calc_con_averaging_value, metric_names=con_metric_names),
        "survival": partial(_calc_cat_averaging_value, metric_names=surv_metric_names),
        "general": partial(_calc_con_averaging_value, metric_names=["loss"]),
    }

    return performance_averaging_functions


@dataclass
class FilteredOutputsAndLabels:
    model_outputs: dict[str, dict[str, torch.Tensor]]
    target_labels: dict[str, dict[str, torch.Tensor]]
    ids: dict[str, dict[str, list[str]]] | None
    common_ids: list[str] | None


def filter_missing_outputs_and_labels(
    batch_ids: list[str],
    model_outputs: dict[str, dict[str, torch.Tensor]],
    target_labels: dict[str, dict[str, torch.Tensor]],
    missing_ids_info: MissingTargetsInfo,
    with_labels: bool = True,
) -> FilteredOutputsAndLabels:
    if missing_ids_info.all_have_same_set:
        return FilteredOutputsAndLabels(
            model_outputs=model_outputs,
            target_labels=target_labels,
            ids=None,
            common_ids=batch_ids,
        )

    filtered_outputs = {}
    filtered_labels = {}
    filtered_ids: dict[str, dict[str, list[str]]] = {}

    missing_per_modality = missing_ids_info.missing_ids_per_modality

    for output_name, output_inner_dict in model_outputs.items():
        cur_output_missing = missing_per_modality.get(output_name, set())
        if not cur_output_missing:
            filtered_outputs[output_name] = output_inner_dict
            filtered_labels[output_name] = (
                target_labels[output_name]
                if with_labels
                else {k: torch.tensor([]) for k in output_inner_dict}
            )
            filtered_ids[output_name] = dict.fromkeys(output_inner_dict, batch_ids)
            continue

        missing_ids = missing_ids_info.missing_ids_per_modality[output_name]
        valid_indices = [i for i, id_ in enumerate(batch_ids) if id_ not in missing_ids]

        device = next(iter(output_inner_dict.values())).device
        valid_indices_tensor = torch.tensor(
            valid_indices,
            device=device,
            dtype=torch.long,
        )

        filtered_outputs[output_name] = {
            k: v.index_select(0, valid_indices_tensor)
            for k, v in output_inner_dict.items()
        }

        if with_labels:
            filtered_labels[output_name] = {
                k: v.index_select(0, valid_indices_tensor)
                for k, v in target_labels[output_name].items()
            }
        else:
            filtered_labels[output_name] = {
                k: torch.tensor([]) for k in output_inner_dict
            }
        filtered_ids[output_name] = {
            k: [batch_ids[i] for i in valid_indices] for k in output_inner_dict
        }

    return FilteredOutputsAndLabels(
        model_outputs=filtered_outputs,
        target_labels=filtered_labels,
        ids=filtered_ids,
        common_ids=None,
    )


@dataclass
class FilteredTabularTargets:
    model_outputs: torch.Tensor
    target_labels: torch.Tensor
    ids: list[str]


def filter_tabular_missing_targets(
    outputs: torch.Tensor,
    target_labels: torch.Tensor,
    ids: list[str],
    target_type: str,
) -> FilteredTabularTargets:
    nan_labels_mask = torch.isnan(target_labels)
    if target_type == "cat":
        nan_labels_mask = torch.logical_or(nan_labels_mask, target_labels == -1)

    if ids:
        cur_ids_np = np.array(ids)
        nan_labels_mask_np = nan_labels_mask.cpu().numpy()
        ids = cur_ids_np[~nan_labels_mask_np].tolist()

    predictions = outputs[~nan_labels_mask]
    target_labels = target_labels[~nan_labels_mask]

    return FilteredTabularTargets(
        model_outputs=predictions,
        target_labels=target_labels,
        ids=ids,
    )


@dataclass
class FilteredSurvivalTargets:
    model_outputs: torch.Tensor
    events: torch.Tensor
    times: torch.Tensor
    ids: list[str]


def filter_survival_missing_targets(
    model_outputs: torch.Tensor,
    events: torch.Tensor,
    times: torch.Tensor,
    cur_ids: list[str],
) -> FilteredSurvivalTargets:
    nan_labels_mask = events == -1
    cur_times_nan = torch.isnan(times)
    nan_labels_mask = torch.logical_or(nan_labels_mask, cur_times_nan)

    predictions = model_outputs[~nan_labels_mask]
    events = events[~nan_labels_mask]
    times = times[~nan_labels_mask]

    if cur_ids:
        nan_labels_mask_np = nan_labels_mask.cpu().numpy()
        cur_ids = np.array(cur_ids)[~nan_labels_mask_np].tolist()

    return FilteredSurvivalTargets(
        model_outputs=predictions,
        events=events,
        times=times,
        ids=cur_ids,
    )


def general_torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.cpu().to(dtype=torch.float32).detach().numpy()
    return array


def estimate_baseline_hazard(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    events = events[sort_idx]
    risk_scores = risk_scores[sort_idx]

    unique_times = np.unique(times[events == 1])
    baseline_hazard = np.zeros_like(unique_times)

    # convert back to risk scores from log-hazard ratios
    risk_scores_exp = np.exp(risk_scores)

    for i, t in enumerate(unique_times):
        events_at_t = events[times == t]
        n_events = np.sum(events_at_t)

        at_risk = times >= t
        risk_set_sum = np.sum(risk_scores_exp[at_risk])

        if risk_set_sum > 0:
            baseline_hazard[i] = n_events / risk_set_sum

    return unique_times, baseline_hazard
