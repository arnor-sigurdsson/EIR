import csv
from copy import copy
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, TYPE_CHECKING, List, Tuple, Callable, Union

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from scipy.stats import pearsonr
from sklearn.metrics import (
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from eir.data_load.data_utils import get_target_columns_generator
from eir.data_load.label_setup import al_label_transformers

if TYPE_CHECKING:
    from eir.train import al_criterions, Config  # noqa: F401
    from eir.models.omics.omics_models import al_models  # noqa: F401
    from eir.train_utils.train_handlers import HandlerConfig
    from eir.data_load.label_setup import (  # noqa: F401
        al_target_columns,
        al_label_transformers_object,
    )

# aliases
al_step_metric_dict = Dict[str, Dict[str, float]]
al_metric_record_dict = Dict[
    str, Union[Tuple["MetricRecord", ...], "al_averaging_functions_dict"]
]
al_averaging_functions_dict = Dict[
    str, Callable[["al_step_metric_dict", str, str], float]
]

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class MetricRecord:
    name: str
    function: Callable
    only_val: bool = False
    minimize_goal: bool = False


def calculate_batch_metrics(
    target_columns: "al_target_columns",
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    mode: str,
    metric_record_dict: al_metric_record_dict,
) -> al_step_metric_dict:
    assert mode in ["val", "train"]

    target_columns_gen = get_target_columns_generator(target_columns=target_columns)

    master_metric_dict = {}

    for column_type, column_name in target_columns_gen:
        cur_metric_dict = {}

        cur_metric_records: Tuple[MetricRecord, ...] = metric_record_dict[column_type]
        cur_outputs = outputs[column_name].detach().cpu().numpy()
        cur_labels = labels[column_name].cpu().numpy()

        for metric_record in cur_metric_records:

            if metric_record.only_val and mode == "train":
                continue

            cur_key = f"{column_name}_{metric_record.name}"
            cur_metric_dict[cur_key] = metric_record.function(
                outputs=cur_outputs, labels=cur_labels, column_name=column_name
            )

        master_metric_dict[column_name] = cur_metric_dict

    return master_metric_dict


def add_loss_to_metrics(
    target_columns: "al_target_columns",
    losses: Dict[str, torch.Tensor],
    metric_dict: al_step_metric_dict,
) -> al_step_metric_dict:

    target_columns_gen = get_target_columns_generator(target_columns=target_columns)
    metric_dict_copy = copy(metric_dict)

    for column_type, column_name in target_columns_gen:
        cur_metric_dict = metric_dict_copy[column_name]
        cur_metric_dict[f"{column_name}_loss"] = losses[column_name].item()

    return metric_dict_copy


def add_multi_task_average_metrics(
    batch_metrics_dict: al_step_metric_dict,
    target_columns: "al_target_columns",
    loss: float,
    performance_average_functions: Dict[str, Callable[[al_step_metric_dict], float]],
) -> al_step_metric_dict:
    average_performance = average_performances_across_tasks(
        metric_dict=batch_metrics_dict,
        target_columns=target_columns,
        performance_calculation_functions=performance_average_functions,
    )
    batch_metrics_dict["average"] = {
        "loss-average": loss,
        "perf-average": average_performance,
    }

    return batch_metrics_dict


def average_performances_across_tasks(
    metric_dict: al_step_metric_dict,
    target_columns: "al_target_columns",
    performance_calculation_functions: Dict[
        str, Callable[[al_step_metric_dict], float]
    ],
) -> float:
    target_columns_gen = get_target_columns_generator(target_columns)

    all_metrics = []

    for column_type, column_name in target_columns_gen:
        cur_metric_func = performance_calculation_functions.get(column_type)

        metric_func_args = {"metric_dict": metric_dict, "column_name": column_name}
        cur_value = cur_metric_func(**metric_func_args)
        all_metrics.append(cur_value)

        all_metrics.append(cur_value)

    average = np.array(all_metrics).mean()

    return average


def calc_mcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(a=outputs, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mcc = matthews_corrcoef(labels, pred)

    return mcc


def calc_roc_auc_ovr(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:
    """
    TODO:   In rare scenarios, we might run into the issue of not having all labels
            represented in the labels array (i.e. labels were in train, but not in
            valid). This is not a problem for metrics like MCC / accuracy, but we
            will have to account for this here and in the AP calculation, possibly
            by ignoring columns in outputs and label_binarize outputs where the columns
            returned from label_binarize are all 0.
    """

    assert average in ["micro", "macro"]

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    else:
        outputs = outputs[:, 1]

    roc_auc = roc_auc_score(y_true=labels, y_score=outputs, average=average)
    return roc_auc


def calc_average_precision_ovr(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:

    assert average in ["micro", "macro"]

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    else:
        outputs = outputs[:, 1]

    average_precision = average_precision_score(
        y_true=labels, y_score=outputs, average=average
    )

    return average_precision


def calc_acc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(outputs, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return accuracy


def calc_pcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    pcc = pearsonr(x=labels.squeeze(), y=outputs.squeeze())[0]
    return pcc


def calc_r2(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    r2 = r2_score(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return r2


def calc_rmse(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    target_transformers: Dict[str, StandardScaler],
    column_name: str,
    *args,
    **kwargs,
) -> float:
    cur_target_transformer = target_transformers[column_name]

    labels = cur_target_transformer.inverse_transform(labels).squeeze()
    preds = cur_target_transformer.inverse_transform(outputs).squeeze()

    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))
    return rmse


def calculate_prediction_losses(
    criterions: "al_criterions",
    inputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses_dict = {}

    for target_column, criterion in criterions.items():
        cur_target_col_labels = targets[target_column]
        cur_target_col_outputs = inputs[target_column]
        losses_dict[target_column] = criterion(
            input=cur_target_col_outputs, target=cur_target_col_labels
        )

    return losses_dict


def aggregate_losses(losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses_values = list(losses_dict.values())
    average_loss = torch.mean(torch.stack(losses_values))

    return average_loss


def get_uncertainty_loss_hook(
    target_cat_columns: List[str],
    target_con_columns: List[str],
    device: str,
):
    loss_module = UncertaintyMultiTaskLoss(
        target_cat_columns=target_cat_columns,
        target_con_columns=target_con_columns,
        device=device,
    )

    hook = partial(hook_add_uncertainty_loss, uncertainty_module=loss_module)

    return hook


def hook_add_uncertainty_loss(
    state: Dict,
    uncertainty_module: "UncertaintyMultiTaskLoss",
    loss_key: str = "per_target_train_losses",
    *args,
    **kwargs,
):
    base_losses_dict = state[loss_key]

    uncertainty_losses = uncertainty_module(losses_dict=base_losses_dict)

    state_updates = {loss_key: uncertainty_losses}

    return state_updates


class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(
        self,
        target_cat_columns: List[str],
        target_con_columns: List[str],
        device: str,
    ):
        super().__init__()

        self.target_cat_columns = target_cat_columns
        self.target_con_columns = target_con_columns
        self.device = device

        self.log_vars = self._construct_params(
            cur_target_columns=self.target_cat_columns + self.target_con_columns,
            device=self.device,
        )

    @staticmethod
    def _construct_params(cur_target_columns: List[str], device: str):
        param_dict = {}
        for column_name in cur_target_columns:
            param_dict[column_name] = nn.Parameter(
                torch.zeros(1), requires_grad=True
            ).to(device=device)

        return param_dict

    def _calc_uncertainty_loss(self, name, loss_value):
        log_var = self.log_vars[name]
        scalar = 2.0 if name in self.target_cat_columns else 1.0

        precision = torch.exp(-log_var)
        loss = scalar * torch.sum(precision * loss_value + log_var)

        return loss

    def forward(self, losses_dict: Dict):
        losses_uncertain = {}

        for target_column, loss_value_base in losses_dict.items():
            loss_value_uncertain = self._calc_uncertainty_loss(
                name=target_column, loss_value=loss_value_base
            )
            losses_uncertain[target_column] = loss_value_uncertain

        return losses_uncertain


def hook_add_l1_loss(
    config: "Config",
    state: Dict,
    loss_key: str = "loss",
    *args,
    **kwargs,
) -> Dict:
    l1_loss = get_model_l1_loss(model=config.model, l1_weight=config.cl_args.l1)

    updated_loss = state[loss_key] + l1_loss

    state_updates = {loss_key: updated_loss}

    return state_updates


def get_model_l1_loss(model: "al_models", l1_weight: float) -> torch.Tensor:
    l1_loss = calc_l1_loss(
        weight_tensor=model.l1_penalized_weights, l1_weight=l1_weight
    )
    return l1_loss


def calc_l1_loss(weight_tensor: torch.Tensor, l1_weight: float):
    l1_loss = torch.norm(weight_tensor, p=1) * l1_weight
    return l1_loss


def add_extra_losses(total_loss: torch.Tensor, extra_loss_functions: List[Callable]):
    """
    TODO: Possibly add inputs and labels as arguments here if needed later.
    """
    for loss_func in extra_loss_functions:
        total_loss += loss_func()

    return total_loss


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: Dict[str, str],
):

    hc = handler_config
    c = handler_config.config
    cl_args = c.cl_args

    metrics_files = get_metrics_files(
        target_columns=c.target_columns,
        run_folder=hc.run_folder,
        train_or_val_target_prefix=f"{prefixes['metrics']}",
    )

    if write_header:
        _ensure_metrics_paths_exists(metrics_files=metrics_files)

    for metrics_name, metrics_history_file in metrics_files.items():
        cur_metric_dict = metrics_dict[metrics_name]

        _add_metrics_to_writer(
            name=f"{prefixes['writer']}/{metrics_name}",
            metric_dict=cur_metric_dict,
            iteration=iteration,
            writer=c.writer,
            plot_skip_steps=cl_args.plot_skip_steps,
        )

        _append_metrics_to_file(
            filepath=metrics_history_file,
            metrics=cur_metric_dict,
            iteration=iteration,
            write_header=write_header,
        )


def get_metrics_files(
    target_columns: "al_target_columns",
    run_folder: Path,
    train_or_val_target_prefix: str,
) -> Dict[str, Path]:
    assert train_or_val_target_prefix in ["validation_", "train_"]

    all_target_columns = target_columns["con"] + target_columns["cat"]

    path_dict = {}
    for target_column in all_target_columns:
        cur_fname = train_or_val_target_prefix + target_column + "_history.log"
        cur_path = Path(run_folder, "results", target_column, cur_fname)
        path_dict[target_column] = cur_path

    average_loss_training_metrics_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix=train_or_val_target_prefix
    )
    path_dict["average"] = average_loss_training_metrics_file

    return path_dict


def get_average_history_filepath(
    run_folder: Path, train_or_val_target_prefix: str
) -> Path:
    assert train_or_val_target_prefix in ["validation_", "train_"]
    metrics_file_path = run_folder / f"{train_or_val_target_prefix}average_history.log"
    return metrics_file_path


def _ensure_metrics_paths_exists(metrics_files: Dict[str, Path]) -> None:
    for path in metrics_files.values():
        ensure_path_exists(path)


def _add_metrics_to_writer(
    name: str,
    metric_dict: Dict[str, float],
    iteration: int,
    writer: SummaryWriter,
    plot_skip_steps: int,
) -> None:
    """
    We do %10 to reduce the amount of training data going to tensorboard, otherwise
    it slows down with many large experiments.
    """
    if iteration >= plot_skip_steps and iteration % 10 == 0:
        for metric_name, metric_value in metric_dict.items():
            cur_name = name + f"/{metric_name}"
            writer.add_scalar(
                tag=cur_name, scalar_value=metric_value, global_step=iteration
            )


def _append_metrics_to_file(
    filepath: Path, metrics: Dict[str, float], iteration: int, write_header=False
):
    """
    TODO:   Have cached file handles here instead of reopening the file at every
            iteration.
    """
    with open(str(filepath), "a") as logfile:
        fieldnames = ["iteration"] + sorted(metrics.keys())
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        dict_to_write = {**{"iteration": iteration}, **metrics}
        writer.writerow(dict_to_write)


def read_metrics_history_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="iteration")

    return df


def get_metrics_dataframes(
    results_dir: Path, target_string: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_history_path = read_metrics_history_file(
        results_dir / f"train_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"validation_{target_string}_history.log"
    )

    return train_history_path, valid_history_path


def get_default_metrics(
    target_transformers: al_label_transformers,
) -> "al_metric_record_dict":
    mcc = MetricRecord(name="mcc", function=calc_mcc)
    acc = MetricRecord(name="acc", function=calc_acc)

    rmse = MetricRecord(
        name="rmse",
        function=partial(calc_rmse, target_transformers=target_transformers),
        minimize_goal=True,
    )

    roc_auc_macro = MetricRecord(
        name="roc-auc-macro", function=calc_roc_auc_ovr, only_val=True
    )
    ap_macro = MetricRecord(
        name="ap-macro", function=calc_average_precision_ovr, only_val=True
    )
    r2 = MetricRecord(name="r2", function=calc_r2, only_val=True)
    pcc = MetricRecord(name="pcc", function=calc_pcc, only_val=True)

    averaging_functions = get_default_performance_averaging_functions(
        cat_metric_name="mcc", con_metric_name="loss"
    )
    default_metrics = {
        "cat": (mcc, acc, roc_auc_macro, ap_macro),
        "con": (rmse, r2, pcc),
        "averaging_functions": averaging_functions,
    }
    return default_metrics


def get_default_performance_averaging_functions(
    cat_metric_name: str, con_metric_name: str
) -> al_averaging_functions_dict:
    def _calc_cat_averaging_value(
        metric_dict: "al_step_metric_dict", column_name: str, metric_name: str
    ) -> float:
        return metric_dict[column_name][f"{column_name}_{metric_name}"]

    def _calc_con_averaging_value(
        metric_dict: "al_step_metric_dict", column_name: str, metric_name: str
    ) -> float:
        return 1 - metric_dict[column_name][f"{column_name}_{metric_name}"]

    performance_averaging_functions = {
        "cat": partial(_calc_cat_averaging_value, metric_name=cat_metric_name),
        "con": partial(_calc_con_averaging_value, metric_name=con_metric_name),
    }

    return performance_averaging_functions
