import csv
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, Union, TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from human_origins_supervised.data_load.data_utils import get_target_columns_generator

if TYPE_CHECKING:
    from human_origins_supervised.train import al_criterions
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.data_load.label_setup import (  # noqa: F401
        al_target_columns,
        al_label_transformers,
    )

# aliases
al_step_metric_dict = Dict[str, Dict[str, float]]


def calculate_batch_metrics(
    target_columns: "al_target_columns",
    target_transformers: Dict[str, "al_label_transformers"],
    losses: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    prefix: str,
) -> al_step_metric_dict:
    target_columns_gen = get_target_columns_generator(target_columns)

    master_metric_dict = {}
    for column_type, column_name in target_columns_gen:

        metric_func = select_metric_func(column_type, target_transformers[column_name])
        cur_outputs = outputs[column_name]
        cur_labels = labels[column_name]

        cur_metric_dict = metric_func(
            outputs=cur_outputs, labels=cur_labels, prefix=f"{prefix}{column_name}"
        )
        cur_metric_dict[f"{prefix}{column_name}_loss"] = losses[column_name].item()

        master_metric_dict[column_name] = cur_metric_dict

    return master_metric_dict


def add_multi_task_average_metrics(
    batch_metrics_dict: al_step_metric_dict,
    target_columns: "al_target_columns",
    prefix: str,
    loss: float,
):
    average_performance = average_performances(
        metric_dict=batch_metrics_dict, target_columns=target_columns, prefix=prefix
    )
    batch_metrics_dict[f"{prefix}average"] = {
        f"{prefix}loss-average": loss,
        f"{prefix}perf-average": average_performance,
    }

    return batch_metrics_dict


def average_performances(
    metric_dict: al_step_metric_dict, target_columns: "al_target_columns", prefix: str
) -> float:
    target_columns_gen = get_target_columns_generator(target_columns)

    all_metrics = []
    for column_type, column_name in target_columns_gen:
        if column_type == "con":
            value = 1 - metric_dict[column_name][f"{prefix}{column_name}_loss"]
        elif column_type == "cat":
            value = metric_dict[column_name][f"{prefix}{column_name}_mcc"]
        else:
            raise ValueError()

        all_metrics.append(value)

    average = np.array(all_metrics).mean()

    return average


def select_metric_func(
    target_column_type: str, target_transformer: Union[StandardScaler, LabelEncoder]
):
    if target_column_type == "cat":
        return calc_multiclass_metrics

    return partial(calc_regression_metrics, target_transformer=target_transformer)


def get_train_metrics(column_type: str, prefix: str = "t") -> List[str]:
    if column_type == "con":
        base = [f"{prefix}_r2", f"{prefix}_rmse", f"{prefix}_pcc"]
    elif column_type == "cat":
        base = [f"{prefix}_mcc"]
    else:
        raise ValueError()

    all_metrics = base + [f"{prefix}_loss"]

    return all_metrics


def calc_multiclass_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:

    _, pred = torch.max(outputs, 1)

    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mcc = matthews_corrcoef(labels, pred)

    return {f"{prefix}_mcc": mcc}


def calc_regression_metrics(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    prefix: str,
    target_transformer: StandardScaler,
) -> Dict[str, float]:
    preds = outputs.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    labels = target_transformer.inverse_transform(labels).squeeze()
    preds = target_transformer.inverse_transform(preds).squeeze()

    if len(preds) < 2:
        r2 = 0
        pcc = 0
    else:
        r2 = r2_score(y_true=labels, y_pred=preds)
        pcc = pearsonr(x=labels, y=preds)[0]
    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))

    return {f"{prefix}_r2": r2, f"{prefix}_rmse": rmse, f"{prefix}_pcc": pcc}


def calculate_losses(
    criterions: "al_criterions",
    labels: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses_dict = {}

    for target_column, criterion in criterions.items():
        cur_target_col_labels = labels[target_column]
        cur_target_col_outputs = outputs[target_column]
        losses_dict[target_column] = criterion(
            input=cur_target_col_outputs, target=cur_target_col_labels
        )

    return losses_dict


def aggregate_losses(losses_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses_values = list(losses_dict.values())
    average_loss = torch.mean(torch.stack(losses_values))

    return average_loss


class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(
        self,
        target_columns: "al_target_columns",
        criterions: "al_criterions",
        device: str,
    ):
        super().__init__()

        self.target_columns = target_columns
        self.criterions = criterions
        self.device = device

        self.log_vars = self._construct_params(
            cur_target_columns=target_columns["cat"] + target_columns["con"],
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

    def _calc_uncertrainty_loss(self, name, loss_value):
        log_var = self.log_vars[name]
        scalar = 2.0 if name in self.target_columns["cat"] else 1.0

        precision = torch.exp(-log_var)
        loss = scalar * torch.sum(precision * loss_value + log_var)

        return loss

    def forward(self, inputs, targets):
        losses_dict = {}

        for target_column, criterion in self.criterions.items():
            cur_target_col_labels = targets[target_column]
            cur_target_col_outputs = inputs[target_column]
            loss_value_base = criterion(
                input=cur_target_col_outputs, target=cur_target_col_labels
            )
            loss_value_uncertain = self._calc_uncertrainty_loss(
                name=target_column, loss_value=loss_value_base
            )
            losses_dict[target_column] = loss_value_uncertain

        return losses_dict


def get_best_average_performance(
    val_metrics_files: Dict[str, Path], target_columns: "al_target_columns"
):
    df_performances = _get_overall_performance(
        val_metrics_files=val_metrics_files, target_columns=target_columns
    )
    best_performance = df_performances["Performance_Average"].max()

    return best_performance


def _get_overall_performance(
    val_metrics_files: Dict[str, Path], target_columns: "al_target_columns"
) -> pd.DataFrame:
    """
    With continuous columns, we use the distance the MSE is from 1 as "performance".
    """
    target_columns_gen = get_target_columns_generator(target_columns)

    df_performances = pd.DataFrame()
    for column_type, column_name in target_columns_gen:
        cur_metric_df = pd.read_csv(val_metrics_files[column_name])

        if column_type == "con":
            df_performances[column_name] = 1 - cur_metric_df[f"v_{column_name}_loss"]

        elif column_type == "cat":
            df_performances[column_name] = cur_metric_df[f"v_{column_name}_mcc"]

        else:
            raise ValueError()

    df_performances["Performance_Average"] = df_performances.mean(axis=1)

    return df_performances


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
        target_prefix=f"{prefixes['metrics']}",
    )

    if write_header:
        _ensure_metrics_paths_exists(metrics_files)

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
    target_columns: "al_target_columns", run_folder: Path, target_prefix: str
) -> Dict[str, Path]:
    all_target_columns = target_columns["con"] + target_columns["cat"]

    path_dict = {}
    for target_column in all_target_columns:
        cur_fname = target_prefix + target_column + "_history.log"
        cur_path = Path(run_folder, "results", target_column, cur_fname)
        path_dict[target_column] = cur_path

    average_loss_training_metrics_file = Path(
        run_folder, f"{target_prefix}average_history.log"
    )
    path_dict[f"{target_prefix}average"] = average_loss_training_metrics_file

    return path_dict


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
        results_dir / f"t_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"v_{target_string}_history.log"
    )

    return train_history_path, valid_history_path
