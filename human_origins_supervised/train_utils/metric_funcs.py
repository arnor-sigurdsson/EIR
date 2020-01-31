from functools import partial
from typing import Dict, Union, TYPE_CHECKING

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr

from human_origins_supervised.data_load.data_utils import get_target_columns_generator

if TYPE_CHECKING:
    from human_origins_supervised.train import al_criterions


def calculate_batch_metrics(
    target_columns, target_transformers, outputs, labels, prefix: str
) -> Dict:
    target_columns_gen = get_target_columns_generator(target_columns)

    master_metric_dict = {}
    for column_type, column_name in target_columns_gen:

        metric_func = select_metric_func(column_type, target_transformers[column_name])
        cur_outputs = outputs[column_name]
        cur_labels = labels[column_name]

        cur_metric_dict = metric_func(
            outputs=cur_outputs, labels=cur_labels, prefix=f"{prefix}_{column_name}"
        )

        master_metric_dict[column_name] = cur_metric_dict

    return master_metric_dict


def select_metric_func(
    target_column_type: str, target_transformer: Union[StandardScaler, LabelEncoder]
):
    if target_column_type == "cat":
        return calc_multiclass_metrics

    return partial(calc_regression_metrics, target_transformer=target_transformer)


def get_train_metrics(model_task, prefix="t"):
    if model_task == "reg":
        return [f"{prefix}_r2", f"{prefix}_rmse", f"{prefix}_pcc"]
    elif model_task == "cls":
        return [f"{prefix}_mcc"]


def calc_multiclass_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:

    _, pred = torch.max(outputs, 1)

    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

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
        pcc = pearsonr(labels, preds)[0]
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
