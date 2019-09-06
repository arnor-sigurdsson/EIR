from typing import Dict

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error


def calc_multiclass_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:
    _, pred = torch.max(outputs, 1)

    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

    mcc = matthews_corrcoef(labels, pred)

    return {f"{prefix}_mcc": mcc}


def calc_regression_metrics(
    outputs: torch.Tensor, labels: torch.Tensor, prefix: str
) -> Dict[str, float]:
    preds = outputs.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    r2 = r2_score(y_true=labels, y_pred=preds)
    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))

    return {f"{prefix}_r2": r2, f"{prefix}_rmse": rmse}


def select_metric_func(model_task: str):
    if model_task == "cls":
        return calc_multiclass_metrics

    return calc_regression_metrics


def get_train_metrics(model_task):
    if model_task == "reg":
        return ["t_r2", "t_rmse"]
    elif model_task == "cls":
        return ["t_mcc"]
