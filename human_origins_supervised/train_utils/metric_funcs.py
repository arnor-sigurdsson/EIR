from functools import partial
from typing import Dict

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


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
    label_encoder: StandardScaler,
) -> Dict[str, float]:
    preds = outputs.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    labels = label_encoder.inverse_transform(labels).squeeze()
    preds = label_encoder.inverse_transform(preds).squeeze()

    if len(preds) < 2:
        r2 = 0
    else:
        r2 = r2_score(y_true=labels, y_pred=preds)
    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))

    return {f"{prefix}_r2": r2, f"{prefix}_rmse": rmse}


def select_metric_func(model_task: str, label_encoder: StandardScaler):
    if model_task == "cls":
        return calc_multiclass_metrics

    return partial(calc_regression_metrics, label_encoder=label_encoder)


def get_train_metrics(model_task, prefix="t"):
    if model_task == "reg":
        return [f"{prefix}_r2", f"{prefix}_rmse"]
    elif model_task == "cls":
        return [f"{prefix}_mcc"]
