from functools import partial
from typing import Dict

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


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


def select_metric_func(model_task: str, target_transformer: StandardScaler):
    if model_task == "cls":
        return calc_multiclass_metrics

    return partial(calc_regression_metrics, target_transformer=target_transformer)


def get_train_metrics(model_task, prefix="t"):
    if model_task == "reg":
        return [f"{prefix}_r2", f"{prefix}_rmse", f"{prefix}_pcc"]
    elif model_task == "cls":
        return [f"{prefix}_mcc"]
