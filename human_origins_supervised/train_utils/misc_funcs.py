from typing import Dict

import torch
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef


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
    train_pred = outputs.detach().cpu().numpy()
    train_labels = labels.cpu().numpy()

    if len(train_pred) < 2:
        r = 0
    else:
        r = pearsonr(train_labels.squeeze(), train_pred.squeeze())[0]

    return {f"{prefix}_r": r}


def get_train_metrics(model_task):
    if model_task == "reg":
        return ["t_r"]
    elif model_task == "cls":
        return ["t_mcc"]
