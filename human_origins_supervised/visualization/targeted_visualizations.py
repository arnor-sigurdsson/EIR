from typing import TYPE_CHECKING, List
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from human_origins_supervised.train import Config


def get_target_vis(config: "Config"):
    if config.cl_args.label_column == "standing_height_f50_0_0":
        return generate_regression_prediction_plot

    return None


def split_based_on_labels(y_labels):
    unique_labels = np.unique(np.array(y_labels))
    groups_dict = {}

    for label in unique_labels:
        groups_dict[label] = np.array([i for i, _ in enumerate(y_labels) if i == label])

    return groups_dict


def generate_regression_prediction_plot(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    y_labels: List[str],
    encoder: StandardScaler,
    outfolder: Path,
    title_extra: str = "",
    *args,
    **kwargs,
):
    groups_dict = split_based_on_labels(y_labels)

    fig, ax = plt.subplots()
    y_true = encoder.inverse_transform(y_true.reshape(-1, 1))
    y_outp = encoder.inverse_transform(y_outp.reshape(-1, 1))

    for group, group_indices in groups_dict.items():
        counter = 0
        cur_y_true = y_true[group_indices]
        cur_y_outp = y_outp[group_indices]

        r2 = r2_score(cur_y_true, cur_y_outp)
        pcc = pearsonr(cur_y_true.squeeze(), cur_y_outp.squeeze())[0]

        ax.scatter(cur_y_outp, cur_y_true, edgecolors=(0, 0, 0), alpha=0.2, s=10)
        ax.text(
            x=0.05,
            y=0.95 - counter * 0.05,
            s=f"{group}: R2 = {r2:.2f}, PCC = {pcc:.2f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Measured")
    ax.set(title=title_extra)

    plt.tight_layout()
    plt.savefig(outfolder / "regression_predictions.png", dpi=200)
    plt.close()
