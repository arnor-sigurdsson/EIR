from pathlib import Path
from textwrap import wrap
from typing import List, Callable, Union, Tuple, TYPE_CHECKING, Dict

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.utils.multiclass import unique_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

from aislib.misc_utils import get_logger

from eir.train_utils import metrics

if TYPE_CHECKING:
    from eir.train_utils.evaluation import PerformancePlotConfig


logger = get_logger(name=__name__, tqdm_compatible=True)


def add_series_to_axis(
    ax_object: plt.Axes,
    series: pd.Series,
    skiprows: int,
    ax_plot_kwargs: Union[Dict, None] = None,
) -> plt.Axes:

    if ax_plot_kwargs is None:
        ax_plot_kwargs = {}
    series_cut = series[series.index > skiprows]

    xlim_upper = series_cut.shape[0] + skiprows
    xticks = np.arange(1 + skiprows, xlim_upper + 1)

    if ax_plot_kwargs is None:
        ax_plot_kwargs = {}

    ax_object.plot(
        xticks, series_cut.values, zorder=1, alpha=0.5, linewidth=0.8, **ax_plot_kwargs
    )

    # plt actually uses the '_' internally
    lines_to_legend = [
        line for line in ax_object.lines if not line.get_label().startswith("_")
    ]
    labels_to_legend = [line.get_label() for line in lines_to_legend]
    ax_object.legend(lines_to_legend, labels_to_legend)

    return ax_object


def generate_validation_curve_from_series(
    series: pd.Series, title_extra: str = "", skiprows: int = 200
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    valid_series_cut = series[series.index > skiprows]

    extreme_func = _get_min_or_max_funcs(valid_series_cut.name)
    extreme_valid_idx, extreme_valid_value = _get_validation_extreme_value_and_iter(
        extreme_func, valid_series_cut
    )

    xlim_upper = valid_series_cut.index.max()

    validation_xticks = valid_series_cut.index
    lines = ax.plot(
        validation_xticks,
        valid_series_cut.values,
        c="red",
        linewidth=0.8,
        alpha=1.0,
        label=f"Validation (best: {extreme_valid_value:.4g} at {extreme_valid_idx})",
        zorder=0,
    )

    ax.axhline(y=extreme_valid_value, linewidth=0.4, c="red", linestyle="dashed")

    ax.set(title=title_extra)

    ax.set_xlabel("Iteration")
    y_label = _parse_metrics_colname(valid_series_cut.name)
    ax.set_ylabel(y_label)

    ax.set_xlim(left=skiprows + 1, right=xlim_upper)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if xlim_upper > 1e4:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    plt.grid()

    return fig, ax


def _get_min_or_max_funcs(
    column_name: str,
) -> Union[pd.Series.idxmin, pd.Series.idxmax]:
    """
    The functions returned here will be explicitly called on a pd.Series instance.
    """

    func = pd.Series.idxmax
    metric = _parse_metrics_colname(column_name)
    if metric in ["LOSS", "RMSE", "LOSS-AVERAGE"]:
        return pd.Series.idxmin

    return func


def _parse_metrics_colname(column_name: str) -> str:

    return column_name.split("_")[-1].upper()


def _get_validation_extreme_value_and_iter(
    extreme_index_func: Union[np.argmax, np.argmin], validation_values: pd.Series
) -> Tuple[int, float]:

    extreme_index: int = extreme_index_func(validation_values)
    extreme_value: float = validation_values[extreme_index]

    return extreme_index, extreme_value


def gen_eval_graphs(plot_config: "PerformancePlotConfig"):
    """
    TODO:
        - Clean this function up – especially when it comes to target_transformers.
        - Use val_ids_total to hook into other labels for plotting.
    """

    pc = plot_config

    if pc.column_type == "cat":
        n_classes = len(pc.target_transformer.classes_)
        val_argmaxed = pc.val_outputs.argmax(axis=1)
        generate_confusion_matrix(
            y_true=pc.val_labels,
            y_outp=val_argmaxed,
            classes=pc.target_transformer.classes_,
            outfolder=pc.output_folder,
            title_extra=pc.column_name,
        )
    elif pc.column_type == "con":
        n_classes = None
    else:
        raise ValueError()

    plot_funcs = select_performance_curve_funcs(
        column_type=pc.column_type, n_classes=n_classes
    )

    for plot_func in plot_funcs:
        try:
            plot_func(
                y_true=pc.val_labels,
                y_outp=pc.val_outputs,
                outfolder=pc.output_folder,
                transformer=pc.target_transformer,
                title_extra=pc.column_name,
            )
        except Exception as e:
            logger.error(
                "Call to function %s resulted in error %s. No plot will be generated.",
                plot_func,
                e,
            )


def select_performance_curve_funcs(
    column_type: str, n_classes: int = None
) -> List[Callable]:
    if column_type == "cat":
        if not n_classes or n_classes < 2:
            raise ValueError("Expected number of classes to be not None and >2.")

        if n_classes == 2:
            return [
                generate_binary_roc_curve,
                generate_binary_pr_curve,
                generate_binary_prediction_distribution,
            ]
        else:
            return [generate_multi_class_roc_curve, generate_multi_class_pr_curve]
    elif column_type == "con":
        return [generate_regression_prediction_plot]


def generate_regression_prediction_plot(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: StandardScaler,
    outfolder: Path,
    title_extra: str = "",
    *args,
    **kwargs,
):
    fig, ax = plt.subplots()
    y_true = transformer.inverse_transform(y_true.reshape(-1, 1))
    y_outp = transformer.inverse_transform(y_outp.reshape(-1, 1))

    r2 = metrics.calc_r2(outputs=y_outp, labels=y_true)
    pcc = metrics.calc_pcc(outputs=y_outp, labels=y_true)

    ax.scatter(x=y_outp, y=y_true, edgecolors=(0, 0, 0), alpha=0.2, s=10)
    ax.text(
        x=0.05,
        y=0.95,
        s=f"R2 = {r2:.4g}, PCC = {pcc:.4g}",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Measured")
    ax.set(title=title_extra)

    plt.tight_layout()
    plt.savefig(outfolder / "regression_predictions.png", dpi=200)
    plt.close("all")


def generate_binary_roc_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    y_true_bin = label_binarize(y_true, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_true_bin, y_outp[:, 1])
    roc_auc = metrics.calc_roc_auc_ovr(outputs=y_outp, labels=y_true)

    plt.plot(fpr, tpr, lw=2, label=f"(area = {roc_auc:0.4g})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve – {title_extra}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 8})

    plt.tight_layout()
    plt.savefig(outfolder / "bin_roc_curve.png", dpi=200)
    plt.close("all")


def generate_binary_pr_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    y_true_bin = label_binarize(y_true, classes=[0, 1])
    precision, recall, _ = precision_recall_curve(y_true_bin, y_outp[:, 1])
    average_precision = metrics.calc_average_precision_ovr(
        outputs=y_outp, labels=y_true
    )

    plt.step(
        recall,
        precision,
        where="post",
        label=f"(area = {average_precision:0.4g})",
        lw=2,
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve – {title_extra}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 8})

    plt.tight_layout()
    plt.savefig(outfolder / "bin_pr_curve.png", dpi=200)
    plt.close("all")


def generate_binary_prediction_distribution(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    outfolder: Path,
    title_extra: str,
    transformer: LabelEncoder,
    *args,
    **kwargs,
):
    y_true_bin = label_binarize(y_true, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_true_bin, y_outp[:, 1])
    roc_auc = metrics.calc_roc_auc_ovr(outputs=y_outp, labels=y_true)

    classes = transformer.classes_
    fig, ax = plt.subplots()

    for class_index, class_name in zip(range(2), classes):
        cur_class_mask = np.argwhere(y_true == class_index)
        cur_probabilities = y_outp[cur_class_mask, 1]

        ax.hist(cur_probabilities, rwidth=0.90, label=class_name, alpha=0.5)

    ax.legend(loc="upper left")
    props = dict(boxstyle="round", facecolor="none", alpha=0.25, edgecolor="gray")
    ax.text(
        0.80,
        0.95,
        f"AUC: {roc_auc:0.4g}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"PGS of class {classes[1]}")
    ax.set_title(title_extra + " PGS")

    plt.tight_layout()
    plt.savefig(outfolder / "positive_prediction_distribution.png", dpi=200)
    plt.close("all")


def generate_multi_class_roc_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: LabelEncoder,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    unique_classes = sorted(transformer.classes_)
    n_classes = len(unique_classes)
    assert len(np.unique(y_true)) == n_classes

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_outp[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_outp.ravel())

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_macro = metrics.calc_roc_auc_ovr(
        outputs=y_outp, labels=y_true, average="macro"
    )
    roc_auc_micro = metrics.calc_roc_auc_ovr(
        outputs=y_outp, labels=y_true, average="micro"
    )

    plt.figure(figsize=(12, 8))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (area = {roc_auc_micro:0.4g})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (area = {roc_auc_macro:0.4g})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = iter(cm.tab20(np.arange(n_classes)))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{transformer.inverse_transform([i])[0]} "
            f"({np.count_nonzero(y_true == i)}) "
            f"(area = {roc_auc[i]:0.4g})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC curve – {title_extra}", fontsize=20)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(outfolder / "mc_roc_curve.png", dpi=200)
    plt.close("all")


def generate_multi_class_pr_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: LabelEncoder,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    precision = dict()
    recall = dict()

    unique_classes = sorted(transformer.classes_)
    n_classes = len(unique_classes)
    assert len(np.unique(y_true)) == n_classes

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_outp[:, i]
        )
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_outp[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_outp.ravel()
    )
    average_precision_micro = metrics.calc_average_precision_ovr(
        outputs=y_outp, labels=y_true, average="micro"
    )

    plt.figure(figsize=(12, 8))

    plt.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=2,
        label=f"Micro-Average Precision-Recall "
        f"(area = {average_precision_micro:0.4g})",
    )

    colors = iter(cm.tab20(np.arange(n_classes)))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"{transformer.inverse_transform([i])[0]} "
            f"({np.count_nonzero(y_true == i)}) "
            f"(area = {average_precision[i]:0.4g})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"Precision-Recall curve – {title_extra}", fontsize=20)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(outfolder / "mc_pr_curve.png", dpi=200)
    plt.close("all")


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    classes: List[str],
    outfolder: Path,
    normalize: bool = False,
    title_extra: str = "",
    cmap: plt = plt.cm.Blues,
):
    if not title_extra:
        if normalize:
            title_extra = "Normalized confusion matrix"
        else:
            title_extra = "Confusion matrix, without normalization"

    conf_mat = confusion_matrix(y_true, y_outp)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_outp)]
    classes_wrapped = ["\n".join(wrap(c, 20)) for c in classes]
    if normalize:
        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(conf_mat.shape[1]),
        yticks=np.arange(conf_mat.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes_wrapped,
        yticklabels=classes_wrapped,
        title=title_extra,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], fmt),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(outfolder / "confusion_matrix.png", dpi=200)
    plt.close("all")


def generate_all_training_curves(
    training_history_df: pd.DataFrame,
    valid_history_df: pd.DataFrame,
    output_folder: Path,
    plot_skip_steps: int,
    title_extra: str = "",
) -> None:

    if training_history_df.shape[0] <= plot_skip_steps:
        return

    metrics = ["_".join(i.split("_")[:]) for i in valid_history_df.columns]

    for metric_suffix in metrics:
        valid_series = valid_history_df[metric_suffix]

        figure_object, axis_object = generate_validation_curve_from_series(
            series=valid_series, title_extra=title_extra, skiprows=plot_skip_steps
        )

        if metric_suffix in training_history_df.columns:
            train_series = training_history_df[metric_suffix]
            _ = add_series_to_axis(
                ax_object=axis_object,
                series=train_series,
                skiprows=plot_skip_steps,
                ax_plot_kwargs={"label": "Train", "c": "orange"},
            )

        fname_identifier = _parse_metrics_colname(column_name=valid_series.name)
        figure_object.savefig(
            output_folder / f"training_curve_{fname_identifier}.png", dpi=200
        )
        plt.close("all")
