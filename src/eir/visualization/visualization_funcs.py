from collections.abc import Callable
from pathlib import Path
from textwrap import wrap
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
)

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.multiclass import unique_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from eir.train_utils import metrics
from eir.utils.logging import get_logger
from eir.visualization.style import COLORS, get_class_visuals

if TYPE_CHECKING:
    from eir.train_utils.evaluation import PerformancePlotConfig

logger = get_logger(name=__name__, tqdm_compatible=True)


def add_series_to_axis(
    ax_object: plt.Axes,
    series: pd.Series,
    skiprows: int,
    ax_plot_kwargs: dict | None = None,
) -> plt.Axes:
    ax_plot_kwargs = ax_plot_kwargs or {}
    series_cut = series[series.index > skiprows]

    xlim_upper = series_cut.shape[0] + skiprows
    xticks = np.arange(1 + skiprows, xlim_upper + 1)

    if "label" in ax_plot_kwargs and ax_plot_kwargs["label"] == "Train":
        defaults = {
            "color": COLORS["secondary"],
            "alpha": 0.7,
            "linewidth": 1.0,
            "zorder": 2,
        }
        for k, v in defaults.items():
            ax_plot_kwargs.setdefault(k, v)

    ax_object.plot(
        xticks,
        np.asarray(series_cut.values),
        **ax_plot_kwargs,
    )

    ax_object.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        fancybox=True,
    )

    return ax_object


def generate_validation_curve_from_series(
    series: pd.Series,
    title_extra: str = "",
    skiprows: int = 200,
) -> tuple[plt.Figure | None, plt.Axes | None]:
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_series_cut = series[series.index > skiprows]

    extreme_func = _get_min_or_max_funcs(column_name=str(valid_series_cut.name))
    extreme_valid_idx, extreme_valid_value = _get_validation_extreme_value_and_iter(
        extreme_index_func=extreme_func, validation_values=valid_series_cut
    )

    if extreme_valid_idx is None or extreme_valid_value is None:
        logger.warning(
            "No valid extreme value found for series %s. Skipping plot generation.",
            valid_series_cut.name,
        )
        return None, None

    values = np.asarray(valid_series_cut.values)

    validation_xticks = valid_series_cut.index
    lines = ax.plot(
        validation_xticks,
        values,
        color=COLORS["primary"],
        linewidth=1.35,
        alpha=1.0,
        label=f"Validation (best: {extreme_valid_value:.4g} at {extreme_valid_idx})",
        zorder=3,
    )

    ax.axhline(
        y=extreme_valid_value,
        linewidth=0.8,
        color=COLORS["primary"],
        linestyle="dashed",
        alpha=0.7,
        zorder=2,
    )

    ax.set_xlabel("Iteration")
    y_label = _parse_metrics_colname(column_name=str(valid_series_cut.name))
    ax.set_ylabel(y_label)

    xlim_upper = valid_series_cut.index.max()
    ax.set_xlim(left=skiprows + 1, right=xlim_upper)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if xlim_upper > 1e4:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    name_lower = str(valid_series_cut.name).lower()
    enforce_limits_matches = ["auc", "ap-macro", "mcc", "acc", "pcc", "r2"]
    should_enforce = any(i in name_lower for i in enforce_limits_matches)
    if should_enforce:
        bottom: float | None = 0.0
        if name_lower in ("pcc", "r2"):
            bottom = None
        ax.set_ylim(bottom, 1.00)

    ax.grid(True, linestyle="--", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels)
    ax.set_title(f"{title_extra}", fontweight="bold")

    return fig, ax


class SeriesMinMaxProtocol(Protocol):
    def __call__(self, series: pd.Series) -> int | str: ...


def _get_min_or_max_funcs(
    column_name: str,
) -> SeriesMinMaxProtocol:
    """
    The functions returned here will be explicitly called on a pd.Series instance.
    """

    func: SeriesMinMaxProtocol = pd.Series.idxmax  # type: ignore
    metric = _parse_metrics_colname(column_name=str(column_name))
    if metric in [
        "LOSS",
        "RMSE",
        "IBS",
        "LOSS-AVERAGE",
    ]:
        return pd.Series.idxmin  # type: ignore

    return func


def _parse_metrics_colname(column_name: str) -> str:
    return column_name.split("_")[-1].upper()


def _get_validation_extreme_value_and_iter(
    extreme_index_func: Callable,
    validation_values: pd.Series,
) -> tuple[int | None, float | None]:
    """
    For sparse targets, we might get e.g. for ROC AUC:

        iteration
        200   NaN

    """
    extreme_index: int = extreme_index_func(validation_values)

    if np.isnan(extreme_index):
        return None, None

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
                "Call to function %s resulted in error: '%s'. "
                "No plot will be generated.",
                plot_func,
                e,
            )


def select_performance_curve_funcs(
    column_type: str, n_classes: int | None = None
) -> list[Callable]:
    if column_type == "cat":
        if not n_classes or n_classes < 2:
            raise ValueError("Expected number of classes to be not None and >2.")

        if n_classes == 2:
            return [
                generate_binary_roc_curve,
                generate_binary_pr_curve,
                generate_binary_prediction_distribution,
            ]
        return [generate_multi_class_roc_curve, generate_multi_class_pr_curve]
    if column_type == "con":
        return [generate_regression_prediction_plot]

    raise ValueError(f"Unknown column type {column_type}.")


def generate_regression_prediction_plot(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: StandardScaler,
    outfolder: Path,
    title_extra: str = "",
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(4, 4))

    y_true = transformer.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_outp = transformer.inverse_transform(y_outp.reshape(-1, 1)).flatten()

    r2 = metrics.calc_r2(outputs=y_outp, labels=y_true)
    pcc = metrics.calc_pcc(outputs=y_outp, labels=y_true)

    ax.scatter(
        x=y_outp,
        y=y_true,
        color=COLORS["primary"],
        alpha=0.4,
        s=12,
        edgecolors=None,
        marker="o",
        label="Data points",
    )

    min_val = min(y_true.min(), y_outp.min())
    max_val = max(y_true.max(), y_outp.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color=COLORS["dark_gray"],
        linestyle="--",
        linewidth=1.2,
        label="Perfect prediction",
    )

    metrics_text = f"$R^2 = {r2:.3f}$\n$\\mathrm{{PCC}} = {pcc:.3f}$"

    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": COLORS["light_gray"],
            "pad": 0.5,
        },
    )

    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Measured value")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, framealpha=0.9, loc="lower right")

    if title_extra:
        ax.set_title(title_extra)

    plt.savefig(outfolder / "regression_predictions.pdf", bbox_inches="tight")

    plt.close("all")

    return fig, ax


def generate_binary_roc_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    y_true_bin = label_binarize(y_true, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_true_bin, y_outp[:, 1])
    roc_auc = metrics.calc_roc_auc_ovo(outputs=y_outp, labels=y_true)

    ax.plot(
        fpr,
        tpr,
        lw=2,
        color=COLORS["primary"],
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1.5,
        color=COLORS["dark_gray"],
        label="Random classifier",
    )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title_extra:
        ax.set_title(f"ROC Curve – {title_extra}")

    ax.legend(loc="lower right", frameon=True, framealpha=0.8)

    plt.savefig(outfolder / "bin_roc_curve.pdf", bbox_inches="tight")
    plt.close("all")

    return fig, ax


def generate_binary_pr_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    y_true_bin = label_binarize(y_true, classes=[0, 1])
    precision, recall, _ = precision_recall_curve(y_true_bin, y_outp[:, 1])
    average_precision = metrics.calc_average_precision(outputs=y_outp, labels=y_true)

    ax.step(
        recall,
        precision,
        where="post",
        color=COLORS["primary"],
        lw=2,
        label=f"PR curve (AP = {average_precision:.3f})",
    )

    no_skill = len(y_true_bin[y_true_bin == 1]) / len(y_true_bin)
    ax.plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        color=COLORS["dark_gray"],
        lw=1.5,
        label="Random classifier",
    )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title_extra:
        ax.set_title(f"Precision-Recall Curve – {title_extra}")

    ax.legend(loc="lower left", frameon=True, framealpha=0.8)

    plt.savefig(outfolder / "bin_pr_curve.pdf", bbox_inches="tight")
    plt.close("all")

    return fig, ax


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
    roc_auc = metrics.calc_roc_auc_ovo(outputs=y_outp, labels=y_true)

    classes = transformer.classes_
    fig, ax = plt.subplots(figsize=(8, 5))

    class_colors = [COLORS["primary"], COLORS["secondary"]]

    for class_index, class_name in zip(range(2), classes, strict=False):
        cur_class_mask = np.argwhere(y_true == class_index)
        cur_probabilities = y_outp[cur_class_mask, 1].flatten()

        ax.hist(
            cur_probabilities,
            rwidth=0.90,
            label=f"{class_name} (n={len(cur_probabilities)})",
            alpha=0.6,
            color=class_colors[class_index],
            edgecolor="black",
            linewidth=0.75,
        )

    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    props = {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.8,
        "edgecolor": "gray",
    }
    ax.text(
        0.95,
        0.95,
        f"AUC: {roc_auc:0.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )

    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"Score of class '{classes[1]}'")
    ax.set_title(f"{title_extra} Score Distribution", fontweight="bold")

    plt.savefig(outfolder / "positive_prediction_distribution.pdf")
    plt.close("all")

    return fig, ax


def generate_multi_class_roc_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: LabelEncoder,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    fpr: dict[int | str, np.ndarray] = {}
    tpr: dict[int | str, np.ndarray] = {}
    roc_auc: dict[int, float] = {}

    unique_classes = sorted(transformer.classes_)
    n_classes = len(unique_classes)
    if len(np.unique(y_true)) != n_classes:
        raise ValueError(
            f"Expected {n_classes} unique classes when plotting multiclass "
            f"ROC-AUC curve, got {np.unique(y_true)}."
        )

    y_true_bin = label_binarize(y=y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true=y_true_bin[:, i], y_score=y_outp[:, i])
        roc_auc[i] = auc(x=fpr[i], y=tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true=y_true_bin.ravel(), y_score=y_outp.ravel()
    )

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_macro = metrics.calc_roc_auc_ovo(
        outputs=y_outp, labels=y_true, average="macro"
    )

    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average (AUC = {roc_auc_macro:.3f})",
        color=COLORS["accent"],
        linestyle=":",
        linewidth=2.5,
    )

    colors, line_styles = get_class_visuals(n_classes=n_classes)

    for i in range(n_classes):
        class_name = str(transformer.inverse_transform([i])[0])
        class_count = np.count_nonzero(y_true == i)

        ax.plot(
            fpr[i],
            tpr[i],
            color=colors[i],
            linestyle=line_styles[i],
            lw=1.5,
            label=f"{class_name} (n={class_count}, AUC={roc_auc[i]:.3f})",
        )

    ax.plot([0, 1], [0, 1], color=COLORS["dark_gray"], linestyle="--", lw=1.5)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    if title_extra:
        ax.set_title(f"ROC Curves – {title_extra}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower left" if n_classes <= 8 else "center left",
        frameon=True,
        framealpha=0.9,
        fontsize=8 if n_classes > 6 else 9,
        title_fontsize=9,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(COLORS["light_gray"])

    plt.savefig(outfolder / "mc_roc_curve.pdf", bbox_inches="tight")
    plt.close("all")

    return fig, ax


def generate_multi_class_pr_curve(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    transformer: LabelEncoder,
    outfolder: Path,
    title_extra: str,
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    precision: dict[int | str, np.ndarray] = {}
    recall: dict[int | str, np.ndarray] = {}

    unique_classes = sorted(transformer.classes_)
    n_classes = len(unique_classes)

    if len(np.unique(y_true)) != n_classes:
        raise ValueError(
            f"Expected {n_classes} unique classes when plotting multiclass"
            f" PR curve, got {np.unique(y_true)}."
        )

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    average_precision = {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_outp[:, i]
        )
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_outp[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_outp.ravel()
    )
    average_precision_micro = metrics.calc_average_precision(
        outputs=y_outp, labels=y_true, average="micro"
    )

    ax.plot(
        recall["micro"],
        precision["micro"],
        color=COLORS["accent"],
        lw=2.5,
        linestyle=":",
        label=f"Micro-average (AP = {average_precision_micro:.3f})",
    )

    colors, line_styles = get_class_visuals(n_classes=n_classes)

    for i in range(n_classes):
        class_name = str(transformer.inverse_transform([i])[0])
        class_count = np.count_nonzero(y_true == i)

        ax.plot(
            recall[i],
            precision[i],
            color=colors[i],
            linestyle=line_styles[i],
            lw=1.5,
            label=f"{class_name} (n={class_count}, AP={average_precision[i]:.3f})",
        )

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    if title_extra:
        ax.set_title(f"Precision-Recall Curves – {title_extra}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower left" if n_classes <= 8 else "center left",
        frameon=True,
        framealpha=0.9,
        fontsize=8 if n_classes > 6 else 9,
        title_fontsize=9,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(COLORS["light_gray"])

    plt.savefig(outfolder / "mc_pr_curve.pdf", bbox_inches="tight")
    plt.close("all")

    return fig, ax


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    classes: list[str],
    outfolder: Path,
    normalize: Literal["true", "pred", "all", None] = None,
    title_extra: str = "",
    cmap: matplotlib.colors.Colormap | None = None,
):
    if cmap is None:
        cmap = sns.color_palette("rocket", as_cmap=True)

    if not title_extra:
        title_extra = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"

    conf_mat = confusion_matrix(
        y_true=y_true,
        y_pred=y_outp,
        normalize=normalize,
    )

    classes = classes[unique_labels(y_true, y_outp)]

    wrap_length = max(20, min(20, 50 // len(classes)))
    classes_wrapped = ["\n".join(wrap(c, wrap_length)) for c in classes]

    df_cm = pd.DataFrame(conf_mat, index=classes_wrapped, columns=classes_wrapped)

    width = max(6, min(16, len(classes) * 1.0))
    height = max(5, min(14, len(classes) * 0.9))
    fig, ax = plt.subplots(figsize=(width, height))

    tick_label_font_size = max(6, min(12, 18 - len(classes) * 0.5))
    annot_font_size = max(7, min(14, 18 - len(classes) * 0.4))

    fmt = ".1%" if normalize else "d"

    sns.heatmap(
        data=df_cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        square=True,
        annot_kws={"fontsize": annot_font_size},
        vmin=0,
        vmax=1.0 if normalize else None,
    )

    ax.set_title(
        f"{title_extra}",
        fontsize=annot_font_size + 2,
        fontweight="bold",
    )
    ax.set_ylabel("True Label", fontsize=annot_font_size + 1, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=annot_font_size + 1, fontweight="bold")

    color_bar = ax.collections[0].colorbar
    assert color_bar is not None
    color_bar.ax.tick_params(labelsize=tick_label_font_size)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_label_font_size,
    )
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_label_font_size)

    for i in range(len(classes)):
        ax.add_patch(
            plt.Rectangle(
                (i, i),
                1,
                1,
                fill=False,
                edgecolor="white",
                lw=1.5,
            )
        )

    plt.savefig(
        outfolder / "confusion_matrix.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    return fig, ax


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
            series=valid_series,
            title_extra=title_extra,
            skiprows=plot_skip_steps,
        )

        if figure_object is None:
            continue

        assert axis_object is not None

        if metric_suffix in training_history_df.columns:
            train_series = training_history_df[metric_suffix]
            _ = add_series_to_axis(
                ax_object=axis_object,
                series=train_series,
                skiprows=plot_skip_steps,
                ax_plot_kwargs={"label": "Train"},
            )

        fname_identifier = _parse_metrics_colname(column_name=str(valid_series.name))
        figure_object.savefig(output_folder / f"training_curve_{fname_identifier}.pdf")
        plt.close("all")
