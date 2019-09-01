from pathlib import Path
from typing import List, Callable
from itertools import groupby

import matplotlib
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator


def generate_training_curve(
    metrics_filename: Path,
    skiprows: int = 0,
    cols: tuple = ("t_mcc", "v_mcc"),
    hook_funcs=None,
) -> None:

    df = pd.read_csv(metrics_filename, usecols=cols)
    df_cut = df.iloc[skiprows:, :]

    fig, ax_1 = plt.subplots()

    xlim_upper = df_cut.shape[0] + skiprows
    xticks = np.arange(1, xlim_upper + 1)
    line_1a = ax_1.plot(xticks, df_cut[cols[0]], c="orange", label="Train", zorder=1)
    line_1b = ax_1.plot(
        xticks,
        df_cut[cols[1]],
        c="red",
        linewidth=0.5,
        alpha=0.5,
        label="Validation",
        zorder=0,
    )

    ax_1.set_xlabel("Epoch")
    y_label = cols[0].split("_")[-1].upper()
    ax_1.set_ylabel(y_label)

    ax_1.set_xlim(left=skiprows + 1, right=xlim_upper)
    ax_1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_1.set_ylim(top=1, bottom=0)

    lines = line_1a + line_1b
    labels = [l.get_label() for l in lines]
    ax_1.legend(lines, labels)

    if hook_funcs:
        for func in hook_funcs:
            func(ax_1)

    plt.grid()

    run_folder = Path(metrics_filename).parent
    plt.savefig(run_folder / f"training_curve_{y_label}.png", dpi=200)
    plt.close()


def generate_basic_curve(metrics_filename, column_name, fname, skiprows=0):
    df = pd.read_csv(metrics_filename, usecols=[column_name])
    df = df.iloc[skiprows:, :]

    fig, ax = plt.subplots()
    ax.plot(df[column_name], c="navy")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_xlim(left=skiprows, right=df.shape[0] + skiprows)
    ax.set_xlim(left=skiprows, right=df.shape[0] + skiprows)

    run_folder = Path(metrics_filename).parent
    plt.savefig(run_folder / fname, dpi=200)
    plt.close()


def select_performance_curve_funcs(
    model_task: str, n_classes: int = None
) -> List[Callable]:
    if model_task == "cls":
        if not n_classes or n_classes < 2:
            raise ValueError("Expected number of classes to be not none and  >2.")

        if n_classes == 2:
            return [generate_binary_roc_curve, generate_binary_pr_curve]
        else:
            return [generate_multi_class_roc_curve, generate_multi_class_pr_curve]
    elif model_task == "reg":
        return []


def gen_eval_graphs(
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    outfolder: Path,
    label_encoder,
    model_task,
):
    """
    TODO: Clean this function up – expecially when it comes to label_encoder.
    """
    if model_task == "cls":
        n_classes = len(label_encoder.classes_)
        val_pred = val_probabilities.argmax(axis=1)
        generate_confusion_matrix(
            val_labels, val_pred, label_encoder.classes_, outfolder
        )
    else:
        n_classes = None

    plot_funcs = select_performance_curve_funcs(model_task, n_classes)
    for plot_func in plot_funcs:
        plot_func(
            y_test=val_labels,
            y_score=val_probabilities,
            outfolder=outfolder,
            label_encoder=label_encoder,
        )


def generate_binary_roc_curve(y_test, y_score, outfolder, *args, **kwargs):
    y_test_bin = label_binarize(y_test, classes=[0, 1])
    fpr, tpr, _ = roc_curve(y_test_bin, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"(area = {roc_auc:0.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 8})

    plt.tight_layout()
    plt.savefig(outfolder / "bin_roc_curve.png", dpi=200)
    plt.close()


def generate_binary_pr_curve(y_test, y_score, outfolder, *args, **kwargs):
    y_test_bin = label_binarize(y_test, classes=[0, 1])
    precision, recall, _ = precision_recall_curve(y_test_bin, y_score[:, 1])
    average_precision = average_precision_score(y_test_bin, y_score[:, 1])

    plt.step(
        recall,
        precision,
        where="post",
        label=f"(area = {average_precision:0.2f})",
        lw=2,
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 8})

    plt.tight_layout()
    plt.savefig(outfolder / "bin_pr_curve.png", dpi=200)
    plt.close()


def generate_multi_class_roc_curve(
    y_test, y_score: np.array, label_encoder, outfolder: Path
):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    unique_classes = sorted(label_encoder.classes_)
    n_classes = len(unique_classes)
    assert len(np.unique(y_test)) == n_classes

    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(12, 8))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
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
            label=f"{label_encoder.inverse_transform([i])[0]} "
            f"({np.count_nonzero(y_test == i)}) "
            f"(area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC curve", fontsize=20)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(outfolder / "mc_roc_curve.png", dpi=200)
    plt.close()


def generate_multi_class_pr_curve(
    y_test, y_score: np.array, label_encoder, outfolder: Path
):
    precision = dict()
    recall = dict()

    unique_classes = sorted(label_encoder.classes_)
    n_classes = len(unique_classes)
    assert len(np.unique(y_test)) == n_classes

    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], y_score[:, i]
        )
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_bin.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test_bin, y_score, average="micro"
    )
    plt.figure(figsize=(12, 8))

    plt.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=2,
        label=f"Micro-Average Precision-Recall "
        f'(area = {average_precision["micro"]:0.2f})',
    )

    colors = iter(cm.tab20(np.arange(n_classes)))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label=f"{label_encoder.inverse_transform([i])[0]} "
            f"({np.count_nonzero(y_test == i)}) "
            f"(area = {average_precision[i]:0.2f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall curve", fontsize=20)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(outfolder / "mc_pr_curve.png", dpi=200)
    plt.close()


def generate_confusion_matrix(
    y_true, y_pred, classes, outfolder, normalize=False, title=None, cmap=plt.cm.Blues
):
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    conf_mat = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
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
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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
    plt.close()


def generate_all_plots(metrics_file: Path, hook_funcs):
    def get_column_pairs():
        with open(str(metrics_file), "r") as infile:
            header_list = infile.readline().strip().split(",")
            header_list.sort(key=lambda x: x.split("_")[-1])

            header_pairs = [
                tuple(v)
                for k, v in groupby(header_list, key=lambda x: x.split("_")[-1])
            ]

            return header_pairs

    try:
        column_pairs = get_column_pairs()
        for pair in column_pairs:
            generate_training_curve(metrics_file, hook_funcs=hook_funcs, cols=pair)
    except ValueError:
        print(f"Skipping {metrics_file} as it has old columns.")
    plt.close("all")


def regenerate_all_run_plots(run_folder: Path):
    metrics_fname = "training_history.log"

    for run_folder in run_folder.iterdir():
        if run_folder.is_dir() and not run_folder.name.startswith("."):

            if run_folder.name != "previous_runs":
                generate_all_plots(run_folder / metrics_fname)

            else:
                for previous_run_collection in run_folder.iterdir():
                    if previous_run_collection.name.startswith("v"):
                        regenerate_all_run_plots(previous_run_collection)
