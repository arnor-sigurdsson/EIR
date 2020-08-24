from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm
import plotly.graph_objects as go

from aislib.misc_utils import get_logger, ensure_path_exists

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.activation_analysis import (
        al_gradients_dict,
        al_top_gradients_dict,
        al_scaled_grads_dict,
    )

logger = get_logger(name=__name__, tqdm_compatible=True)


def generate_snp_gradient_matrix(
    snp_df: pd.DataFrame, snp_indexes: np.ndarray
) -> np.ndarray:
    """
    TODO:
        Make this robust for .bim files later. Here we are basically mapping between
        the one hot coordinates and whether we are pulling out a reference or
        alternative (minor, A1) allele from the .bim / .snp file.
        Note that plink uses: <Variant ID>_<counted allele> in the .raw file, where
        'By default, A1 alleles are counted; this can be customized with
        --recode-allele.'

    TODO:
        Check the 'forms' of the recoded file and what each number there corresponds
        to w.r.t REF / HET / ALT. The columns positions in the .bim file are well
        defined, so we just have to make sure we are mapping correctly between the
        one-hot and genotype files (i.e. what exactly does [1,0,0,0] map to).
        See: https://www.cog-genomics.org/plink/1.9/formats#bim
        See: https://www.cog-genomics.org/plink/2.0/formats#bim
        See: http://zzz.bwh.harvard.edu/plink/dataman.shtml#recode
        'By default, the minor allele is assigned to be A1.'

    NOTE:
        Plink counts alternative (minor) alleles, while eigenstrat counts reference
        alleles. For now we are going to write this with additive format in mind (i.e
        0 = 0 copies of alternative (i.e. 2 copies of reference), 2 = 2 copies of
        alternative).

    NOTE:
        What we flag as "alternative" and "reference" can in theory be somewhat
        arbitrary, as long as we are grabbing the correct nucleotides.
    """
    snp_matrix = np.zeros((4, len(snp_indexes)), dtype=object)

    for matrix_col, snp_idx in enumerate(snp_indexes):
        cur_row = snp_df.iloc[snp_idx]
        snp_matrix[0, matrix_col] = cur_row.REF * 2  # 0 alt alleles counted
        snp_matrix[1, matrix_col] = cur_row.REF + cur_row.ALT  # 1 alt alleles counted
        snp_matrix[2, matrix_col] = cur_row.ALT * 2  # 2 alt alleles counted
        snp_matrix[3, matrix_col] = ""  # missing
    return snp_matrix


def plot_top_gradients(
    gathered_scaled_grads: "al_scaled_grads_dict",
    top_gradients_dict: "al_top_gradients_dict",
    snp_df: pd.DataFrame,
    output_folder: Path,
    fname: str = "top_snps.png",
    custom_ylabel: str = None,
):
    n_cls = len(top_gradients_dict.keys())
    classes = sorted(list(top_gradients_dict.keys()))

    fig = plt.figure(figsize=(n_cls * 4, n_cls * 2 + 1))
    gs = gridspec.GridSpec(n_cls, n_cls, wspace=0.2, hspace=0.2)

    snp_names = snp_df["VAR_ID"].array
    for grad_idx, col_name in enumerate(classes):
        cls_top_idxs = top_gradients_dict[col_name]["top_n_idxs"]
        cur_snp_matrix = generate_snp_gradient_matrix(snp_df, cls_top_idxs)

        for cls_idx, row_name in enumerate(classes):

            cur_grads = gathered_scaled_grads[col_name][row_name]

            cur_ax = plt.subplot(gs[cls_idx, grad_idx])
            cur_ax.imshow(cur_grads, vmin=0, vmax=1)

            ylabel = row_name if not custom_ylabel else custom_ylabel
            cur_ax.set_ylabel(ylabel)

            if cls_idx == n_cls - 1:
                top_snp_names = snp_names[cls_top_idxs]
                cur_ax.set_xticks(np.arange(len(top_snp_names)))
                cur_ax.set_xticklabels(top_snp_names)
                plt.setp(cur_ax.get_xticklabels(), rotation=90, ha="center")
            else:
                cur_ax.set_xticklabels([])
                cur_ax.set_xticks([])

            cur_ax.set_yticks(np.arange(4))
            cur_ax.set_yticklabels(["REF", "HET", "ALT", "MIS"])

            for snp_form in range(cur_snp_matrix.shape[0]):
                for snp_id in range(cur_snp_matrix.shape[1]):
                    cur_ax.text(
                        snp_id,
                        snp_form,
                        cur_snp_matrix[snp_form, snp_id],
                        ha="center",
                        va="center",
                        color="w",
                    )

    axs = fig.get_axes()
    for ax, col_title in zip(axs[::n_cls], classes):
        ax.set_title(f"Top {col_title} SNPs")

    for ax in axs:
        ax.label_outer()

    # plt.tight_layout gives a warning here, seems to be gs specific behavior
    gs.tight_layout(fig)
    plt.savefig(output_folder / fname, bbox_inches="tight")
    plt.close("all")


def plot_snp_manhattan_plots(
    df_snp_grads: pd.DataFrame, outfolder: Path, title_extra: str = ""
):
    activations_columns = [
        i for i in df_snp_grads.columns if i.endswith("_activations")
    ]

    for col in activations_columns:
        label_name = col.split("_activations")[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            x=df_snp_grads["BP_COORD"],
            y=df_snp_grads[col],
            label=label_name,
            color="black",
            marker=".",
        )
        ax.set_ylim(ymin=0.0)

        ax.set_xlabel("BP Coordinate")
        ax.set_ylabel("Activation")

        plt.title(f"Manhattan Plot{title_extra}")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        out_path = outfolder / f"activations/manhattan/{label_name}_manhattan.png"
        ensure_path_exists(path=out_path)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")


def plot_snp_manhattan_plots_plotly(
    df_snp_grads: pd.DataFrame,
    outfolder: Path,
    quantile: float = 0.95,
    title_extra: str = "",
):
    """
    We have the percentile here to avoid creating html files that are massive in size.
    """
    activations_columns = [
        i for i in df_snp_grads.columns if i.endswith("_activations")
    ]

    fig = go.Figure()

    for col in activations_columns:
        label_name = col.split("_activations")[0]
        quantile_cutoff = df_snp_grads[col].quantile(quantile)
        df_snp_grads_cut = df_snp_grads[df_snp_grads[col] >= quantile_cutoff]

        fig.add_trace(
            go.Scatter(
                x=df_snp_grads_cut["BP_COORD"],
                y=df_snp_grads_cut[col],
                name=label_name,
                mode="markers",
                text=df_snp_grads_cut["VAR_ID"],
            )
        )

    fig.update_layout(
        title=f"Manhattan Plot (top {quantile}%){title_extra}",
        xaxis_title="BP Coordinate",
        yaxis_title="Activation",
    )

    out_path = outfolder / f"activations/manhattan/manhattan_interactive.html"
    ensure_path_exists(path=out_path)
    fig.write_html(str(out_path))
