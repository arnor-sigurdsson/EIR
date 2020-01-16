from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

from aislib.misc_utils import get_logger

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
    accumulated_grads: Dict,
    top_gradients_dict: Dict,
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

            cur_grads = accumulated_grads[col_name][row_name]

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
    plt.close()


def plot_snp_gradients(accumulated_grads, outfolder, type_="avg"):
    n_classes = len(accumulated_grads.keys())

    colors = iter(cm.tab20(np.arange(n_classes)))

    fig, ax = plt.subplots(figsize=(12, 6))

    out_path = outfolder / f"snp_grads_per_class_{type_}.png"

    for label, grads in accumulated_grads.items():
        if grads:
            color = next(colors)

            if type_ == "avg":
                grads_averaged = np.array(grads).mean(0).sum(0)

                ax.plot(grads_averaged, label=label, color=color, lw=0.5)

            elif type_ == "single":
                for idx, single_grad in enumerate(grads):
                    single_grad_sum = single_grad.sum(0)

                    ax.plot(
                        single_grad_sum,
                        color=color,
                        label=label if idx == 0 else "",
                        lw=0.5,
                    )
        else:
            logger.warning(
                "No gradients aggregated for class %s due to no "
                "correct predictions for the class, gradient "
                "will not be plotted in line plot (%s).",
                label,
                out_path,
            )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
