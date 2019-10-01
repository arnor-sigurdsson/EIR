from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

from aislib.misc_utils import get_logger

logger = get_logger(__name__)


def plot_top_gradients(
    accumulated_grads: Dict,
    top_gradients_dict: Dict,
    snp_names: np.array,
    output_folder: Path,
    fname="top_snps.png",
    custom_ylabel=None,
):
    n_cls = len(top_gradients_dict.keys())
    classes = sorted(list(top_gradients_dict.keys()))

    fig = plt.figure(figsize=(n_cls * 4, n_cls * 2 + 1))
    gs = gridspec.GridSpec(n_cls, n_cls, wspace=0.2, hspace=0.2)

    for grad_idx, col_name in enumerate(classes):
        cls_top_idxs = top_gradients_dict[col_name]["top_n_idxs"]

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
            cur_ax.set_yticklabels(["0", "1", "2", "MIS"])

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
