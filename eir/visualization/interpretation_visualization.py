from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aislib.misc_utils import ensure_path_exists
from matplotlib import gridspec

from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.interpretation.interpret_omics import (
        al_scaled_grads_dict,
        al_top_gradients_dict,
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
    df_snps: pd.DataFrame,
    output_folder: Path,
    fname: str = "top_snps.pdf",
    custom_ylabel: Optional[str] = None,
):
    n_cls = len(top_gradients_dict.keys())
    classes = sorted(list(top_gradients_dict.keys()))

    fig = plt.figure(figsize=(n_cls * 4, n_cls * 2 + 1))
    gs = gridspec.GridSpec(n_cls, n_cls, wspace=0.2, hspace=0.2)

    snp_names = df_snps["VAR_ID"].array
    for grad_idx, col_name in enumerate(classes):
        cls_top_idxs = top_gradients_dict[col_name]["top_n_idxs"]
        cur_snp_matrix = generate_snp_gradient_matrix(df_snps, cls_top_idxs)

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
    df_snp_grads: pd.DataFrame,
    outfolder: Path,
    title_extra: str = "",
):
    df_snp_grads_copy = df_snp_grads.copy()

    attributions_columns = [
        i for i in df_snp_grads_copy.columns if i.endswith("_attributions")
    ]

    df_snp_grads_copy = df_snp_grads_copy.sort_values(by=["CHR_CODE", "BP_COORD"])

    for col in attributions_columns:
        label_name = col.split("_attributions")[0]

        ax, fig = _get_manhattan_axis_and_figure(
            df=df_snp_grads_copy,
            chr_column_name="CHR_CODE",
            attribution_column_name=col,
        )

        y_ticks = ax.get_yticks()
        y_axis_tick_spacing = y_ticks[1] - y_ticks[0]
        y_max = df_snp_grads_copy[col].max() + y_axis_tick_spacing
        ax.set_ylim(ymin=0.0, ymax=y_max)

        ax.set_xlabel("Chromosome")
        ax.set_ylabel("Attribution")

        ax.set_title(f"{label_name}{title_extra}")
        plt.tight_layout()

        out_path = outfolder / f"manhattan/{label_name}_manhattan.png"
        ensure_path_exists(path=out_path)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close("all")


def _get_manhattan_axis_and_figure(
    df: pd.DataFrame,
    chr_column_name: str,
    attribution_column_name: str,
    colors: Optional[Union[Tuple[str, str], List[str]]] = None,
    figure_size: Tuple[int, int] = (12, 6),
    axis_rotation: int = 90,
    gwas_significant_line: bool = False,
    gwas_p_value: float = 5e-08,
    dot_size: int = 8,
    alpha_value: int = 1,
    axis_label_x: Optional[str] = None,
    axis_label_y: Optional[str] = None,
    axis_label_font_size: int = 9,
    axis_label_font_name: str = "Arial",
) -> Tuple[Axes, Figure]:
    """Adapted from https://github.com/reneshbedre/bioinfokit#manhattan-plot."""

    available_colors = _get_available_colors()

    df["tpval"] = df[attribution_column_name]
    df["ind"] = range(len(df))

    color_list = get_colors_list(
        df=df,
        chr_column_name=chr_column_name,
        colors=colors,
        available_colors=available_colors,
    )

    x_labels, x_ticks, fig, ax = initialize_plot(
        df=df,
        chr_column_name=chr_column_name,
        color_list=color_list,
        figure_size=figure_size,
        dot_size=dot_size,
        alpha_value=alpha_value,
    )

    if gwas_significant_line:
        ax.axhline(
            y=-np.log10(gwas_p_value),
            linestyle="--",
            color="#7d7d7d",
            linewidth=1,
        )

    finalize_plot(
        ax=ax,
        x_labels=x_labels,
        x_ticks=x_ticks,
        axis_rotation=axis_rotation,
        axis_label_x=axis_label_x,
        axis_label_y=axis_label_y,
        axis_label_font_size=axis_label_font_size,
        axis_label_font_name=axis_label_font_name,
    )

    return ax, fig


def _get_available_colors() -> Tuple[str, ...]:
    colors = (
        "#a7414a",
        "#282726",
        "#6a8a82",
        "#a37c27",
        "#563838",
        "#0584f2",
        "#f28a30",
        "#f05837",
        "#6465a5",
        "#00743f",
        "#be9063",
        "#de8cf0",
        "#888c46",
        "#c0334d",
        "#270101",
        "#8d2f23",
        "#ee6c81",
        "#65734b",
        "#14325c",
        "#704307",
        "#b5b3be",
        "#f67280",
        "#ffd082",
        "#ffd800",
        "#ad62aa",
        "#21bf73",
        "#a0855b",
        "#5edfff",
        "#08ffc8",
        "#ca3e47",
        "#c9753d",
        "#6c5ce7",
        "#a997df",
        "#513b56",
        "#590925",
        "#007fff",
        "#bf1363",
        "#f39237",
        "#0a3200",
        "#8c271e",
    )
    return colors


def get_colors_list(
    df: pd.DataFrame,
    chr_column_name: str,
    colors: Optional[Union[Tuple[str, str], List[str]]],
    available_colors: Tuple[str, ...],
) -> List[str]:
    if colors is not None and len(colors) == 2:
        assert isinstance(colors, tuple)
        return get_two_colors_list(
            df=df, chr_column_name=chr_column_name, colors=colors
        )
    elif colors is not None and len(colors) == df[chr_column_name].nunique():
        assert isinstance(colors, list)
        return colors
    elif colors is None:
        return list(available_colors[: df[chr_column_name].nunique()])
    else:
        raise ValueError("Error in color argument.")


def get_two_colors_list(
    df: pd.DataFrame, chr_column_name: str, colors: Tuple[str, str]
) -> List[str]:
    color_1 = int(df[chr_column_name].nunique() / 2) * [colors[0]]
    color_2 = int(df[chr_column_name].nunique() / 2) * [colors[1]]

    if df[chr_column_name].nunique() % 2 == 0:
        color_list = list(chain(*zip(color_1, color_2)))
    elif df[chr_column_name].nunique() % 2 == 1:
        color_list = list(chain(*zip(color_1, color_2)))
        color_list.append(colors[0])
    else:
        raise ValueError("Error in color argument.")

    return color_list


def initialize_plot(
    df: pd.DataFrame,
    chr_column_name: str,
    color_list: List[str],
    figure_size: Tuple[int, int],
    dot_size: int,
    alpha_value: float,
) -> Tuple[List[str], List[int], Figure, Axes]:
    x_labels: List[str] = []
    x_ticks: List[int] = []
    fig, ax = plt.subplots(figsize=figure_size)

    for i, (label, df1) in enumerate(df.groupby(chr_column_name)):
        df1.plot(
            kind="scatter",
            x="ind",
            y="tpval",
            color=color_list[i],
            s=dot_size,
            alpha=alpha_value,
            ax=ax,
        )
        df1_max_ind = df1["ind"].iloc[-1]
        df1_min_ind = df1["ind"].iloc[0]
        x_labels.append(str(label))
        x_ticks.append(int((df1_max_ind - (df1_max_ind - df1_min_ind) / 2)))

    return x_labels, x_ticks, fig, ax


def finalize_plot(
    ax: Axes,
    x_labels: List[str],
    x_ticks: List[int],
    axis_rotation: int,
    axis_label_x: Optional[str],
    axis_label_y: Optional[str],
    axis_label_font_size: int,
    axis_label_font_name: str,
) -> None:
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=axis_rotation)
    ax.set_xlabel(
        axis_label_x or "Chromosome",
        fontsize=axis_label_font_size,
        fontname=axis_label_font_name,
    )
    ax.set_ylabel(
        axis_label_y or r"Attribution",
        fontsize=axis_label_font_size,
        fontname=axis_label_font_name,
    )
