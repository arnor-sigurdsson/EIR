import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Callable, Tuple, TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import torch
from shap import DeepExplainer
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

from human_origins_supervised.models.model_utils import gather_dloader_samples

from aislib.misc_utils import get_logger

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

logger = get_logger(__name__)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_target_classes(cl_args, label_encoder):

    if cl_args.model_task == "reg":
        return ["Regression"]

    if cl_args.act_classes:
        target_classes = cl_args.act_classes
    else:
        target_classes = label_encoder.classes_

    return target_classes


def get_act_condition(sample_label, label_encoder, target_classes, model_task):
    if model_task == "reg":
        return "Regression"

    le_it = label_encoder.inverse_transform
    cur_trn_label = le_it([sample_label.item()])[0]
    if cur_trn_label in target_classes:
        return cur_trn_label

    return None


def accumulate_activations(
    config: "Config",
    valid_dataset: Dataset,
    act_func: Callable,
    transform_funcs: Dict[str, Tuple[Callable]],
):
    c = config

    target_classes = get_target_classes(c.cl_args, c.label_encoder)

    valid_sampling_dloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    acc_acts = {name: [] for name in target_classes}
    acc_acts_masked = {name: [] for name in target_classes}

    for single_sample, sample_label, *_ in valid_sampling_dloader:
        # we want to keep the original sample for masking
        single_sample_org = deepcopy(single_sample).cpu().numpy().squeeze()
        cur_trn_label = get_act_condition(
            sample_label, c.label_encoder, target_classes, c.cl_args.model_task
        )

        if cur_trn_label:
            # apply pre-processing functions on sample and input
            for pre_func in transform_funcs.get("pre", ()):
                single_sample, sample_label = pre_func(
                    single_sample=single_sample, sample_label=sample_label
                )

            single_acts = act_func(
                single_sample=single_sample, sample_label=sample_label
            )
            if single_acts is not None:
                # apply post-processing functions on activations
                for post_func in transform_funcs.get("post", ()):
                    single_acts = post_func(single_acts)

                acc_acts[cur_trn_label].append(single_acts.squeeze())

                single_acts_masked = single_sample_org * single_acts
                acc_acts_masked[cur_trn_label].append(single_acts_masked.squeeze())

    return acc_acts, acc_acts_masked


def rescale_gradients(gradients):
    gradients_resc = gradients - gradients.min()
    gradients_resc = gradients_resc / (gradients.max() - gradients.min())
    return gradients_resc


def get_shap_object(
    model: torch.nn.Module,
    device: str,
    train_loader: DataLoader,
    n_background_samples: int = 64,
):
    background, *_ = gather_dloader_samples(train_loader, device, n_background_samples)

    explainer = DeepExplainer(model, background)
    return explainer


def get_shap_sample_acts_deep(
    explainer: DeepExplainer,
    single_sample: torch.Tensor,
    sample_label: torch.Tensor,
    model_task: str,
):
    """
    Note: We only get the grads for a correct prediction.

    TODO: Add functionality to use ranked_outputs or all outputs.
    """
    with suppress_stdout():
        output = explainer.shap_values(single_sample, ranked_outputs=1)
        if model_task == "reg":
            assert isinstance(output, np.ndarray)
            return output

    assert len(output) == 2
    shap_grads, pred_label = output
    if pred_label.item() == sample_label.item():
        return shap_grads[0]

    return None


def get_snp_cols_w_top_grads(
    accumulated_grads: Dict[str, List[np.array]],
    n: int = 10,
    custom_indexes_dict: dict = None,
) -> Dict[str, Dict[str, np.array]]:
    """
    `accumulated_grads` specs:

        {'Asia': [all grads for samples w. Asia label], ...}

    First we find the average gradients for each SNP form by averaging accross
    samples.

    Then we have something that is (4 x width) dimension, and we find the per
    column (i.e. SNP form) max gradient. Of the max grad per SNP, we want to
    find the top N SNPs where the maximum gradient (any form) is found.

    We use those indexes to grab the top SNPs and grads per class.
    """
    top_snps_per_class = {}

    for cls, grads in accumulated_grads.items():

        if grads:
            grads_arr = np.array(grads)
            grads_arr_mean = grads_arr.mean(axis=0)

            top_snps_per_class[cls] = {}

            if not custom_indexes_dict:
                sum_snp_values = grads_arr_mean.sum(0)
                top_n_idxs = sorted(np.argpartition(sum_snp_values, -n)[-n:])

                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs
                top_snps_per_class[cls]["top_n_grads"] = grads_arr_mean[:, top_n_idxs]

            else:
                top_n_idxs = custom_indexes_dict[cls]
                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs
                top_snps_per_class[cls]["top_n_grads"] = grads_arr_mean[:, top_n_idxs]
        else:
            logger.warning(
                "No gradients aggregated for class %s due to no "
                "correct predictions for the class, top activations "
                "will not be plotted.",
                cls,
            )

    return top_snps_per_class


def get_snp_names(snp_file: str, data_folder: Path = None) -> np.array:
    """
    Not super happy about this implementation, as the infer option is kind of
    restricted to the project structure - but maybe that's ok?

    The common structure is data/UKBB/processed/<ind_size>/<snp_size>/<type>
    """
    if snp_file == "infer":
        if not data_folder:
            raise ValueError(
                f"'data_folder' variable must be set with 'infer'"
                f" as snp_file parameter."
            )

        snp_size = data_folder.parts[3]
        ind_size = data_folder.parts[4]
        assert snp_size.startswith("full") or int(snp_size.split("_")[0])
        assert ind_size.startswith("full") or int(ind_size.split("_")[0])

        snp_string = f"parsed_files/{ind_size}/{snp_size}/data_final.snp"
        snp_file = Path(data_folder).parents[2] / snp_string

        if not snp_file.exists():
            raise FileNotFoundError(
                f"Could not find {snp_file} when inferring" f"about it's location."
            )

    snp_df = pd.read_csv(snp_file, sep=r"\s+", usecols=[0], names=["snps"])
    snp_arr = snp_df.snps.array

    return snp_arr


def gather_and_rescale_snps(
    all_gradients_dict: Dict[str, List[np.array]],
    top_gradients_dict: Dict[str, Dict[str, np.array]],
    classes: List[str],
) -> Dict[str, Dict[str, np.array]]:
    """
    `accumulated_grads` specs:

        {'Asia': [all grads for samples w. Asia label],
         'Europe': [all grads for samples w. Europe label]}

    `top_gradients_dict` specs:

        {'Asia': {'top_n_idxs': [],
                  'top_n_grads': [],
                 }
        }

    We want to create a dict with the following specs:

        {
            'Asia': {
                        'Asia Grads Indexed By Top Asia SNPs': [...],
                        'Europe Grads Indexed By Top Asia SNPs': [...],
                        'Africa Grads Indexed By Top Asia SNPs': [...],
                    }
            'Europe': {
                        'Asia Grads Indexed By Top Europe SNPs': [...],
                        'Europe Grads Indexed By Top Europe SNPs': [...],
                        'Africa Grads Indexed By Top Europe SNPs': [...],
                    }
        }

    The each row of the above is a label (i.e. country, e.g. 'Asia') and each
    column contains gradient values as indexed by the top gradient SNPs for
    that label (i.e. highest positive gradients sum over Asian samples).

    We make sure that the top gradients gotten in this function are the
    same as gotten previously (when we just look at each country separately).

    We then accumulate each column (average grads for each country, indexed
    by the top grads for a given country), and rescale them (so we have
    column-wise rescaling).
    """
    top_snps_dict = {}

    for col_name_ in classes:
        cls_top_idxs_ = top_gradients_dict[col_name_]["top_n_idxs"]
        top_snps_dict[col_name_] = {}

        cur_top_snps_all_labels = []
        for row_name_ in classes:
            cur_row_grads_arr = np.array(all_gradients_dict[row_name_])
            cur_row_grads_mean = cur_row_grads_arr.mean(axis=0)
            cur_row_indexed_grads = cur_row_grads_mean[:, cls_top_idxs_]

            cur_top_snps_all_labels.append(cur_row_indexed_grads)

        cur_top_snps_all_labels = np.array(cur_top_snps_all_labels)
        cur_top_snps_all_labels_rscl = rescale_gradients(cur_top_snps_all_labels)

        for list_idx, row_name_ in enumerate(classes):
            top_snps_dict[col_name_][row_name_] = cur_top_snps_all_labels_rscl[list_idx]

    return top_snps_dict


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

    grads_scaled: dict = gather_and_rescale_snps(
        accumulated_grads, top_gradients_dict, classes
    )

    fig = plt.figure(figsize=(n_cls * 4, n_cls * 2 + 1))
    gs = gridspec.GridSpec(n_cls, n_cls, wspace=0.2, hspace=0.2)

    for grad_idx, col_name in enumerate(classes):
        cls_top_idxs = top_gradients_dict[col_name]["top_n_idxs"]

        for cls_idx, row_name in enumerate(classes):

            cur_grads = grads_scaled[col_name][row_name]

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


def index_masked_grads(top_grads_dict, accumulated_grads_times_input):
    indexes_from_all_grads = {
        key: top_grads_dict[key]["top_n_idxs"] for key in top_grads_dict.keys()
    }

    top_gradients_dict_masked_inputs = get_snp_cols_w_top_grads(
        accumulated_grads_times_input, custom_indexes_dict=indexes_from_all_grads
    )

    return top_gradients_dict_masked_inputs


def save_masked_grads(
    acc_grads_times_inp, top_gradients_dict, snp_names, sample_outfolder
):
    top_grads_msk_inputs = index_masked_grads(top_gradients_dict, acc_grads_times_inp)

    plot_top_gradients(
        acc_grads_times_inp,
        top_grads_msk_inputs,
        snp_names,
        sample_outfolder,
        "top_snps_masked.png",
    )

    np.save(sample_outfolder / "top_grads_masked.npy", top_grads_msk_inputs)


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


def analyze_activations(config: "Config", act_func, proc_funcs, outfolder):
    c = config
    args = config.cl_args

    acc_acts, acc_acts_masked = accumulate_activations(
        c, c.valid_dataset, act_func, proc_funcs
    )

    top_gradients_dict = get_snp_cols_w_top_grads(acc_acts)
    snp_names = get_snp_names(args.snp_file, Path(args.data_folder))

    plot_top_gradients(acc_acts, top_gradients_dict, snp_names, outfolder)

    np.save(outfolder / "top_acts.npy", top_gradients_dict)

    save_masked_grads(acc_acts_masked, top_gradients_dict, snp_names, outfolder)

    plot_snp_gradients(acc_acts, outfolder, "avg")
