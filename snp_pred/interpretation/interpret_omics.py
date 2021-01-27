from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Union, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, Subset

from snp_pred.data_load.datasets import al_datasets
from snp_pred.train_utils.utils import call_hooks_stage_iterable
from snp_pred.visualization import interpretation_visualization as av

if TYPE_CHECKING:
    from snp_pred.train import Config

al_gradients_dict = Dict[str, List[np.ndarray]]
al_top_gradients_dict = Dict[str, Dict[str, np.ndarray]]
al_scaled_grads_dict = Dict[str, Dict[str, np.ndarray]]
al_transform_funcs = Dict[str, Tuple[Callable]]

logger = get_logger(name=__name__, tqdm_compatible=True)


def analyze_omics_activations(
    config: "Config",
    act_func: Callable,
    proc_funcs: al_transform_funcs,
    column_name: str,
    column_type: str,
    activation_outfolder: Path,
) -> None:
    c = config
    cl_args = config.cl_args

    acc_acts, acc_acts_masked = accumulate_single_omics_activations(
        config=c,
        target_column=column_name,
        column_type=column_type,
        act_func=act_func,
        transform_funcs=proc_funcs,
    )

    abs_grads = True if column_type == "con" else False
    top_gradients_dict = get_snp_cols_w_top_grads(
        accumulated_grads=acc_acts, abs_grads=abs_grads
    )

    snp_df = read_snp_df(snp_file_path=Path(cl_args.snp_file))

    classes = sorted(list(top_gradients_dict.keys()))
    scaled_grads = gather_and_rescale_snps(
        all_gradients_dict=acc_acts,
        top_gradients_dict=top_gradients_dict,
        classes=classes,
    )
    av.plot_top_gradients(
        gathered_scaled_grads=scaled_grads,
        top_gradients_dict=top_gradients_dict,
        snp_df=snp_df,
        output_folder=activation_outfolder,
    )

    np.save(file=str(activation_outfolder / "top_acts.npy"), arr=top_gradients_dict)

    save_masked_grads(
        acc_grads_times_inp=acc_acts_masked,
        top_gradients_dict=top_gradients_dict,
        snp_df=snp_df,
        sample_outfolder=activation_outfolder,
    )

    df_snp_grads = _save_snp_gradients(
        accumulated_grads=acc_acts, outfolder=activation_outfolder, snp_df=snp_df
    )
    av.plot_snp_manhattan_plots(
        df_snp_grads=df_snp_grads,
        outfolder=activation_outfolder,
        title_extra=f" - {column_name}",
    )
    av.plot_snp_manhattan_plots_plotly(
        df_snp_grads=df_snp_grads,
        outfolder=activation_outfolder,
        title_extra=f" - {column_name}",
    )


def read_snp_df(
    snp_file_path: Path, data_source: Union[Path, None] = None
) -> pd.DataFrame:
    """
    TODO: Deprecate support for .snp files – see hacky note below.

    NOTE:
        Here we have actually flipped the order of ALT / REF in the eigenstrat snp
        format. This is because while plink .raw format counts alternative alleles
        (usually minor, A1), eigensoft counts the major allele.

        That is, a high activation for the first position ([x, 0, 0, 0]) in a snp column
        means high activation for ALT in eigensoft format (0 REF counted) but
        high activation for REF in .bim format (0 ALT, i.e. 2 REF counted). In
        `generate_snp_gradient_matrix` we are assuming the .bim format, hence
        we have a little hack for now by using flipped eigensoft columns for
        compatibility (support for eigensoft to be deprecated).

        So if we were to have the correct eigensoft column order, we would have to
        change `generate_snp_gradient_matrix` row order and `plot_top_gradients` y-label
        order of (REF / HET / ALT).
    """

    if not snp_file_path:
        snp_file_path = infer_snp_file_path(data_source)

    bim_columns = ["CHR_CODE", "VAR_ID", "POS_CM", "BP_COORD", "ALT", "REF"]
    eig_snp_file_columns = ["VAR_ID", "CHR_CODE", "POS_CM", "BP_COORD", "ALT", "REF"]

    if snp_file_path.suffix == ".snp":
        logger.warning(
            "Support for .snp files will be deprecated soon, for now the program "
            "runs but when reading file %s, reference and alternative "
            "allele columns will probably be switched. Please consider using .bim.",
            snp_file_path,
        )
        snp_names = eig_snp_file_columns
    elif snp_file_path.suffix == ".bim":
        snp_names = bim_columns
    else:
        raise ValueError(
            "Please input either a .snp file or a .bim file for the snp_file argument."
        )

    df = pd.read_csv(snp_file_path, names=snp_names, sep=r"\s+")

    return df


def _save_snp_gradients(
    accumulated_grads: "al_gradients_dict", outfolder: Path, snp_df: pd.DataFrame
) -> pd.DataFrame:

    df_output = deepcopy(snp_df)
    for label, grads in accumulated_grads.items():

        grads_np = np.array(grads)
        grads_averaged = grads_np.mean(0).sum(0)

        column_name = label + "_activations"
        df_output[column_name] = grads_averaged

    df_output.to_csv(path_or_buf=outfolder / "snp_activations.csv")
    return df_output


def accumulate_single_omics_activations(
    config: "Config",
    omics_input_name: str,
    target_column: str,
    column_type: str,
    act_func: Callable,
    transform_funcs: al_transform_funcs,
) -> Tuple[Dict, Dict]:

    c = config
    cl_args = c.cl_args
    target_transformer = c.target_transformers[target_column]

    acc_acts = defaultdict(list)
    acc_acts_masked = defaultdict(list)

    target_classes_numerical = _get_numerical_target_classes(
        target_transformer=target_transformer,
        column_type=column_type,
        act_classes=cl_args.act_classes,
    )

    activations_data_loader = _get_activations_dataloader(
        validation_dataset=c.valid_dataset,
        max_acts_per_class=cl_args.max_acts_per_class,
        target_column=target_column,
        column_type=column_type,
        target_classes_numerical=target_classes_numerical,
    )

    for loader_batch in activations_data_loader:

        state = call_hooks_stage_iterable(
            hook_iterable=c.hooks.step_func_hooks.base_prepare_batch,
            common_kwargs={"config": c, "loader_batch": loader_batch},
            state=None,
        )
        batch = state["batch"]

        sample_inputs = batch.inputs
        sample_target_labels = batch.target_labels

        # we want to keep the original sample for masking
        inputs_omics = sample_inputs[omics_input_name]
        single_sample_copy = deepcopy(inputs_omics).cpu().numpy().squeeze()

        # apply pre-processing functions on sample and input
        # TODO: Update to account for all inputs, so don't use extra tabular
        for pre_func in transform_funcs.get("pre", ()):
            inputs_omics, _ = pre_func(
                single_sample=inputs_omics, sample_label=sample_target_labels
            )
            sample_inputs[omics_input_name] = inputs_omics

        if "tabular_cl_args" in sample_inputs:
            sample_inputs["tabular_cl_args"] = sample_inputs["tabular_cl_args"].detach()

        sample_acts = act_func(
            inputs=sample_inputs, sample_label=sample_target_labels[target_column]
        )
        if sample_acts is not None:
            # currently we are only going to get acts for snps
            # TODO: Add analysis / plots for embeddings / extra inputs.
            if isinstance(sample_acts, list):
                sample_acts = sample_acts[0]

            # apply post-processing functions on activations
            for post_func in transform_funcs.get("post", ()):
                sample_acts = post_func(sample_acts)

            cur_label_name = _get_target_class_name(
                sample_label=sample_target_labels[target_column],
                target_transformer=target_transformer,
                column_type=column_type,
                target_column_name=target_column,
            )

            acc_acts[cur_label_name].append(sample_acts.squeeze())

            single_acts_masked = single_sample_copy * sample_acts
            acc_acts_masked[cur_label_name].append(single_acts_masked.squeeze())

    return acc_acts, acc_acts_masked


def _get_numerical_target_classes(
    target_transformer, column_type: str, act_classes: Union[List[str], None]
):
    if column_type == "con":
        return [None]

    if act_classes is not None:
        target_transformer.transform(act_classes)

    return target_transformer.transform(target_transformer.classes_)


def _get_activations_dataloader(
    validation_dataset: al_datasets,
    max_acts_per_class: int,
    target_column: str,
    column_type: str,
    target_classes_numerical: Sequence[int],
) -> DataLoader:
    common_args = {"batch_size": 1, "shuffle": False}

    if max_acts_per_class is None:
        data_loader = DataLoader(dataset=validation_dataset, **common_args)
        return data_loader

    indices_func = _get_categorical_sample_indices_for_activations
    if column_type == "con":
        indices_func = _get_continuous_sample_indices_for_activations

    subset_indices = indices_func(
        dataset=validation_dataset,
        max_acts_per_class=max_acts_per_class,
        target_column=target_column,
        target_classes_numerical=target_classes_numerical,
    )
    subset_dataset = _subsample_dataset(
        dataset=validation_dataset, indices=subset_indices
    )
    data_loader = DataLoader(dataset=subset_dataset, **common_args)
    return data_loader


def _get_categorical_sample_indices_for_activations(
    dataset: al_datasets,
    max_acts_per_class: int,
    target_column: str,
    target_classes_numerical: Sequence[int],
) -> Tuple[int, ...]:
    acc_label_counts = defaultdict(lambda: 0)
    acc_label_limit = max_acts_per_class
    indices = []

    for index, sample in enumerate(dataset.samples):
        target_labels = sample.labels["target_labels"]
        cur_sample_target_label = target_labels[target_column]

        is_over_limit = acc_label_counts[cur_sample_target_label] == acc_label_limit
        is_not_in_target_classes = (
            cur_sample_target_label not in target_classes_numerical
        )
        if is_over_limit or is_not_in_target_classes:
            continue

        indices.append(index)
        acc_label_counts[cur_sample_target_label] += 1

    return tuple(indices)


def _get_continuous_sample_indices_for_activations(
    dataset: al_datasets, max_acts_per_class: int, *args, **kwargs
) -> Tuple[int, ...]:

    acc_label_limit = max_acts_per_class
    num_sample = len(dataset)
    indices = np.random.choice(num_sample, acc_label_limit)

    return tuple(indices)


def _subsample_dataset(dataset: al_datasets, indices: Sequence[int]):
    dataset_subset = Subset(dataset=dataset, indices=indices)
    return dataset_subset


def _get_target_class_name(
    sample_label: torch.Tensor,
    target_transformer: Union[StandardScaler, LabelEncoder],
    column_type: str,
    target_column_name: str,
):
    if column_type == "con":
        return target_column_name

    tt_it = target_transformer.inverse_transform
    cur_trn_label = tt_it([sample_label.item()])[0]

    return cur_trn_label


def rescale_gradients(gradients: np.ndarray) -> np.ndarray:
    gradients_resc = gradients - gradients.min()
    gradients_resc = gradients_resc / (gradients.max() - gradients.min())
    return gradients_resc


def get_snp_cols_w_top_grads(
    accumulated_grads: Dict[str, List[np.array]],
    n: int = 10,
    custom_indexes_dict: dict = None,
    abs_grads: bool = False,
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
                if abs_grads:
                    grads_arr_mean = np.abs(grads_arr_mean)
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


def infer_snp_file_path(data_source: Path):
    if not data_source:
        raise ValueError(
            "'data_source' variable must be set with 'infer'" " as snp_file parameter."
        )

    snp_size = data_source.parts[3]
    ind_size = data_source.parts[4]
    assert snp_size.startswith("full") or int(snp_size.split("_")[0])
    assert ind_size.startswith("full") or int(ind_size.split("_")[0])

    inferred_snp_string = f"parsed_files/{ind_size}/{snp_size}/data_final.snp"
    inferred_snp_file = Path(data_source).parents[2] / inferred_snp_string

    logger.info(
        "SNP file path not passed in as CL argument, will try inferred path: %s",
        inferred_snp_file,
    )

    if not inferred_snp_file.exists():
        raise FileNotFoundError(
            f"Could not find {inferred_snp_file} when inferring" f"about it's location."
        )

    return inferred_snp_file


def gather_and_rescale_snps(
    all_gradients_dict: al_gradients_dict,
    top_gradients_dict: al_top_gradients_dict,
    classes: List[str],
) -> al_scaled_grads_dict:
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


def index_masked_grads(
    top_grads_dict: al_top_gradients_dict,
    accumulated_grads_times_input: al_gradients_dict,
) -> al_top_gradients_dict:
    indexes_from_all_grads = {
        key: top_grads_dict[key]["top_n_idxs"] for key in top_grads_dict.keys()
    }

    top_gradients_dict_masked_inputs = get_snp_cols_w_top_grads(
        accumulated_grads_times_input, custom_indexes_dict=indexes_from_all_grads
    )

    return top_gradients_dict_masked_inputs


def save_masked_grads(
    acc_grads_times_inp: al_gradients_dict,
    top_gradients_dict: al_top_gradients_dict,
    snp_df: pd.DataFrame,
    sample_outfolder: Path,
) -> None:
    top_grads_msk_inputs = index_masked_grads(top_gradients_dict, acc_grads_times_inp)

    classes = sorted(list(top_gradients_dict.keys()))
    scaled_grads = gather_and_rescale_snps(
        all_gradients_dict=acc_grads_times_inp,
        top_gradients_dict=top_grads_msk_inputs,
        classes=classes,
    )
    av.plot_top_gradients(
        gathered_scaled_grads=scaled_grads,
        top_gradients_dict=top_grads_msk_inputs,
        snp_df=snp_df,
        output_folder=sample_outfolder,
        fname="top_snps_masked.png",
    )

    np.save(str(sample_outfolder / "top_acts_masked.npy"), top_grads_msk_inputs)
