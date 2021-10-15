from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Union, TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger

from eir.interpretation.interpretation_utils import get_target_class_name
from eir.visualization import interpretation_visualization as av

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleActivation
    from eir.data_load.label_setup import al_label_transformers_object

al_gradients_dict = Dict[str, List[np.ndarray]]
al_top_gradients_dict = Dict[str, Dict[str, np.ndarray]]
al_scaled_grads_dict = Dict[str, Dict[str, np.ndarray]]
al_transform_funcs = Dict[str, Tuple[Callable]]

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class ParsedOmicsActivations:
    accumulated_acts: Dict[str, np.ndarray]
    accumulated_acts_masked: Dict[str, np.ndarray]


def get_omics_consumer(
    target_transformer: "al_label_transformers_object",
    input_name: str,
    target_column: str,
    column_type: str,
) -> Callable[[Union["SampleActivation", None]], ParsedOmicsActivations]:

    acc_acts = {}
    acc_acts_masked = {}

    n_samples = 0

    def _consumer(
        activation: Union["SampleActivation", None]
    ) -> ParsedOmicsActivations:

        nonlocal n_samples

        if activation is None:

            for key, value in acc_acts.items():
                acc_acts[key] = value / n_samples

            for key, value in acc_acts_masked.items():
                acc_acts_masked[key] = value / n_samples

            parsed_activations = ParsedOmicsActivations(
                accumulated_acts=acc_acts, accumulated_acts_masked=acc_acts_masked
            )
            return parsed_activations

        sample_inputs = activation.sample_info.inputs
        sample_target_labels = activation.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[target_column],
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )

        sample_acts = activation.sample_activations[input_name].squeeze()
        if cur_label_name not in acc_acts:
            acc_acts[cur_label_name] = sample_acts
        else:
            acc_acts[cur_label_name] += sample_acts

        # we want to keep the original sample for masking
        inputs_omics = sample_inputs[input_name]
        single_sample_copy = deepcopy(inputs_omics).cpu().numpy()
        single_acts_masked = (single_sample_copy * sample_acts).squeeze()

        if cur_label_name not in acc_acts_masked:
            acc_acts_masked[cur_label_name] = single_acts_masked
        else:
            acc_acts_masked[cur_label_name] += single_acts_masked

        n_samples += 1

    return _consumer


def analyze_omics_input_activations(
    experiment: "Experiment",
    input_name: str,
    target_column_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: ParsedOmicsActivations,
) -> None:
    exp = experiment

    acc_acts = all_activations.accumulated_acts
    acc_acts_masked = all_activations.accumulated_acts_masked

    abs_grads = True if target_column_type == "con" else False
    top_gradients_dict = get_snp_cols_w_top_grads(
        accumulated_grads=acc_acts, abs_grads=abs_grads
    )

    omics_data_type_config = exp.inputs[input_name].input_config.input_type_info
    cur_snp_path = Path(omics_data_type_config.snp_file)
    snp_df = read_snp_df(snp_file_path=cur_snp_path)

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
    df_snp_grads_with_abs = _add_absolute_summed_snp_gradients_to_df(
        df_snp_grads=df_snp_grads
    )
    av.plot_snp_manhattan_plots(
        df_snp_grads=df_snp_grads_with_abs,
        outfolder=activation_outfolder,
        title_extra=f" - {target_column_name}",
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
    accumulated_grads: Dict[str, np.ndarray], outfolder: Path, snp_df: pd.DataFrame
) -> pd.DataFrame:

    df_output = deepcopy(snp_df)
    for label, grads in accumulated_grads.items():

        grads_np = grads
        grads_averaged = grads_np.sum(0)

        column_name = label + "_activations"
        df_output[column_name] = grads_averaged

    df_output.to_csv(path_or_buf=outfolder / "snp_activations.csv")
    return df_output


def _add_absolute_summed_snp_gradients_to_df(
    df_snp_grads: pd.DataFrame,
) -> pd.DataFrame:

    df_snp_grads_copy = df_snp_grads.copy()

    activations_columns = [
        i for i in df_snp_grads_copy.columns if i.endswith("_activations")
    ]
    if "Aggregated_activations" in activations_columns:
        raise ValueError(
            "Cannot compute aggregated activations as reserved column name "
            "'Aggregated' already present as target (all activation columns: %s). "
            "Please rename the column 'Aggregated' to something else "
            "in the relevant target file.",
            activations_columns,
        )

    df_snp_grads_copy["Aggregated_activations"] = (
        df_snp_grads_copy[activations_columns].abs().sum(axis=1)
    )

    return df_snp_grads_copy


def parse_single_omics_activations(
    experiment: "Experiment",
    omics_input_name: str,
    target_column: str,
    column_type: str,
    activations: Sequence["SampleActivation"],
) -> Tuple[Dict, Dict]:

    exp = experiment
    target_transformer = exp.target_transformers[target_column]

    acc_acts = defaultdict(list)
    acc_acts_masked = defaultdict(list)

    for sample_activation in activations:

        sample_inputs = sample_activation.sample_info.inputs
        sample_target_labels = sample_activation.sample_info.target_labels
        sample_acts = sample_activation.sample_activations[omics_input_name]

        # we want to keep the original sample for masking
        inputs_omics = sample_inputs[omics_input_name]
        single_sample_copy = deepcopy(inputs_omics).cpu().numpy().squeeze()

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[target_column],
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )

        acc_acts[cur_label_name].append(sample_acts.squeeze())

        single_acts_masked = single_sample_copy * sample_acts
        acc_acts_masked[cur_label_name].append(single_acts_masked.squeeze())

    return acc_acts, acc_acts_masked


def rescale_gradients(gradients: np.ndarray) -> np.ndarray:
    gradients_resc = gradients - gradients.min()
    gradients_resc = gradients_resc / (gradients.max() - gradients.min())
    return gradients_resc


def get_snp_cols_w_top_grads(
    accumulated_grads: Dict[str, np.ndarray],
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

        if grads is not None:
            top_snps_per_class[cls] = {}

            if not custom_indexes_dict:
                if abs_grads:
                    grads = np.abs(grads)

                sum_snp_values = grads.sum(0)

                top_n_idxs = sorted(np.argpartition(sum_snp_values, -n)[-n:])

                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs
                top_snps_per_class[cls]["top_n_grads"] = grads[:, top_n_idxs]

            else:
                top_n_idxs = custom_indexes_dict[cls]
                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs
                top_snps_per_class[cls]["top_n_grads"] = grads[:, top_n_idxs]
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
    all_gradients_dict: Dict[str, np.ndarray],
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
            cur_row_grads_arr = all_gradients_dict[row_name_]
            cur_row_grads_mean = cur_row_grads_arr
            cur_row_indexed_grads = cur_row_grads_mean[:, cls_top_idxs_]

            cur_top_snps_all_labels.append(cur_row_indexed_grads)

        cur_top_snps_all_labels = np.array(cur_top_snps_all_labels)
        cur_top_snps_all_labels_rscl = rescale_gradients(
            gradients=cur_top_snps_all_labels
        )

        for list_idx, row_name_ in enumerate(classes):
            top_snps_dict[col_name_][row_name_] = cur_top_snps_all_labels_rscl[list_idx]

    return top_snps_dict


def save_masked_grads(
    acc_grads_times_inp: Dict[str, np.ndarray],
    top_gradients_dict: al_top_gradients_dict,
    snp_df: pd.DataFrame,
    sample_outfolder: Path,
) -> None:
    top_grads_msk_inputs = index_masked_grads(
        top_grads_dict=top_gradients_dict,
        accumulated_grads_times_input=acc_grads_times_inp,
    )

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
        fname="top_snps_masked.pdf",
    )

    np.save(str(sample_outfolder / "top_acts_masked.npy"), top_grads_msk_inputs)


def index_masked_grads(
    top_grads_dict: al_top_gradients_dict,
    accumulated_grads_times_input: Dict[str, np.ndarray],
) -> al_top_gradients_dict:
    indexes_from_all_grads = {
        key: top_grads_dict[key]["top_n_idxs"] for key in top_grads_dict.keys()
    }

    top_gradients_dict_masked_inputs = get_snp_cols_w_top_grads(
        accumulated_grads_times_input, custom_indexes_dict=indexes_from_all_grads
    )

    return top_gradients_dict_masked_inputs
