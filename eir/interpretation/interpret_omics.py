import pickle
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists

from eir.interpretation.interpretation_utils import get_target_class_name
from eir.setup.input_setup_modules.setup_omics import (
    ComputedOmicsInputInfo,
    read_bim,
    read_subset_file,
)
from eir.setup.schemas import OmicsInputDataConfig
from eir.utils.logging import get_logger
from eir.visualization import interpretation_visualization as av

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_transformers_object
    from eir.interpretation.interpretation import SampleAttribution
    from eir.train import Experiment

al_gradients_dict = Dict[str, List[np.ndarray]]
al_top_gradients_dict = Dict[str, Dict[str, np.ndarray]]
al_scaled_grads_dict = Dict[str, Dict[str, np.ndarray]]
al_transform_funcs = Dict[str, Tuple[Callable]]

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class ParsedOmicsAttributions:
    accumulated_acts: Dict[str, np.ndarray]
    accumulated_acts_masked: Dict[str, np.ndarray]


class OmicsConsumerCallable(Protocol):
    def __call__(
        self,
        attribution: Optional["SampleAttribution"],
    ) -> Optional[ParsedOmicsAttributions]: ...


def get_omics_consumer(
    target_transformer: "al_label_transformers_object",
    input_name: str,
    output_name: str,
    target_column: str,
    column_type: str,
) -> OmicsConsumerCallable:
    acc_acts: dict[str, np.ndarray] = {}
    acc_acts_masked: dict[str, np.ndarray] = {}
    n_samples: dict[str, int] = {}

    def _consumer(
        attribution: Optional["SampleAttribution"],
    ) -> Optional[ParsedOmicsAttributions]:
        nonlocal n_samples

        if attribution is None:
            for key, value in acc_acts.items():
                acc_acts[key] = value / n_samples[key]

            for key, value in acc_acts_masked.items():
                acc_acts_masked[key] = value / n_samples[key]

            parsed_attributions = ParsedOmicsAttributions(
                accumulated_acts=acc_acts, accumulated_acts_masked=acc_acts_masked
            )
            return parsed_attributions

        sample_inputs = attribution.sample_info.inputs
        sample_target_labels = attribution.sample_info.target_labels

        cur_label_name = get_target_class_name(
            sample_label=sample_target_labels[output_name][target_column],
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )

        sample_acts = attribution.sample_attributions[input_name].squeeze()
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
            n_samples[cur_label_name] = 1
        else:
            acc_acts_masked[cur_label_name] += single_acts_masked
            n_samples[cur_label_name] += 1

        return None

    return _consumer


def analyze_omics_input_attributions(
    experiment: "Experiment",
    input_name: str,
    target_column_name: str,
    target_column_type: str,
    attribution_outfolder: Path,
    all_attributions: ParsedOmicsAttributions,
) -> None:
    exp = experiment

    acc_acts = all_attributions.accumulated_acts
    acc_acts_masked = all_attributions.accumulated_acts_masked

    abs_grads = True if target_column_type == "con" else False
    top_gradients_dict = get_snp_cols_w_top_grads(
        accumulated_grads=acc_acts, abs_grads=abs_grads
    )

    input_object = exp.inputs[input_name]
    assert isinstance(input_object, ComputedOmicsInputInfo)

    omics_data_type_config = input_object.input_config.input_type_info
    assert isinstance(omics_data_type_config, OmicsInputDataConfig)

    assert omics_data_type_config.snp_file is not None
    cur_snp_path = Path(omics_data_type_config.snp_file)
    df_snps = read_bim(bim_file_path=str(cur_snp_path))

    if omics_data_type_config.subset_snps_file:
        subset_snps = read_subset_file(
            subset_snp_file_path=omics_data_type_config.subset_snps_file
        )
        df_snps = df_snps[df_snps["VAR_ID"].isin(subset_snps)]

    classes = sorted(list(top_gradients_dict.keys()))
    scaled_grads = gather_and_rescale_snps(
        all_gradients_dict=acc_acts,
        top_gradients_dict=top_gradients_dict,
        classes=classes,
    )
    av.plot_top_gradients(
        gathered_scaled_grads=scaled_grads,
        top_gradients_dict=top_gradients_dict,
        df_snps=df_snps,
        output_folder=attribution_outfolder,
    )

    _save_omics_grads(
        sample_output_folder=attribution_outfolder,
        file_name="top_acts.npy",
        grads_dict=top_gradients_dict,
    )

    save_masked_grads(
        acc_grads_times_inp=acc_acts_masked,
        top_gradients_dict=top_gradients_dict,
        df_snps=df_snps,
        sample_outfolder=attribution_outfolder,
    )

    df_snp_grads = _save_snp_gradients(
        accumulated_grads=acc_acts, outfolder=attribution_outfolder, df_snps=df_snps
    )
    df_snp_grads_with_abs = _add_absolute_summed_snp_gradients_to_df(
        df_snp_grads=df_snp_grads
    )
    av.plot_snp_manhattan_plots(
        df_snp_grads=df_snp_grads_with_abs,
        outfolder=attribution_outfolder,
        title_extra=f" - {target_column_name}",
    )


def _save_snp_gradients(
    accumulated_grads: Dict[str, np.ndarray], outfolder: Path, df_snps: pd.DataFrame
) -> pd.DataFrame:
    df_output = deepcopy(df_snps)
    for label, grads in accumulated_grads.items():
        grads_np = grads
        grads_averaged = grads_np.sum(0)

        column_name = label + "_attributions"
        df_output[column_name] = grads_averaged

    df_output.to_csv(path_or_buf=outfolder / "snp_attributions.csv")
    return df_output


def _add_absolute_summed_snp_gradients_to_df(
    df_snp_grads: pd.DataFrame,
) -> pd.DataFrame:
    df_snp_grads_copy = df_snp_grads.copy()

    attributions_columns = [
        i for i in df_snp_grads_copy.columns if i.endswith("_attributions")
    ]
    if "Aggregated_attributions" in attributions_columns:
        raise ValueError(
            "Cannot compute aggregated attributions as reserved column name "
            "'Aggregated' already present as target (all attribution columns: %s). "
            "Please rename the column 'Aggregated' to something else "
            "in the relevant target file.",
            attributions_columns,
        )

    df_snp_grads_copy["Aggregated_attributions"] = (
        df_snp_grads_copy[attributions_columns].abs().sum(axis=1)
    )

    return df_snp_grads_copy


def rescale_gradients(gradients: np.ndarray) -> np.ndarray:
    gradients_resc = gradients - gradients.min()
    gradients_resc = gradients_resc / (gradients.max() - gradients.min())
    return gradients_resc


def get_snp_cols_w_top_grads(
    accumulated_grads: Dict[str, np.ndarray],
    n: int = 10,
    custom_indexes_dict: Optional[dict[str, np.ndarray]] = None,
    abs_grads: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
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
    top_snps_per_class: Dict[str, Dict[str, np.ndarray]] = {}

    for cls, grads in accumulated_grads.items():
        if grads is not None:
            top_snps_per_class[cls] = {}

            if not custom_indexes_dict:
                if abs_grads:
                    grads = np.abs(grads)

                sum_snp_values = grads.sum(0)

                top_n_idxs = sorted(np.argpartition(sum_snp_values, -n)[-n:])
                top_n_idxs_np: np.ndarray = np.array(top_n_idxs)

                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs_np
                top_snps_per_class[cls]["top_n_grads"] = grads[:, top_n_idxs_np]

            else:
                top_n_idxs_np = custom_indexes_dict[cls]
                top_snps_per_class[cls]["top_n_idxs"] = top_n_idxs_np
                top_snps_per_class[cls]["top_n_grads"] = grads[:, top_n_idxs_np]
        else:
            logger.warning(
                "No gradients aggregated for class %s due to no "
                "correct predictions for the class, top attributions "
                "will not be plotted.",
                cls,
            )

    return top_snps_per_class


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
    top_snps_dict: al_scaled_grads_dict = {}

    for col_name_ in classes:
        cls_top_idxs_ = top_gradients_dict[col_name_]["top_n_idxs"]
        top_snps_dict[col_name_] = {}

        cur_top_snps_all_labels = []
        for row_name_ in classes:
            cur_row_grads_arr = all_gradients_dict[row_name_]
            cur_row_grads_mean = cur_row_grads_arr
            cur_row_indexed_grads = cur_row_grads_mean[:, cls_top_idxs_]

            cur_top_snps_all_labels.append(cur_row_indexed_grads)

        cur_top_snps_all_labels_np = np.array(cur_top_snps_all_labels)
        cur_top_snps_all_labels_rescaled = rescale_gradients(
            gradients=cur_top_snps_all_labels_np
        )

        for list_idx, row_name_ in enumerate(classes):
            cur_rescaled = cur_top_snps_all_labels_rescaled[list_idx]
            top_snps_dict[col_name_][row_name_] = cur_rescaled

    return top_snps_dict


def save_masked_grads(
    acc_grads_times_inp: Dict[str, np.ndarray],
    top_gradients_dict: al_top_gradients_dict,
    df_snps: pd.DataFrame,
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
        df_snps=df_snps,
        output_folder=sample_outfolder,
        fname="top_snps_masked.pdf",
    )

    _save_omics_grads(
        sample_output_folder=sample_outfolder,
        file_name="top_acts_masked.npy",
        grads_dict=top_grads_msk_inputs,
    )


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


def _save_omics_grads(
    sample_output_folder: Path,
    file_name: str,
    grads_dict: dict[str, dict[str, np.ndarray]],
) -> None:
    output_path = sample_output_folder / file_name
    ensure_path_exists(path=output_path)

    with open(output_path, "wb") as f:
        pickle.dump(obj=grads_dict, file=f)
