import copy
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Union, Callable, Dict, Tuple, List, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from ignite.engine import Engine
from shap import DeepExplainer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader

import human_origins_supervised.visualization.activation_visualization as av
from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.models import model_utils
from human_origins_supervised.models.extra_inputs_module import get_extra_inputs
from human_origins_supervised.models.model_utils import gather_dloader_samples
from human_origins_supervised.models.models import CNNModel, MLPModel

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.train import Config

logger = get_logger(name=__name__, tqdm_compatible=True)

# Type aliases
# would be better to use Tuple here, but shap does literal type check for list, i.e.
# if type(data) == list:
al_model_inputs = List[Union[torch.Tensor, Union[torch.Tensor, None]]]
al_gradients_dict = Dict[str, List[np.array]]
al_top_gradients_dict = Dict[str, Dict[str, np.array]]
al_transform_funcs = Dict[str, Tuple[Callable]]


@contextmanager
def suppress_stdout() -> None:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_target_classes(
    cl_args: Namespace,
    column_type: str,
    target_column_name: str,
    target_transformer: Union[StandardScaler, LabelEncoder],
) -> List[str]:

    if column_type == "con":
        return [target_column_name]

    if cl_args.act_classes:
        target_classes = cl_args.act_classes
    else:
        target_classes = target_transformer.classes_

    return target_classes


def get_act_condition(
    sample_label: torch.Tensor,
    target_transformer: Union[StandardScaler, LabelEncoder],
    target_classes: List[str],
    column_type: str,
    target_column_name: str,
):
    if column_type == "con":
        return target_column_name

    tt_it = target_transformer.inverse_transform
    cur_trn_label = tt_it([sample_label.item()])[0]
    if cur_trn_label in target_classes:
        return cur_trn_label

    return None


def accumulate_activations(
    config: "Config",
    target_column: str,
    column_type: str,
    act_func: Callable,
    transform_funcs: al_transform_funcs,
) -> Tuple[Dict, Dict]:

    c = config
    cl_args = c.cl_args

    target_classes = get_target_classes(
        cl_args=c.cl_args,
        column_type=column_type,
        target_column_name=target_column,
        target_transformer=c.target_transformers[target_column],
    )

    valid_sampling_dloader = DataLoader(c.valid_dataset, batch_size=1, shuffle=False)

    acc_acts = {name: [] for name in target_classes}
    acc_acts_masked = {name: [] for name in target_classes}

    for single_sample, sample_label, sample_id in valid_sampling_dloader:
        # we want to keep the original sample for masking
        single_sample_copy = deepcopy(single_sample).cpu().numpy().squeeze()

        cur_trn_label = get_act_condition(
            sample_label=sample_label[target_column],
            target_transformer=c.target_transformers[target_column],
            target_classes=target_classes,
            column_type=column_type,
            target_column_name=target_column,
        )

        if cur_trn_label:
            # apply pre-processing functions on sample and input
            for pre_func in transform_funcs.get("pre", ()):
                single_sample, sample_label = pre_func(
                    single_sample=single_sample, sample_label=sample_label
                )

            extra_inputs = get_extra_inputs(
                cl_args, list(sample_id), c.valid_dataset.labels_dict, c.model
            )
            # detach for shap
            if extra_inputs is not None:
                extra_inputs = extra_inputs.detach()

            shap_inputs = [i for i in (single_sample, extra_inputs) if i is not None]
            single_acts = act_func(inputs=shap_inputs, sample_label=sample_label)
            if single_acts is not None:
                # currently we are only going to get acts for snps
                # TODO: Add analysis / plots for embeddings / extra inputs.
                if isinstance(single_acts, list):
                    single_acts = single_acts[0]

                # apply post-processing functions on activations
                for post_func in transform_funcs.get("post", ()):
                    single_acts = post_func(single_acts)

                acc_acts[cur_trn_label].append(single_acts.squeeze())

                single_acts_masked = single_sample_copy * single_acts
                acc_acts_masked[cur_trn_label].append(single_acts_masked.squeeze())

    return acc_acts, acc_acts_masked


def rescale_gradients(gradients: np.ndarray) -> np.ndarray:
    gradients_resc = gradients - gradients.min()
    gradients_resc = gradients_resc / (gradients.max() - gradients.min())
    return gradients_resc


def get_shap_object(
    config: "Config",
    model: Union[CNNModel, MLPModel],
    column_name: str,
    device: str,
    train_loader: DataLoader,
    n_background_samples: int = 64,
):
    c = config
    cl_args = c.cl_args

    background, _, ids = gather_dloader_samples(
        train_loader, device, n_background_samples
    )

    extra_inputs = get_extra_inputs(cl_args, ids, c.labels_dict, c.model)
    # detach for shap
    if extra_inputs is not None:
        extra_inputs = extra_inputs.detach()

    shap_inputs = [i for i in (background, extra_inputs) if i is not None]

    hook_partial = partial(
        _grab_single_target_from_model_output_hook, output_target_column="Origin"
    )
    hook_handle = model.register_forward_hook(hook_partial)

    explainer = DeepExplainer(model=model, data=shap_inputs)

    return explainer, hook_handle


def _grab_single_target_from_model_output_hook(
    self: Union[CNNModel, MLPModel],
    input_: torch.Tensor,
    output: Dict[str, torch.Tensor],
    output_target_column: str,
) -> torch.Tensor:
    return output[output_target_column]


def get_shap_sample_acts_deep(
    explainer: DeepExplainer,
    inputs: al_model_inputs,
    sample_label: torch.Tensor,
    column_type: str,
):
    """
    Note: We only get the grads for a correct prediction.

    TODO: Add functionality to use ranked_outputs or all outputs.
    """
    with suppress_stdout():
        output = explainer.shap_values(inputs, ranked_outputs=1)

    if column_type == "con":
        assert isinstance(output[0], np.ndarray)
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


def infer_snp_file_path(data_folder: Path):
    if not data_folder:
        raise ValueError(
            f"'data_folder' variable must be set with 'infer'"
            f" as snp_file parameter."
        )

    snp_size = data_folder.parts[3]
    ind_size = data_folder.parts[4]
    assert snp_size.startswith("full") or int(snp_size.split("_")[0])
    assert ind_size.startswith("full") or int(ind_size.split("_")[0])

    inferred_snp_string = f"parsed_files/{ind_size}/{snp_size}/data_final.snp"
    inferred_snp_file = Path(data_folder).parents[2] / inferred_snp_string

    logger.info(
        "SNP file path not passed in as CL argument, will try inferred path: %s",
        inferred_snp_file,
    )

    if not inferred_snp_file.exists():
        raise FileNotFoundError(
            f"Could not find {inferred_snp_file} when inferring" f"about it's location."
        )

    return inferred_snp_file


def read_snp_df(
    snp_file_path: Path, data_folder: Union[Path, None] = None
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
        snp_file_path = infer_snp_file_path(data_folder)

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


def gather_and_rescale_snps(
    all_gradients_dict: al_gradients_dict,
    top_gradients_dict: al_top_gradients_dict,
    classes: List[str],
) -> al_top_gradients_dict:
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
        acc_grads_times_inp, top_grads_msk_inputs, classes
    )
    av.plot_top_gradients(
        scaled_grads,
        top_grads_msk_inputs,
        snp_df,
        sample_outfolder,
        "top_snps_masked.png",
    )

    np.save(str(sample_outfolder / "top_grads_masked.npy"), top_grads_msk_inputs)


def analyze_activations(
    config: "Config",
    act_func: Callable,
    proc_funcs: al_transform_funcs,
    target_column: str,
    column_type: str,
    outfolder: Path,
) -> None:
    c = config
    cl_args = config.cl_args

    acc_acts, acc_acts_masked = accumulate_activations(
        config=c,
        target_column=target_column,
        column_type=column_type,
        act_func=act_func,
        transform_funcs=proc_funcs,
    )

    np.save(str(outfolder / "all_acts.npy"), acc_acts)

    abs_grads = True if column_type == "con" else False
    top_gradients_dict = get_snp_cols_w_top_grads(acc_acts, abs_grads=abs_grads)

    snp_df = read_snp_df(Path(cl_args.snp_file), Path(cl_args.data_folder))

    classes = sorted(list(top_gradients_dict.keys()))
    scaled_grads = gather_and_rescale_snps(acc_acts, top_gradients_dict, classes)
    av.plot_top_gradients(scaled_grads, top_gradients_dict, snp_df, outfolder)

    np.save(str(outfolder / "top_acts.npy"), top_gradients_dict)

    save_masked_grads(acc_acts_masked, top_gradients_dict, snp_df, outfolder)

    av.plot_snp_gradients(acc_acts, outfolder, "avg")


def activation_analysis_handler(
    engine: Engine, handler_config: "HandlerConfig"
) -> None:
    """
    We need to copy the model to avoid affecting the actual model during
    training (e.g. zero-ing out gradients).

    TODO: Refactor this function further – reuse for parts for benchmarking.
    """

    c = handler_config.config
    cl_args = c.cl_args
    iteration = engine.state.iteration

    def pre_transform(single_sample, sample_label, column_name: str):
        single_sample = single_sample.to(device=cl_args.device, dtype=torch.float32)

        sample_label = model_utils.cast_labels(
            target_columns=c.target_columns, device=cl_args.device, labels=sample_label
        )[column_name]

        return single_sample, sample_label

    if cl_args.get_acts:
        model_copy = copy.deepcopy(c.model)
        target_columns_gen = get_target_columns_generator(c.target_columns)

        for column_type, column_name in target_columns_gen:
            sample_outfolder = _get_sample_outfolder(
                run_folder=handler_config.run_folder,
                target_column_name=column_name,
                iteration=iteration,
            )
            ensure_path_exists(sample_outfolder, is_folder=True)

            no_explainer_background_samples = np.max([int(cl_args.batch_size / 8), 16])

            explainer, handle = get_shap_object(
                config=c,
                model=model_copy,
                column_name=column_name,
                device=cl_args.device,
                train_loader=c.train_loader,
                n_background_samples=no_explainer_background_samples,
            )

            proc_funcs = {"pre": (partial(pre_transform, column_name=column_name),)}
            act_func = partial(
                get_shap_sample_acts_deep, explainer=explainer, column_type=column_type
            )

            analyze_activations(
                config=c,
                act_func=act_func,
                proc_funcs=proc_funcs,
                target_column=column_name,
                column_type=column_type,
                outfolder=sample_outfolder,
            )

            handle.remove()


def _get_sample_outfolder(
    run_folder: Path, target_column_name: str, iteration: int
) -> Path:
    sample_outfolder = Path(
        run_folder, "results", target_column_name, "samples", str(iteration)
    )

    return sample_outfolder
