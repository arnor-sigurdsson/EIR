import warnings
from pathlib import Path
from textwrap import wrap
from typing import Dict, Sequence, TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from matplotlib import MatplotlibDeprecationWarning
from sklearn.preprocessing import LabelEncoder

from eir.experiment_io.experiment_io import load_transformers
from eir.interpretation.interpretation_utils import (
    stratify_activations_by_target_classes,
    plot_activations_bar,
)

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.interpretation.interpretation import SampleActivation


logger = get_logger(__name__)


def analyze_tabular_input_activations(
    experiment: "Experiment",
    input_name: str,
    output_name: str,
    target_column_name: str,
    target_column_type: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
):
    exp = experiment

    tabular_type_info_config = exp.inputs[input_name].input_config.input_type_info
    cat_columns = tabular_type_info_config.input_cat_columns
    con_columns = tabular_type_info_config.input_con_columns

    tabular_model = experiment.model.input_modules[input_name]
    activation_tensor_slices = set_up_tabular_tensor_slices(
        cat_input_columns=cat_columns,
        con_input_columns=con_columns,
        embedding_module=tabular_model,
    )

    parsed_activations = parse_tabular_activations_for_feature_importance(
        activation_tensor_slices=activation_tensor_slices,
        all_activations=all_activations,
        input_name=input_name,
    )
    df_activations = get_tabular_activation_df(parsed_activations=parsed_activations)
    plot_activations_bar(
        df_activations=df_activations,
        outpath=activation_outfolder / "feature_importance.pdf",
    )
    df_activations.to_csv(activation_outfolder / "feature_importances.csv")

    target_transformer = exp.outputs[output_name].target_transformers[
        target_column_name
    ]
    all_activations_class_stratified = stratify_activations_by_target_classes(
        all_activations=all_activations,
        target_transformer=target_transformer,
        output_name=output_name,
        target_column=target_column_name,
        column_type=target_column_type,
    )
    for class_name, class_activations in all_activations_class_stratified.items():

        cur_class_outfolder = activation_outfolder / class_name
        ensure_path_exists(path=cur_class_outfolder, is_folder=True)

        if con_columns:

            cat_to_con_cutoff = get_cat_to_con_cutoff_from_slices(
                slices=activation_tensor_slices,
                cat_input_columns=cat_columns,
            )
            continuous_shap = _gather_continuous_shap_values(
                all_activations=class_activations,
                cat_to_con_cutoff=cat_to_con_cutoff,
                input_name=input_name,
            )
            continuous_inputs = _gather_continuous_inputs(
                all_activations=class_activations,
                cat_to_con_cutoff=cat_to_con_cutoff,
                con_names=con_columns,
                input_name=input_name,
            )
            plot_tabular_beeswarm(
                shap_values=continuous_shap,
                features=continuous_inputs,
                activation_outfolder=cur_class_outfolder,
                class_name=class_name,
            )

        cat_act_dfs = []
        for cat_column in cat_columns:

            categorical_shap = _gather_categorical_shap_values(
                all_activations=class_activations,
                cur_slice=activation_tensor_slices[cat_column],
                input_name=input_name,
            )
            categorical_inputs = _gather_categorical_inputs(
                all_activations=class_activations,
                cat_name=cat_column,
                input_name=input_name,
            )
            cat_column_transformers = load_transformers(
                output_folder=experiment.configs.global_config.output_folder,
                transformers_to_load={input_name: [cat_column]},
            )

            categorical_inputs_mapped = map_categorical_labels_to_names(
                cat_column_transformers=cat_column_transformers[input_name],
                cat_column=cat_column,
                categorical_inputs=categorical_inputs,
            )

            plot_tabular_categorical_feature(
                shap_values=categorical_shap,
                features=categorical_inputs_mapped,
                feature_name_to_plot=cat_column,
                class_name=class_name,
                activation_outfolder=cur_class_outfolder,
            )

            df_cur_categorical_acts = _parse_categorical_shap_values_for_serialization(
                categorical_inputs_mapped=categorical_inputs_mapped,
                shap_values_for_input=categorical_shap,
                column_name=cat_column,
            )
            cat_act_dfs.append(df_cur_categorical_acts)

        _save_categorical_acts(
            dfs_categorical_acts_for_class=cat_act_dfs,
            class_name=class_name,
            output_folder=cur_class_outfolder,
        )


def set_up_tabular_tensor_slices(
    cat_input_columns: Sequence[str],
    con_input_columns: Sequence[str],
    embedding_module: torch.nn.Module,
) -> Dict[str, slice]:
    """
    TODO: Probably better to pass in Dict[column_name, nn.Embedding]
    """
    slices = {}

    current_index = 0
    for cat_column in cat_input_columns:
        cur_embedding = getattr(embedding_module, "embed_" + cat_column)
        cur_embedding_dim = cur_embedding.embedding_dim

        slice_start = current_index
        slice_end = current_index + cur_embedding_dim
        slices[cat_column] = slice(slice_start, slice_end)

        current_index = slice_end

    for con_column in con_input_columns:
        slice_start = current_index
        slice_end = slice_start + 1

        slices[con_column] = slice(slice_start, slice_end)
        current_index = slice_end

    return slices


def get_cat_to_con_cutoff_from_slices(
    slices: Dict[str, slice], cat_input_columns: Sequence[str]
) -> int:
    cutoff = 0

    for cat_column in cat_input_columns:
        cur_cat_slice = slices[cat_column]
        slice_size = cur_cat_slice.stop - cur_cat_slice.start
        cutoff += slice_size

    return cutoff


def parse_tabular_activations_for_feature_importance(
    activation_tensor_slices: Dict,
    all_activations: Sequence["SampleActivation"],
    input_name: str,
) -> Dict[str, List[float]]:
    """
    Note we need to use abs here to get the absolute feature importance, before
    we sum so different signs don't cancel each other out.
    """
    column_activations = {column: [] for column in activation_tensor_slices.keys()}

    for sample_activation in all_activations:
        sample_acts = sample_activation.sample_activations[input_name].squeeze(0)

        for column in activation_tensor_slices.keys():
            cur_slice = activation_tensor_slices[column]
            cur_column_activations = np.abs(sample_acts[cur_slice]).sum()
            column_activations[column].append(cur_column_activations)

    finalized_activations = {}

    for column, aggregated_activations in column_activations.items():
        mean_activation = np.array(column_activations[column]).mean()
        finalized_activations[column] = [mean_activation]

    return finalized_activations


def get_tabular_activation_df(
    parsed_activations: Dict[str, List[float]]
) -> pd.DataFrame:
    df_activations = pd.DataFrame.from_dict(
        parsed_activations, orient="index", columns=["Shap_Value"]
    )
    df_activations = df_activations.sort_values(by="Shap_Value", ascending=False)

    return df_activations


def _gather_continuous_inputs(
    all_activations: Sequence["SampleActivation"],
    cat_to_con_cutoff: int,
    con_names: Sequence[str],
    input_name: str,
) -> pd.DataFrame:
    con_inputs = []

    for sample in all_activations:
        cur_full_input = sample.sample_info.inputs[input_name]
        assert len(cur_full_input.shape) == 2

        cur_con_input_part = cur_full_input[:, cat_to_con_cutoff:]
        con_inputs.append(cur_con_input_part)

    con_inputs_array = np.vstack([np.array(i.cpu()) for i in con_inputs])
    df = pd.DataFrame(con_inputs_array, columns=con_names)

    return df


def _gather_continuous_shap_values(
    all_activations: Sequence["SampleActivation"],
    cat_to_con_cutoff: int,
    input_name: str,
) -> np.ndarray:

    con_acts = []

    for sample in all_activations:
        cur_full_input = sample.sample_activations[input_name]
        assert len(cur_full_input.shape) == 2
        assert cur_full_input.shape[0] == 1

        cur_con_input_part = cur_full_input[:, cat_to_con_cutoff:]
        con_acts.append(cur_con_input_part)

    array_np = np.vstack(con_acts)

    return array_np


def _gather_categorical_shap_values(
    all_activations: Sequence["SampleActivation"],
    cur_slice: slice,
    input_name: str,
) -> np.ndarray:
    """
    We need to expand_dims in the case where we only have one input value, otherwise
    shap will not accept a vector.
    """
    cat_acts = []

    for sample in all_activations:
        cur_full_input = sample.sample_activations[input_name]
        assert len(cur_full_input.shape) == 2
        assert cur_full_input.shape[0] == 1

        cur_input_squeezed = cur_full_input.squeeze(0)

        cur_cat_input_part = cur_input_squeezed[cur_slice]
        cur_cat_summed_act = np.sum(np.array(cur_cat_input_part))
        cat_acts.append(cur_cat_summed_act)

    array_np = np.vstack(cat_acts)

    if len(array_np.shape) == 1:
        return np.expand_dims(array_np, 1)

    return array_np


def _gather_categorical_inputs(
    all_activations: Sequence["SampleActivation"],
    cat_name: str,
    input_name: str,
) -> pd.DataFrame:
    cat_inputs = []

    for sample in all_activations:

        cur_raw_cat_input = sample.raw_inputs[input_name][cat_name]
        cur_cat_input_part = cur_raw_cat_input
        cat_inputs.append(cur_cat_input_part)

    cat_inputs_array = np.array([np.array(i.cpu()) for i in cat_inputs])
    df = pd.DataFrame(cat_inputs_array, columns=[cat_name])

    return df


def map_categorical_labels_to_names(
    cat_column_transformers: Dict[str, LabelEncoder],
    cat_column: str,
    categorical_inputs: pd.DataFrame,
) -> pd.DataFrame:
    cat_column_transformer = cat_column_transformers[cat_column]

    categorical_inputs_copy = categorical_inputs.copy()

    categorical_inputs_copy[cat_column] = cat_column_transformer.inverse_transform(
        categorical_inputs_copy[cat_column]
    )

    return categorical_inputs_copy


def plot_tabular_categorical_feature(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_name_to_plot: str,
    class_name: str,
    activation_outfolder: Path,
):
    """
    We catch the warnings there because shap is causing them in the dependence plot,
    to avoid filling the screen with warnings.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        _ = shap.dependence_plot(
            ind=feature_name_to_plot,
            shap_values=shap_values,
            features=features,
            interaction_index=None,
            color="black",
        )

    fig = plt.gcf()

    plt.title(class_name, fontsize=14)
    plt.yticks(fontsize=10, wrap=True)

    plt.tight_layout()

    fig_name = f"cat_features_{feature_name_to_plot}_{class_name}.pdf"
    output_path = str(activation_outfolder / fig_name)

    try:
        fig.savefig(output_path, bbox_inches="tight")

    # See: https://github.com/matplotlib/matplotlib/issues/17579
    except ZeroDivisionError:
        logger.error(
            "Encountered ZeroDivisionError when saving %s. Skipping.", output_path
        )

    plt.close("all")


def _save_categorical_acts(
    dfs_categorical_acts_for_class: Sequence[pd.DataFrame],
    class_name: str,
    output_folder: Path,
) -> None:

    if len(dfs_categorical_acts_for_class) == 0:
        return None

    df_cat_categorical_acts = pd.concat(dfs_categorical_acts_for_class)
    df_cat_categorical_acts.index.name = "Input_Value"
    csv_name = f"cat_features_{class_name}.csv"
    output_path = str(output_folder / csv_name)
    df_cat_categorical_acts.to_csv(path_or_buf=output_path)


def _parse_categorical_shap_values_for_serialization(
    categorical_inputs_mapped: pd.DataFrame,
    shap_values_for_input: np.ndarray,
    column_name: str,
) -> pd.DataFrame:

    categorical_inputs_copy = categorical_inputs_mapped.copy()

    assert len(categorical_inputs_copy) == shap_values_for_input.shape[0]

    categorical_inputs_copy["Shap_Value"] = shap_values_for_input
    average_effects = categorical_inputs_copy.groupby(by=column_name).mean()
    average_effects["Input_Name"] = average_effects.index.name

    return average_effects


def plot_tabular_beeswarm(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    activation_outfolder: Path,
    class_name: str,
):

    _ = shap.summary_plot(
        shap_values=shap_values,
        features=features,
        show=False,
        title=class_name,
        max_display=20,
    )

    fig = plt.gcf()

    ax = _get_shap_main_axis(figure=fig)
    ax_wrapped_ytick_labels = _get_wrapped_ax_ytick_text(axis=ax)
    ax.set_yticklabels(ax_wrapped_ytick_labels)

    plt.title(class_name, fontsize=14)
    plt.yticks(fontsize=10, wrap=True)

    plt.tight_layout()
    fig.savefig(
        str(activation_outfolder / f"con_features_beeswarm_{class_name}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )

    plt.close("all")


def _get_shap_main_axis(figure: plt.Figure):
    return figure.axes[0]


def _get_wrapped_ax_ytick_text(axis: plt.Axes):
    labels = [item.get_text() for item in axis.get_yticklabels()]
    labels_wrapped = ["\n".join(wrap(label, 20)) for label in labels]

    return labels_wrapped
