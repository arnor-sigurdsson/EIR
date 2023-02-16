from pathlib import Path
from typing import Dict, Sequence, TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
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
    df_activations = get_tabular_attribution_df(parsed_activations=parsed_activations)
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
            continuous_attr = _gather_continuous_attributions(
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

            plot_tabular_continuous_attribution(
                attributions=continuous_attr,
                df_features=continuous_inputs,
                activation_output_folder=cur_class_outfolder,
                class_name=class_name,
            )

        cat_act_dfs = []
        for cat_column in cat_columns:
            categorical_attr = _gather_categorical_attributions(
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

            plot_tabular_categorical_attributions(
                attributions=categorical_attr,
                df_features=categorical_inputs_mapped,
                feature_name_to_plot=cat_column,
                class_name=class_name,
                activation_output_folder=cur_class_outfolder,
            )

            df_cur_categorical_acts = _parse_categorical_attrs_for_serialization(
                categorical_inputs_mapped=categorical_inputs_mapped,
                attributions_for_input=categorical_attr,
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


def get_tabular_attribution_df(
    parsed_activations: Dict[str, List[float]]
) -> pd.DataFrame:
    df_attributions = pd.DataFrame.from_dict(
        parsed_activations, orient="index", columns=["Attribution"]
    )
    df_attributions = df_attributions.sort_values(by="Attribution", ascending=False)

    return df_attributions


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
    df = pd.DataFrame(con_inputs_array, columns=list(con_names))

    return df


def _gather_continuous_attributions(
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


def _gather_categorical_attributions(
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


def plot_tabular_categorical_attributions(
    attributions: np.ndarray,
    df_features: pd.DataFrame,
    feature_name_to_plot: str,
    class_name: str,
    activation_output_folder: Path,
) -> None:
    """
    attribution: (num_samples, num_features)
    features: (num_samples, num_features)
    """

    df_categorical = df_features[[feature_name_to_plot]]

    num_categories = len(df_categorical[feature_name_to_plot].unique())
    fig_width = max(num_categories * 0.5, 8)
    fig_width = min(fig_width, 20)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    category_values = df_categorical[feature_name_to_plot]
    category_attributions = attributions[
        :, df_features.columns.get_loc(feature_name_to_plot)
    ]
    sns.stripplot(
        x=category_values,
        y=category_attributions,
        color="gray",
        alpha=0.5,
        ax=ax,
    )
    sns.pointplot(
        x=category_values,
        y=category_attributions,
        color="black",
        markers=".",
        scale=1.0,
        errorbar=("ci", 95),
        capsize=0.1,
        linestyles="",
        ax=ax,
    )
    ax.set_title(f"Attribution plot for {feature_name_to_plot}")
    ax.set_xlabel(feature_name_to_plot)
    ax.set_ylabel("Attribution (impact on model output)")

    plt.tight_layout()
    name = f"categorical_attributions_{feature_name_to_plot}_{class_name}.pdf"
    plt.savefig(
        str(activation_output_folder / name),
        bbox_inches="tight",
    )
    plt.close(fig)


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


def _parse_categorical_attrs_for_serialization(
    categorical_inputs_mapped: pd.DataFrame,
    attributions_for_input: np.ndarray,
    column_name: str,
) -> pd.DataFrame:
    categorical_inputs_copy = categorical_inputs_mapped.copy()

    assert len(categorical_inputs_copy) == attributions_for_input.shape[0]

    categorical_inputs_copy["Attribution"] = attributions_for_input
    average_effects = categorical_inputs_copy.groupby(by=column_name).mean()
    average_effects["Input_Name"] = average_effects.index.name

    return average_effects


def plot_tabular_continuous_attribution(
    attributions: np.ndarray,
    df_features: pd.DataFrame,
    class_name: str,
    activation_output_folder: Path,
    top_n_features_to_plot: int = 20,
) -> None:
    """
    attribution: (num_samples, num_features)
    features: (num_samples, num_features)
    """

    feature_importance = np.abs(attributions).mean(axis=0)
    top_n_features = np.argsort(feature_importance)[::-1][:top_n_features_to_plot]

    num_plots = len(top_n_features)
    num_cols = min(num_plots, 4)
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(4 * num_cols, 4 * num_rows),
        squeeze=False,
    )

    for i, feature_index in enumerate(top_n_features):
        row_index = i // num_cols
        col_index = i % num_cols
        feature_name = df_features.columns[feature_index]

        sns.regplot(
            x=df_features[feature_name],
            y=attributions[:, feature_index],
            color="black",
            scatter_kws={"alpha": 0.7},
            order=1,
            ci=95,
            ax=axes[row_index, col_index],
        )
        axes[row_index, col_index].set_xlabel(feature_name)
        axes[row_index, col_index].set_ylabel("")

    fig.text(
        -0.01,
        0.5,
        "Attribution (impact on model output)",
        va="center",
        rotation="vertical",
    )

    plt.tight_layout()
    name = f"continuous_attributions_{class_name}.pdf"
    plt.savefig(
        str(activation_output_folder / name),
        bbox_inches="tight",
    )
    plt.close(fig)
