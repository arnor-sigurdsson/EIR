from pathlib import Path
from typing import Dict, Sequence, TYPE_CHECKING, List

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from snp_pred.train import Config
    from snp_pred.interpretation.interpretation import SampleActivation


def analyze_tabular_input_activations(
    config: "Config",
    input_name: str,
    activation_outfolder: Path,
    all_activations: Sequence["SampleActivation"],
):

    tabular_model = config.model.modules_to_fuse[input_name]
    activation_tensor_slices = set_up_tabular_tensor_slices(
        cat_input_columns=config.cl_args.extra_cat_columns,
        con_input_columns=config.cl_args.extra_con_columns,
        embedding_module=tabular_model,
    )

    parsed_activations = parse_tabular_activations(
        activation_tensor_slices=activation_tensor_slices,
        all_activations=all_activations,
        input_name=input_name,
    )

    df_activations = get_tabular_activation_df(parsed_activations=parsed_activations)

    plot_tabular_activations(
        df_activations=df_activations, activation_outfolder=activation_outfolder
    )

    cat_to_con_cutoff = get_cat_to_con_cutoff_from_slices(
        slices=activation_tensor_slices,
        cat_input_columns=config.cl_args.extra_cat_columns,
    )
    continuous_shap = _gather_continuous_shap_values(
        all_activations=all_activations,
        cat_to_con_cutoff=cat_to_con_cutoff,
        input_name=input_name,
    )
    continuous_inputs = _gather_continuous_inputs(
        all_activations=all_activations,
        cat_to_con_cutoff=cat_to_con_cutoff,
        con_names=config.cl_args.extra_con_columns,
        input_name=input_name,
    )
    plot_tabular_beeswarm(
        shap_values=continuous_shap,
        features=continuous_inputs,
        activation_outfolder=activation_outfolder,
    )


def set_up_tabular_tensor_slices(
    cat_input_columns: Sequence[str],
    con_input_columns: Sequence[str],
    embedding_module,
) -> Dict:
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
    slices: Dict, cat_input_columns: Sequence[str]
) -> int:
    cutoff = 0

    for cat_column in cat_input_columns:
        cur_cat_slice = slices[cat_column]
        slice_size = cur_cat_slice.stop - cur_cat_slice.start
        cutoff += slice_size

    return cutoff


def parse_tabular_activations(
    activation_tensor_slices: Dict,
    all_activations: Sequence["SampleActivation"],
    input_name: str,
) -> Dict[str, List[float]]:
    column_activations = {column: [] for column in activation_tensor_slices.keys()}

    for sample_activation in all_activations:
        sample_acts = sample_activation.sample_activations[input_name].squeeze()

        for column in activation_tensor_slices.keys():
            cur_slice = activation_tensor_slices[column]
            cur_column_activations = sample_acts[cur_slice].sum()
            column_activations[column].append(cur_column_activations)

    finalized_activations = {}

    for column, aggregated_activations in column_activations.items():
        mean_activation = np.abs(np.array(column_activations[column]).mean())
        finalized_activations[column] = [mean_activation]

    return finalized_activations


def plot_tabular_beeswarm(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    activation_outfolder: Path,
):

    plt.close("all")

    _ = shap.summary_plot(
        shap_values=shap_values,
        features=features,
        show=False,
        max_display=20,
    )

    fig = plt.gcf()
    plt.tight_layout()
    fig.savefig(str(activation_outfolder / "con_features_beeswarm.png"))


def _gather_continuous_inputs(
    all_activations: Sequence["SampleActivation"],
    cat_to_con_cutoff: int,
    con_names: Sequence[str],
    input_name: str,
) -> pd.DataFrame:
    con_inputs = []

    for sample in all_activations:
        cur_full_input = sample.sample_info.inputs[input_name]
        cur_con_input_part = cur_full_input.squeeze()[cat_to_con_cutoff:]
        con_inputs.append(cur_con_input_part)

    con_inputs_array = np.array([np.array(i.cpu()) for i in con_inputs])
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
        cur_con_input_part = cur_full_input.squeeze()[cat_to_con_cutoff:]
        con_acts.append(cur_con_input_part)

    return np.array(con_acts).squeeze()


def get_tabular_activation_df(
    parsed_activations: Dict[str, List[float]]
) -> pd.DataFrame:
    df_activations = pd.DataFrame.from_dict(
        parsed_activations, orient="index", columns=["Shap_Value"]
    )
    df_activations = df_activations.sort_values(by="Shap_Value", ascending=False)

    return df_activations


def plot_tabular_activations(
    df_activations: pd.DataFrame, activation_outfolder: Path
) -> None:

    sns_plot = sns.barplot(
        x=df_activations["Shap_Value"],
        y=df_activations.index,
        palette="Blues_d",
    )
    plt.tight_layout()
    sns_figure = sns_plot.get_figure()
    sns_figure.set_size_inches(10, 0.5 * len(df_activations))
    sns_figure.savefig(str(activation_outfolder / "feature_importance.png"))
