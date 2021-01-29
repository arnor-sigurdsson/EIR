from pathlib import Path
from typing import Dict, Sequence, TYPE_CHECKING, List

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
        mean_activation = np.array(column_activations[column]).mean()
        finalized_activations[column] = [mean_activation]

    return finalized_activations


def plot_tabular_activations(
    df_activations: pd.DataFrame, activation_outfolder: Path
) -> None:

    sns_plot = sns.barplot(
        x=df_activations["Feature_Importance"],
        y=df_activations.index,
        palette="Blues_d",
    )
    plt.tight_layout()
    sns_figure = sns_plot.get_figure()
    sns_figure.savefig(str(activation_outfolder / "feature_importance.png"))


def get_tabular_activation_df(
    parsed_activations: Dict[str, List[float]]
) -> pd.DataFrame:
    df_activations = pd.DataFrame.from_dict(
        parsed_activations, orient="index", columns=["Feature_Importance"]
    )
    df_activations = df_activations.sort_values(
        by="Feature_Importance", ascending=False
    )

    return df_activations
