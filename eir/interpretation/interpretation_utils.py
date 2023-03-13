import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Dict, Generator

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from eir.setup import schemas

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_transformers_object
    from eir.interpretation.interpretation import SampleAttribution


def get_target_class_name(
    sample_label: torch.Tensor,
    target_transformer: "al_label_transformers_object",
    column_type: str,
    target_column_name: str,
):
    if column_type == "con":
        return target_column_name

    tt_it = target_transformer.inverse_transform
    cur_trn_label = tt_it([sample_label.item()])[0]

    return cur_trn_label


def stratify_attributions_by_target_classes(
    all_attributions: Sequence["SampleAttribution"],
    target_transformer: "al_label_transformers_object",
    output_name: str,
    target_column: str,
    column_type: str,
) -> Dict[str, Sequence["SampleAttribution"]]:
    all_attributions_target_class_stratified = defaultdict(list)

    for sample in all_attributions:
        cur_labels_all = sample.sample_info.target_labels
        cur_labels = cur_labels_all[output_name][target_column]
        cur_label_name = get_target_class_name(
            sample_label=cur_labels,
            target_transformer=target_transformer,
            column_type=column_type,
            target_column_name=target_column,
        )
        all_attributions_target_class_stratified[cur_label_name].append(sample)

    return all_attributions_target_class_stratified


def plot_attributions_bar(
    df_attributions: pd.DataFrame, outpath: Path, top_n: int = 20, title: str = ""
) -> None:
    df_token_means = df_attributions.groupby("Input").mean()
    df_token_top_n = df_token_means.nlargest(top_n, "Attribution")
    df_token_top_n_sorted = df_token_top_n.sort_values(
        by="Attribution", ascending=False
    )

    df_attributions_top_n = df_attributions[
        df_attributions["Input"].isin(df_token_top_n_sorted.index)
    ]

    df_attributions_top_n_sorted = df_attributions_top_n.sort_values(
        by="Attribution", ascending=False
    )

    ax: plt.Axes = sns.barplot(
        data=df_attributions_top_n_sorted,
        x="Attribution",
        y="Input",
        order=df_token_top_n.index,
        palette="Blues_d",
        errcolor=".2",
        capsize=0.1,
        errorbar=("ci", 95),
    )

    plt.tight_layout()
    sns_figure: plt.Figure = ax.get_figure()

    if title:
        ax.set_title(title)

    sns_figure.set_size_inches(10, 0.5 * top_n)
    sns_figure.savefig(fname=outpath, bbox_inches="tight")

    plt.close("all")


def get_basic_sample_attributions_to_analyse_generator(
    interpretation_config: schemas.BasicInterpretationConfig,
    all_attributions: Sequence["SampleAttribution"],
) -> Generator["SampleAttribution", None, None]:
    strategy = interpretation_config.interpretation_sampling_strategy
    n_samples = interpretation_config.num_samples_to_interpret

    if strategy == "first_n":
        base = all_attributions[:n_samples]
    elif strategy == "random_sample":
        base = random.sample(all_attributions, n_samples)
    else:
        raise ValueError(f"Unrecognized option for sequence sampling: {strategy}.")

    manual_samples = interpretation_config.manual_samples_to_interpret
    if manual_samples:
        for attribution in all_attributions:
            if attribution.sample_info.ids in manual_samples:
                base.append(attribution)

    for item in base:
        yield item
