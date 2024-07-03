import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
)
from captum._utils.typing import TargetType
from captum.attr import IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import _reshape_and_sum
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from torch import Tensor

from eir.setup import schemas
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_transformers_object
    from eir.interpretation.interpretation import SampleAttribution


logger = get_logger(name=__name__)


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
) -> DefaultDict[Any, list["SampleAttribution"]]:
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
    df_attributions: pd.DataFrame,
    output_path: Union[str, Path],
    top_n: int = 20,
    title: str = "",
    use_bootstrap: bool = True,
) -> None:
    df_token_top_n = calculate_top_n_tokens(
        df_attributions=df_attributions,
        top_n=top_n,
        use_bootstrap=use_bootstrap,
    )
    df_attributions_top_n_sorted = df_attributions[
        df_attributions["Input"].isin(df_token_top_n.index)
    ]

    order = (
        df_attributions_top_n_sorted.groupby("Input")
        .mean()
        .sort_values(by="Attribution", ascending=False)
        .index
    )

    ax: plt.Axes = sns.barplot(
        data=df_attributions_top_n_sorted,
        x="Attribution",
        y="Input",
        order=order,
        color="#1f77b4",
        capsize=0.1,
        errorbar=("ci", 95),
    )

    plt.tight_layout()
    sns_figure: Optional[plt.Figure] = ax.get_figure()
    assert sns_figure is not None

    if title:
        ax.set_title(title)

    sns_figure.set_size_inches(10, 0.5 * top_n)
    sns_figure.savefig(fname=output_path, bbox_inches="tight")

    plt.close("all")


def calculate_token_statistics(
    df_attributions: pd.DataFrame,
    use_bootstrap: bool = True,
    n_bootstraps: int = 1000,
) -> pd.DataFrame:
    df_token_stats = pd.DataFrame()
    log_once_callable = get_log_once_callable(logger_=logger)

    for input_feature in df_attributions["Input"].unique():
        feature_attributions = df_attributions[
            df_attributions["Input"] == input_feature
        ]["Attribution"].to_numpy()

        if use_bootstrap:

            if len(np.unique(feature_attributions)) < 2:
                log_once_callable(
                    f"Feature '{input_feature}' has less than 2 unique attribution "
                    f"values; unable to perform bootstrapping. Calculating mean "
                    f"without bootstrapping and setting CI_Size to 0 as a default. "
                    f"There might be other input features with the same issue, this "
                    f"message will only be logged once per attribution calculation."
                )

                df_token_stats.loc[input_feature, "AttributionMean"] = np.mean(
                    feature_attributions
                )
                df_token_stats.loc[input_feature, "CI_Size"] = np.nan
                continue

            res = bootstrap(
                (feature_attributions,),
                np.mean,
                n_resamples=n_bootstraps,
                method="basic",
            )
            mean = np.mean(feature_attributions)
            ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
            ci_size = ci_high - ci_low

            df_token_stats.loc[input_feature, "AttributionMean"] = mean
            df_token_stats.loc[input_feature, "CI_Size"] = ci_size
        else:
            mean = feature_attributions.mean()
            df_token_stats.loc[input_feature, "AttributionMean"] = mean
            df_token_stats.loc[input_feature, "CI_Size"] = 0

    return df_token_stats


def get_log_once_callable(logger_: logging.Logger) -> Callable:
    has_logged = False

    def _do_log_once(msg: str):
        nonlocal has_logged
        if not has_logged:
            logger_.info(msg)
            has_logged = True

    return _do_log_once


def calculate_top_n_tokens(
    df_attributions: pd.DataFrame,
    top_n: int,
    use_bootstrap: bool,
) -> pd.DataFrame:
    df_token_stats = calculate_token_statistics(
        df_attributions=df_attributions,
        use_bootstrap=use_bootstrap,
    )

    if use_bootstrap:
        df_token_stats["AttributionScore"] = df_token_stats["AttributionMean"] - (
            0.5 * df_token_stats["CI_Size"]
        )
    else:
        df_token_stats["AttributionScore"] = df_token_stats["AttributionMean"]

    df_token_stats_top_n = df_token_stats.nlargest(top_n, "AttributionScore")
    df_token_stats_top_n_sorted = df_token_stats_top_n.sort_values(
        by="AttributionScore", ascending=False
    )

    return df_token_stats_top_n_sorted


def filter_and_sort_attributions(
    df_attributions: pd.DataFrame, df_token_top_n: pd.DataFrame
) -> pd.DataFrame:
    df_attributions_top_n = df_attributions[
        df_attributions["Input"].isin(df_token_top_n.index)
    ]

    df_attributions_top_n_sorted = df_attributions_top_n.sort_values(
        by="Attribution", ascending=False
    )

    return df_attributions_top_n_sorted


def get_basic_sample_attributions_to_analyse_generator(
    interpretation_config: schemas.BasicInterpretationConfig,
    all_attributions: list["SampleAttribution"],
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


class MyIntegratedGradients(IntegratedGradients):
    """
    Only change here is casting the input to float32 for MPS devices.
    """

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )

        # flattening grads so that we can multiply it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes, dtype=torch.float32)  # here cast for MPS
            .view(n_steps, 1)
            .to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )

        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiplies_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions


def get_long_format_attribution_df(
    parsed_attributions: dict[str, list[float]]
) -> pd.DataFrame:
    if not isinstance(parsed_attributions, dict):
        raise TypeError("Input must be a dictionary")

    if not parsed_attributions:
        return pd.DataFrame(columns=["Input", "Attribution"])

    series_dict: dict[str, pd.Series] = {}
    for k, v in parsed_attributions.items():
        if not isinstance(v, list):
            raise ValueError(f"Value for key '{k}' must be a list")
        if not all(isinstance(x, (int, float, np.float32, np.float64)) for x in v):
            raise ValueError(f"All values in list {v} for key '{k}' must be numbers")
        series_dict[k] = pd.Series(v)

    df: pd.DataFrame = pd.concat(series_dict)
    df = df.reset_index(level=0).reset_index(drop=True)

    df = df.rename(columns={df.columns[0]: "Input", df.columns[1]: "Attribution"})

    return df
