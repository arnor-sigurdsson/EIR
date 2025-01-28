from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

from eir.experiment_io.output_object_io import get_output_serialization_path
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.train_utils import utils
from eir.train_utils.metrics import (
    al_step_metric_dict,
    filter_survival_missing_targets,
    general_torch_to_numpy,
)

if TYPE_CHECKING:
    from eir.train import Experiment


def save_survival_evaluation_results_wrapper(
    val_outputs: dict[str, dict[str, torch.Tensor]],
    val_labels: dict[str, dict[str, torch.Tensor]],
    val_ids: dict[str, dict[str, list[str]]],
    iteration: int,
    experiment: "Experiment",
    evaluation_metrics: al_step_metric_dict,
) -> None:
    for output_name, output_object in experiment.outputs.items():
        output_type = output_object.output_config.output_info.output_type
        if output_type != "survival":
            continue

        assert isinstance(output_object, ComputedSurvivalOutputInfo)

        output_type_info = output_object.output_config.output_type_info
        time_name = output_type_info.time_column
        event_name = output_type_info.event_column
        model_type = (
            "cox" if output_type_info.loss_function == "CoxPHLoss" else "discrete"
        )

        cur_sample_output_folder = utils.prepare_sample_output_folder(
            output_folder=experiment.configs.gc.be.output_folder,
            column_name=event_name,
            output_name=output_name,
            iteration=iteration,
        )

        ids = val_ids[output_name][event_name]
        model_outputs = val_outputs[output_name][event_name]
        events = val_labels[output_name][event_name]
        times = val_labels[output_name][time_name]

        filtered = filter_survival_missing_targets(
            model_outputs=model_outputs,
            events=events,
            times=times,
            cur_ids=ids,
        )

        model_outputs = filtered.model_outputs
        events = general_torch_to_numpy(tensor=filtered.events)
        events = events.astype(int)
        times = general_torch_to_numpy(tensor=filtered.times)
        ids = filtered.ids

        if model_type == "discrete":
            transformers = output_object.target_transformers
            time_kbins_transformer = transformers[time_name]
            time_bins = time_kbins_transformer.bin_edges_[0]
            times_binned = times
            it_func = time_kbins_transformer.inverse_transform
            times = it_func(times_binned.reshape(-1, 1)).flatten()

            hazards = torch.sigmoid(model_outputs).cpu().numpy()
            survival_probs = np.cumprod(1 - hazards, 1)
            time_bins_except_last = time_bins[:-1]

            plot_discrete_survival_curves(
                times=times,
                events=events,
                predicted_probs=survival_probs,
                time_bins=time_bins_except_last,
                output_folder=cur_sample_output_folder,
            )

            plot_discrete_risk_stratification(
                times=times,
                events=events,
                predicted_probs=survival_probs,
                time_bins=time_bins_except_last,
                output_folder=cur_sample_output_folder,
            )

            df = pd.DataFrame(
                {
                    "ID": ids,
                    time_name: times,
                    event_name: events,
                    "Predicted_Risk": hazards[:, -1],
                }
            )

            for i, t in enumerate(time_bins_except_last):
                df[f"Surv_Prob_t{i}"] = survival_probs[:, i]

            plot_discrete_individual_survival_curves(
                df=df,
                time_bins=time_bins_except_last,
                output_folder=str(cur_sample_output_folder),
                n_samples=5,
            )

        else:
            risk_scores = model_outputs.cpu().numpy()

            unique_times, baseline_hazard = estimate_baseline_hazard(
                times=times,
                events=events,
                risk_scores=risk_scores,
            )
            baseline_survival = np.exp(-np.cumsum(baseline_hazard))

            run_folder = Path(experiment.configs.gc.be.output_folder)
            cur_serialization_path = get_output_serialization_path(
                output_name=output_name,
                output_type=output_type,
                run_folder=Path(experiment.configs.gc.be.output_folder),
            )

            em = evaluation_metrics
            cur_performance_avg = em["average"]["average"]["perf-average"]

            maybe_save_survival_hazards_and_times(
                serialization_output_folder=cur_serialization_path,
                run_folder=run_folder,
                cur_performance=cur_performance_avg,
                baseline_hazards=baseline_hazard,
                unique_times=unique_times,
            )

            max_time = np.max(times)
            time_points = np.linspace(0, max_time, 100)

            survival_probs = calculate_cox_survival_probs(
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
            )

            plot_cox_survival_curves(
                times=times,
                events=events,
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
                output_folder=cur_sample_output_folder,
            )

            plot_cox_risk_stratification(
                times=times,
                events=events,
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
                output_folder=cur_sample_output_folder,
            )

            base_data = {
                "ID": ids,
                time_name: times,
                event_name: events,
                "Risk_Score": risk_scores.squeeze(),
            }

            surv_prob_cols = {
                f"Surv_Prob_t{i}": survival_probs[:, i]
                for i in range(survival_probs.shape[1])
            }

            df = pd.DataFrame({**base_data, **surv_prob_cols})

            plot_cox_individual_curves(
                df=df,
                time_points=time_points,
                output_folder=str(cur_sample_output_folder),
                n_samples=5,
            )

        csv_path = f"{cur_sample_output_folder}/survival_predictions.csv"
        df.to_csv(csv_path, index=False)


def maybe_save_survival_hazards_and_times(
    serialization_output_folder: Path,
    run_folder: Path,
    cur_performance: float,
    baseline_hazards: np.ndarray,
    unique_times: np.ndarray,
) -> None:

    sof = serialization_output_folder
    baseline_hazard_path = sof / "baseline_hazard.npy"
    baseline_unique_times_path = sof / "baseline_unique_times.npy"

    if not baseline_hazard_path.exists():
        np.save(baseline_hazard_path, baseline_hazards)
        np.save(baseline_unique_times_path, unique_times)
    else:
        is_current_iter_best = check_is_current_iter_best(
            run_folder=run_folder,
            cur_performance=cur_performance,
        )

        if is_current_iter_best:
            np.save(baseline_hazard_path, baseline_hazards)
            np.save(baseline_unique_times_path, unique_times)


def check_is_current_iter_best(
    run_folder: Path,
    cur_performance: float,
) -> bool:
    validation_history_file = run_folder / "validation_average_history.log"
    df = pd.read_csv(validation_history_file)

    best_performance = df["perf-average"].max()
    return cur_performance >= best_performance


def estimate_baseline_hazard(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    events = events[sort_idx]
    risk_scores = risk_scores[sort_idx]

    unique_times = np.unique(times[events == 1])
    baseline_hazard = np.zeros_like(unique_times)

    risk_scores_exp = np.exp(risk_scores)

    for i, t in enumerate(unique_times):
        events_at_t = events[times == t]
        n_events = np.sum(events_at_t)

        at_risk = times >= t
        risk_set_sum = np.sum(risk_scores_exp[at_risk])

        if risk_set_sum > 0:
            baseline_hazard[i] = n_events / risk_set_sum

    return unique_times, baseline_hazard


def calculate_cox_survival_probs(
    risk_scores: np.ndarray,
    unique_times: np.ndarray,
    baseline_survival: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    interpolated_baseline = np.interp(
        time_points, unique_times, baseline_survival, right=baseline_survival[-1]
    )

    survival_probs = np.zeros((len(risk_scores), len(time_points)))
    for i, risk_score in enumerate(risk_scores):
        survival_probs[i] = interpolated_baseline ** np.exp(risk_score)

    return survival_probs


def plot_cox_survival_curves(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    unique_times: np.ndarray,
    baseline_survival: np.ndarray,
    time_points: np.ndarray,
    output_folder: Path,
) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(times, events)

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function(label="Kaplan-Meier Estimate")

    risk_quantiles = np.percentile(risk_scores, [25, 50, 75])

    for q, risk in zip(
        ["Low Risk (25%)", "Median Risk", "High Risk (75%)"], risk_quantiles
    ):
        interp_baseline = np.interp(time_points, unique_times, baseline_survival)
        surv = interp_baseline ** np.exp(risk)
        plt.plot(time_points, surv, label=f"{q}")

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.title("Kaplan-Meier vs Model Predictions")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/cox_survival_curves.pdf")
    plt.close()


def plot_cox_risk_stratification(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    unique_times: np.ndarray,
    baseline_survival: np.ndarray,
    time_points: np.ndarray,
    output_folder: Path,
) -> None:
    risk_percentiles = [25, 50, 75]
    risk_thresholds = np.percentile(risk_scores, risk_percentiles)

    colors = ["tab:orange", "tab:green", "tab:red"]
    group_labels = ["Low Risk (25%)", "Median Risk", "High Risk (75%)"]

    plt.figure(figsize=(12, 8))

    for q, risk, color, label in zip(
        risk_percentiles,
        risk_thresholds,
        colors,
        group_labels,
    ):
        interp_baseline = np.interp(time_points, unique_times, baseline_survival)
        surv = interp_baseline ** np.exp(risk)
        plt.plot(
            time_points,
            surv,
            color=color,
            linestyle="--",
            label=f"{label} (predicted)",
        )

    kmf = KaplanMeierFitter()

    mask_low = risk_scores <= risk_thresholds[0]
    mask_low = mask_low.squeeze()
    kmf.fit(times[mask_low], events[mask_low], label="KM Low Risk (actual)")
    kmf.plot_survival_function(color="tab:orange")

    mask_med = (risk_scores > risk_thresholds[0]) & (risk_scores <= risk_thresholds[2])
    mask_med = mask_med.squeeze()
    kmf.fit(times[mask_med], events[mask_med], label="KM Median Risk (actual)")
    kmf.plot_survival_function(color="tab:green")

    mask_high = risk_scores > risk_thresholds[2]
    mask_high = mask_high.squeeze()
    kmf.fit(times[mask_high], events[mask_high], label="KM High Risk (actual)")
    kmf.plot_survival_function(color="tab:red")

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Risk Stratification: Predicted vs Actual Survival")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/cox_risk_stratification.pdf", bbox_inches="tight")
    plt.close()


def plot_cox_individual_curves(
    df: pd.DataFrame,
    time_points: np.ndarray,
    output_folder: str,
    n_samples: int = 5,
) -> None:
    random_samples = df.sample(n=n_samples)

    plt.figure(figsize=(12, 8))

    for _, row in random_samples.iterrows():
        surv_probs = row[
            [col for col in df.columns if col.startswith("Surv_Prob_t")]
        ].to_numpy()
        plt.plot(
            time_points,
            surv_probs,
            label=f"ID: {row['ID']} (Risk Score: {row['Risk_Score']:.2f})",
        )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title(f"Individual Survival Curves for {n_samples} Random Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/cox_individual_curves.pdf")
    plt.close()


def plot_discrete_survival_curves(
    times: np.ndarray,
    events: np.ndarray,
    predicted_probs: np.ndarray,
    time_bins: np.ndarray,
    output_folder: Path,
):
    n_intervals = len(time_bins)
    at_risk = np.zeros(n_intervals)
    events_in_bin = np.zeros(n_intervals)

    bin_edges = np.append(time_bins, np.inf)

    for i in range(n_intervals):
        mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
        at_risk[i] = np.sum(times >= bin_edges[i])
        events_in_bin[i] = np.sum(events[mask])

    hazard = events_in_bin / at_risk
    actual_survival = np.cumprod(1 - hazard)

    plt.figure(figsize=(10, 6))
    plt.step(time_bins, actual_survival, where="post", label="Actual Survival")
    plt.step(
        time_bins, predicted_probs.mean(axis=0), where="post", label="Mean Predicted"
    )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.title("Discrete-time Survival: Actual vs Predicted")
    plt.savefig(f"{output_folder}/survival_curves.pdf")
    plt.close()


def plot_discrete_risk_stratification(
    times: np.ndarray,
    events: np.ndarray,
    predicted_probs: np.ndarray,
    time_bins: np.ndarray,
    output_folder: Path,
) -> None:
    risk_scores = predicted_probs.mean(axis=1)

    risk_percentiles = [75, 50, 25]
    risk_thresholds = np.percentile(risk_scores, [25, 50, 75])

    colors = ["tab:orange", "tab:green", "tab:red"]
    group_labels = ["Low Risk (75%)", "Median Risk", "High Risk (25%)"]

    plt.figure(figsize=(12, 8))

    bin_edges = np.append(time_bins, np.inf)
    n_intervals = len(time_bins)

    for risk_level, color, label in zip(risk_percentiles, colors, group_labels):
        # Low risk (top 25% survival probability)
        if risk_level == 75:
            mask = risk_scores > risk_thresholds[2]
        # Medium risk
        elif risk_level == 50:
            mask = (risk_scores > risk_thresholds[0]) & (
                risk_scores <= risk_thresholds[2]
            )
        # High risk (bottom 25% survival probability)
        else:
            mask = risk_scores <= risk_thresholds[0]

        mean_pred = predicted_probs[mask].mean(axis=0)
        plt.step(
            time_bins,
            mean_pred,
            where="post",
            color=color,
            linestyle="--",
            label=f"{label} (predicted)",
        )

        # Calculate actual survival for this risk group
        times_group = times[mask]
        events_group = events[mask]

        at_risk = np.zeros(n_intervals)
        events_in_bin = np.zeros(n_intervals)

        for i in range(n_intervals):
            bin_mask = (times_group >= bin_edges[i]) & (times_group < bin_edges[i + 1])
            at_risk[i] = np.sum(times_group >= bin_edges[i])
            events_in_bin[i] = np.sum(events_group[bin_mask])

        hazard = np.divide(
            events_in_bin,
            at_risk,
            out=np.zeros_like(events_in_bin),
            where=at_risk != 0,
        )
        actual_survival = np.cumprod(1 - hazard)

        plt.step(
            time_bins,
            actual_survival,
            where="post",
            color=color,
            label=f"{label} (actual)",
        )

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Risk Stratification: Predicted vs Actual Survival")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        f"{output_folder}/discrete_risk_stratification.pdf",
        bbox_inches="tight",
    )
    plt.close()


def plot_discrete_individual_survival_curves(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_folder: str,
    n_samples: int = 5,
) -> None:
    random_samples = df.sample(n=n_samples)

    plt.figure(figsize=(12, 8))

    for _, row in random_samples.iterrows():
        surv_probs = row[
            [col for col in df.columns if col.startswith("Surv_Prob_t")]
        ].to_numpy()

        plt.step(x=time_bins, y=surv_probs, where="post", label=f"ID: {row['ID']}")

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title(f"Survival Curves for {n_samples} Random Samples")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f"{output_folder}/individual_survival_curves.pdf")
    plt.close()
