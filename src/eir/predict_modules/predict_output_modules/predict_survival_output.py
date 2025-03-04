from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.schemas import SurvivalOutputTypeConfig
from eir.train_utils.evaluation_modules.evaluation_output_survival import (
    calculate_cox_survival_probs,
    plot_cox_individual_curves,
    plot_cox_risk_stratification,
    plot_cox_survival_curves,
    plot_discrete_individual_survival_curves,
    plot_discrete_risk_stratification,
    plot_discrete_survival_curves,
)
from eir.train_utils.metrics import (
    filter_survival_missing_targets,
    general_torch_to_numpy,
)

if TYPE_CHECKING:
    from eir.predict import PredictExperiment


def predict_survival_wrapper_with_labels(
    predict_config: "PredictExperiment",
    all_predictions: dict[str, dict[str, torch.Tensor]],
    all_labels: dict[str, dict[str, torch.Tensor]],
    all_ids: dict[str, dict[str, list[str]]],
    predict_cl_args: Namespace,
) -> None:
    for output_name, output_object in predict_config.outputs.items():
        if not isinstance(output_object, ComputedSurvivalOutputInfo):
            continue

        output_type_info = output_object.output_config.output_type_info
        assert isinstance(output_type_info, SurvivalOutputTypeConfig)

        time_name = output_type_info.time_column
        event_name = output_type_info.event_column
        model_type = (
            "cox" if output_type_info.loss_function == "CoxPHLoss" else "discrete"
        )

        output_folder = Path(predict_cl_args.output_folder, output_name)
        output_folder.mkdir(parents=True, exist_ok=True)

        ids = all_ids[output_name][event_name]
        model_outputs = all_predictions[output_name][event_name]

        if model_type == "discrete":
            times_binned = all_labels[output_name][time_name].cpu().numpy()
            transformers = output_object.target_transformers
            time_kbins_transformer = transformers[time_name]
            it_func = time_kbins_transformer.inverse_transform
            times = it_func(times_binned.reshape(-1, 1)).flatten()
            times = torch.tensor(times).to(device=model_outputs.device)
        else:
            times = all_labels[output_name][time_name]

        events = all_labels[output_name][event_name]

        filtered = filter_survival_missing_targets(
            model_outputs=model_outputs,
            events=events,
            times=times,
            cur_ids=ids,
        )

        model_outputs = filtered.model_outputs
        events_np = general_torch_to_numpy(tensor=filtered.events)
        events_np = events_np.astype(int)

        times_np = general_torch_to_numpy(tensor=filtered.times)
        ids = filtered.ids

        if model_type == "discrete":
            transformers = output_object.target_transformers
            time_kbins_transformer = transformers[time_name]
            time_bins = time_kbins_transformer.bin_edges_[0]
            time_bins_except_last = time_bins[:-1]

            hazards = torch.sigmoid(model_outputs).cpu().numpy()
            survival_probs = np.cumprod(1 - hazards, 1)

            base_data = {"ID": ids, "Predicted_Risk": hazards[:, -1]}

            event_transformer = transformers[event_name]
            events_untransformed = event_transformer.inverse_transform(events_np)
            base_data.update(
                {
                    time_name: times_np,
                    event_name: events_np,
                    f"{event_name} Untransformed": events_untransformed,
                }
            )

            plot_discrete_survival_curves(
                times=times_np,
                events=events_np,
                predicted_probs=survival_probs,
                time_bins=time_bins_except_last,
                output_folder=output_folder,
            )

            plot_discrete_risk_stratification(
                times=times_np,
                events=events_np,
                predicted_probs=survival_probs,
                time_bins=time_bins_except_last,
                output_folder=output_folder,
            )

            surv_prob_cols = {
                f"Surv_Prob_t{i}": survival_probs[:, i]
                for i in range(survival_probs.shape[1])
            }

            df = pd.DataFrame({**base_data, **surv_prob_cols})

            plot_discrete_individual_survival_curves(
                df=df,
                time_bins=time_bins_except_last,
                output_folder=str(output_folder),
                n_samples=5,
            )

        else:
            # note: model outputs are log hazard ratios
            risk_scores = model_outputs.cpu().numpy()

            baseline_hazard = output_object.baseline_hazard
            unique_times = output_object.baseline_unique_times
            assert baseline_hazard is not None
            assert unique_times is not None

            baseline_survival = np.exp(-np.cumsum(baseline_hazard))

            max_time = unique_times.max()
            time_points = np.linspace(0, max_time, 100)

            survival_probs = calculate_cox_survival_probs(
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
            )

            base_data = {"ID": ids, "Risk_Score": risk_scores.squeeze()}
            base_data.update(
                {
                    time_name: times_np,
                    event_name: events_np,
                }
            )

            plot_cox_survival_curves(
                times=times_np,
                events=events_np,
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
                output_folder=output_folder,
            )

            plot_cox_risk_stratification(
                times=times_np,
                events=events_np,
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
                output_folder=output_folder,
            )

            surv_prob_cols = {
                f"Surv_Prob_t{i}": survival_probs[:, i]
                for i in range(survival_probs.shape[1])
            }

            df = pd.DataFrame({**base_data, **surv_prob_cols})

            plot_cox_individual_curves(
                df=df,
                time_points=time_points,
                output_folder=str(output_folder),
                n_samples=5,
            )

        csv_path = output_folder / "survival_predictions.csv"
        df.to_csv(csv_path, index=False)


def predict_survival_wrapper_no_labels(
    predict_config: "PredictExperiment",
    all_predictions: dict[str, dict[str, torch.Tensor]],
    all_ids: dict[str, dict[str, list[str]]],
    predict_cl_args: Namespace,
) -> None:
    for output_name, output_object in predict_config.outputs.items():
        if not isinstance(output_object, ComputedSurvivalOutputInfo):
            continue

        output_type_info = output_object.output_config.output_type_info
        assert isinstance(output_type_info, SurvivalOutputTypeConfig)

        time_name = output_type_info.time_column
        event_name = output_type_info.event_column
        model_type = (
            "cox" if output_type_info.loss_function == "CoxPHLoss" else "discrete"
        )

        output_folder = Path(predict_cl_args.output_folder, output_name)
        output_folder.mkdir(parents=True, exist_ok=True)

        ids = all_ids[output_name][event_name]
        model_outputs = all_predictions[output_name][event_name].cpu()

        if model_type == "discrete":
            transformers = output_object.target_transformers
            time_kbins_transformer = transformers[time_name]
            time_bins = time_kbins_transformer.bin_edges_[0]
            time_bins_except_last = time_bins[:-1]

            hazards = torch.sigmoid(model_outputs).cpu().numpy()
            survival_probs = np.cumprod(1 - hazards, 1)

            base_df = pd.DataFrame(
                {
                    "ID": ids,
                    "Predicted_Risk": hazards[:, -1],
                }
            )

            surv_prob_columns = {
                f"Surv_Prob_t{i}": survival_probs[:, i]
                for i, _t in enumerate(time_bins_except_last)
            }
            surv_prob_df = pd.DataFrame(surv_prob_columns)
            df = pd.concat([base_df, surv_prob_df], axis=1)

            plot_discrete_individual_survival_curves(
                df=df,
                time_bins=time_bins_except_last,
                output_folder=str(output_folder),
                n_samples=5,
            )

        else:
            # note: model outputs are log hazard ratios
            risk_scores = model_outputs.cpu().numpy()

            baseline_hazard = output_object.baseline_hazard
            unique_times = output_object.baseline_unique_times
            assert baseline_hazard is not None
            assert unique_times is not None

            baseline_survival = np.exp(-np.cumsum(baseline_hazard))

            max_time = unique_times.max()
            time_points = np.linspace(0, max_time, 100)

            survival_probs = calculate_cox_survival_probs(
                risk_scores=risk_scores,
                unique_times=unique_times,
                baseline_survival=baseline_survival,
                time_points=time_points,
            )

            base_data = {
                "ID": ids,
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
                output_folder=str(output_folder),
                n_samples=5,
            )

        csv_path = output_folder / "survival_predictions.csv"
        df.to_csv(csv_path, index=False)
