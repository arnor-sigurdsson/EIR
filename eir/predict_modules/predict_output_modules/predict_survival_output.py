from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

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
    plot_cox_survival_curves,
    plot_discrete_individual_survival_curves,
    plot_discrete_survival_curves,
)

if TYPE_CHECKING:
    from eir.predict import PredictExperiment


def predict_survival_wrapper_with_labels(
    predict_config: "PredictExperiment",
    all_predictions: Dict[str, Dict[str, torch.Tensor]],
    all_labels: Dict[str, Dict[str, torch.Tensor]],
    all_ids: Dict[str, Dict[str, List[str]]],
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

        times = events = None
        if predict_cl_args.evaluate:
            events = all_labels[output_name][event_name].cpu().numpy()
            if model_type == "discrete":
                times_binned = all_labels[output_name][time_name].cpu().numpy()
                transformers = output_object.target_transformers
                time_kbins_transformer = transformers[time_name]
                times = time_kbins_transformer.inverse_transform(
                    times_binned.reshape(-1, 1)
                ).flatten()
            else:
                times = all_labels[output_name][time_name].cpu().numpy()

        if model_type == "discrete":
            transformers = output_object.target_transformers
            time_kbins_transformer = transformers[time_name]
            time_bins = time_kbins_transformer.bin_edges_[0]
            time_bins_except_last = time_bins[:-1]

            hazards = torch.sigmoid(model_outputs).numpy()
            survival_probs = np.cumprod(1 - hazards, 1)

            base_data = {"ID": ids, "Predicted_Risk": hazards[:, -1]}

            if predict_cl_args.evaluate:
                event_transformer = transformers[event_name]
                events_untransformed = event_transformer.inverse_transform(events)
                base_data.update(
                    {
                        time_name: times,
                        event_name: events,
                        f"{event_name} Untransformed": events_untransformed,
                    }
                )

                plot_discrete_survival_curves(
                    times=times,
                    events=events,
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
            risk_scores = model_outputs.numpy()

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

            if predict_cl_args.evaluate:
                base_data.update(
                    {
                        time_name: times,
                        event_name: events,
                    }
                )

                assert times is not None
                assert events is not None
                plot_cox_survival_curves(
                    times=times,
                    events=events,
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
    all_predictions: Dict[str, Dict[str, torch.Tensor]],
    all_ids: Dict[str, Dict[str, List[str]]],
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

            hazards = torch.sigmoid(model_outputs).numpy()
            survival_probs = np.cumprod(1 - hazards, 1)

            df = pd.DataFrame(
                {
                    "ID": ids,
                    "Predicted_Risk": hazards[:, -1],
                }
            )

            for i, t in enumerate(time_bins_except_last):
                df[f"Surv_Prob_t{i}"] = survival_probs[:, i]

            plot_discrete_individual_survival_curves(
                df=df,
                time_bins=time_bins_except_last,
                output_folder=str(output_folder),
                n_samples=5,
            )

        else:
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
