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
from eir.train_utils.evaluation import (
    plot_individual_survival_curves,
    plot_survival_curves,
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

        transformers = output_object.target_transformers
        time_kbins_transformer = transformers[time_name]
        time_bins = time_kbins_transformer.bin_edges_[0]

        output_folder = Path(predict_cl_args.output_folder, output_name)
        output_folder.mkdir(parents=True, exist_ok=True)

        ids = all_ids[output_name][event_name]
        hazards_logits = all_predictions[output_name][event_name].cpu()

        times_binned = None
        events = None
        if predict_cl_args.evaluate:
            times_binned = all_labels[output_name][time_name].cpu().numpy()
            events = all_labels[output_name][event_name].cpu().numpy()

        hazards = torch.sigmoid(hazards_logits).numpy()
        survival_probs = np.cumprod(1 - hazards, 1)

        time_bins_except_last = time_bins[:-1]

        if predict_cl_args.evaluate:
            assert times_binned is not None
            times = time_kbins_transformer.inverse_transform(
                times_binned.reshape(-1, 1)
            ).flatten()

            plot_survival_curves(
                times=times,
                events=events,
                predicted_probs=survival_probs,
                time_bins=time_bins_except_last,
                output_folder=output_folder,
            )

        df = pd.DataFrame({"ID": ids})

        if predict_cl_args.evaluate:
            event_transformer = transformers[event_name]
            events_untransformed = event_transformer.inverse_transform(events)
            df[time_name] = times
            df[event_name] = events
            df[f"{event_name} Untransformed"] = events_untransformed

        df["Predicted_Risk"] = hazards[:, -1]

        for i, t in enumerate(time_bins_except_last):
            df[f"Surv_Prob_t{i}"] = survival_probs[:, i]

        csv_path = output_folder / "survival_predictions.csv"
        df.to_csv(csv_path, index=False)

        plot_individual_survival_curves(
            df=df,
            time_bins=time_bins_except_last,
            output_folder=str(output_folder),
            n_samples=5,
        )


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

        transformers = output_object.target_transformers
        time_kbins_transformer = transformers[time_name]
        time_bins = time_kbins_transformer.bin_edges_[0]

        output_folder = Path(predict_cl_args.output_folder, output_name)
        output_folder.mkdir(parents=True, exist_ok=True)

        ids = all_ids[output_name][event_name]
        hazards_logits = all_predictions[output_name][event_name].cpu()

        hazards = torch.sigmoid(hazards_logits).numpy()
        survival_probs = np.cumprod(1 - hazards, 1)

        time_bins_except_last = time_bins[:-1]

        df = pd.DataFrame({"ID": ids})
        df["Predicted_Risk"] = hazards[:, -1]

        for i, t in enumerate(time_bins_except_last):
            df[f"Surv_Prob_t{i}"] = survival_probs[:, i]

        csv_path = output_folder / "survival_predictions.csv"
        df.to_csv(csv_path, index=False)

        plot_individual_survival_curves(
            df=df,
            time_bins=time_bins_except_last,
            output_folder=str(output_folder),
            n_samples=5,
        )
