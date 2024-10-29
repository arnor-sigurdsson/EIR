from pathlib import Path

import numpy as np

from eir.experiment_io.label_transformer_io import load_transformers
from eir.experiment_io.output_object_io_modules.output_io_utils import (
    load_output_config_from_yaml,
)
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
    set_up_survival_output,
)


def load_survival_output_object(
    serialized_output_folder: Path,
    run_folder: Path,
) -> ComputedSurvivalOutputInfo:
    config_path = serialized_output_folder / "output_config.yaml"

    transformers = load_transformers(run_folder=run_folder)

    output_config = load_output_config_from_yaml(output_config_path=config_path)

    baseline_hazard = None
    if (serialized_output_folder / "baseline_hazard.npy").exists():
        baseline_hazard = np.load(serialized_output_folder / "baseline_hazard.npy")

    baseline_unique_times = None
    if (serialized_output_folder / "baseline_unique_times.npy").exists():
        baseline_unique_times = np.load(
            serialized_output_folder / "baseline_unique_times.npy"
        )

    loaded_object = set_up_survival_output(
        output_config=output_config,
        target_transformers=transformers,
        baseline_hazard=baseline_hazard,
        baseline_unique_times=baseline_unique_times,
    )

    return loaded_object
