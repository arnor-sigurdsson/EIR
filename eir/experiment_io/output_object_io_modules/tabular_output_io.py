from pathlib import Path

from eir.experiment_io.label_transformer_io import load_transformers
from eir.experiment_io.output_object_io_modules.output_io_utils import (
    load_output_config_from_yaml,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
    set_up_tabular_output,
)


def load_tabular_output_object(
    serialized_output_folder: Path,
    run_folder: Path,
) -> ComputedTabularOutputInfo:
    config_path = serialized_output_folder / "output_config.yaml"

    transformers = load_transformers(run_folder=run_folder)

    output_config = load_output_config_from_yaml(output_config_path=config_path)

    loaded_object = set_up_tabular_output(
        output_config=output_config,
        target_transformers=transformers,
    )

    return loaded_object
