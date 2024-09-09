import json
from pathlib import Path

from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo


def load_bytes_input_object(
    serialized_input_folder: Path,
) -> ComputedBytesInputInfo:
    with open(serialized_input_folder / "vocab.json", "r") as f:
        vocab = json.load(f)

    with open(serialized_input_folder / "computed_max_length.json", "r") as f:
        computed_max_length = json.load(f)

    input_config = load_input_config_from_yaml(
        input_config_path=serialized_input_folder / "input_config.yaml"
    )

    loaded_object = ComputedBytesInputInfo(
        input_config=input_config,
        vocab=vocab,
        computed_max_length=computed_max_length,
    )

    return loaded_object
