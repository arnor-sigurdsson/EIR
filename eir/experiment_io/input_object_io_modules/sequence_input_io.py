import json
from copy import deepcopy
from pathlib import Path

from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.setup.input_setup_modules.setup_sequence import (
    ComputedSequenceInputInfo,
    set_up_computed_sequence_input,
)
from eir.setup.schemas import SequenceInputDataConfig


def load_sequence_input_object(
    serialized_input_folder: Path,
) -> ComputedSequenceInputInfo:
    config_path = serialized_input_folder / "input_config.yaml"
    vocab_ordered_path = serialized_input_folder / "vocab_ordered.txt"
    computed_max_length_path = serialized_input_folder / "computed_max_length.json"

    input_config = load_input_config_from_yaml(input_config_path=config_path)
    computed_max_length = json.loads(computed_max_length_path.read_text())

    input_config_modified = deepcopy(input_config)
    input_type_info_modified = deepcopy(input_config.input_type_info)
    assert isinstance(input_type_info_modified, SequenceInputDataConfig)

    input_type_info_modified.vocab_file = str(vocab_ordered_path)
    input_type_info_modified.max_length = computed_max_length

    input_config_modified.input_type_info = input_type_info_modified

    loaded_object = set_up_computed_sequence_input(input_config=input_config_modified)

    return loaded_object
