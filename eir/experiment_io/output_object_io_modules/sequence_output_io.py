import json
from copy import deepcopy
from pathlib import Path

from eir.experiment_io.input_object_io import get_input_serialization_path
from eir.experiment_io.input_object_io_modules.sequence_input_io import (
    load_sequence_input_object,
)
from eir.experiment_io.output_object_io_modules.output_io_utils import (
    load_output_config_from_yaml,
)
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
    set_up_sequence_output,
)
from eir.setup.schemas import SequenceOutputTypeConfig


def load_sequence_output_object(
    serialized_output_folder: Path,
    run_folder: Path,
) -> ComputedSequenceOutputInfo:
    config_path = serialized_output_folder / "output_config.yaml"
    vocab_json_path = _get_vocab_path(serialized_output_folder=serialized_output_folder)
    computed_max_length_path = serialized_output_folder / "computed_max_length.json"

    output_config = load_output_config_from_yaml(output_config_path=config_path)
    computed_max_length = json.loads(computed_max_length_path.read_text())

    output_name = output_config.output_info.output_name
    matching_input_path = get_input_serialization_path(
        run_folder=run_folder,
        input_name=output_name,
        input_type="sequence",
    )
    matching_input_object = load_sequence_input_object(
        serialized_input_folder=matching_input_path,
    )

    output_config_modified = deepcopy(output_config)
    output_type_info_modified = deepcopy(output_config.output_type_info)
    assert isinstance(output_type_info_modified, SequenceOutputTypeConfig)

    output_type_info_modified.vocab_file = str(vocab_json_path)
    output_type_info_modified.max_length = computed_max_length

    output_config_modified.output_type_info = output_type_info_modified

    loaded_object = set_up_sequence_output(
        output_config=output_config_modified,
        input_objects={output_name: matching_input_object},
    )

    return loaded_object


def _get_vocab_path(serialized_output_folder: Path) -> Path:
    vocab_json_path = serialized_output_folder / "vocab.json"
    bpe_tokenizer_path = serialized_output_folder / "bpe_tokenizer.json"
    if bpe_tokenizer_path.exists():
        return bpe_tokenizer_path
    return vocab_json_path
