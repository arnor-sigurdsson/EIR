import json
from copy import deepcopy
from pathlib import Path

from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.experiment_io.io_utils import load_dataclass
from eir.setup import schemas
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    ImageNormalizationStats,
    set_up_computed_image_input_object,
)


def load_image_input_object(
    serialized_input_folder: Path,
) -> ComputedImageInputInfo:
    config_path = serialized_input_folder / "input_config.yaml"
    normalization_stats_path = serialized_input_folder / "normalization_stats.json"
    num_channels_path = serialized_input_folder / "num_channels.json"

    input_config = load_input_config_from_yaml(input_config_path=config_path)
    input_type_info_modified = deepcopy(input_config.input_type_info)
    assert isinstance(input_type_info_modified, schemas.ImageInputDataConfig)

    normalization_stats = load_dataclass(
        cls=ImageNormalizationStats, file_path=normalization_stats_path
    )

    image_mode = input_type_info_modified.mode
    if not image_mode:
        num_channels = json.loads(num_channels_path.read_text())["num_channels"]
        input_type_info_modified.num_channels = num_channels
    else:
        input_type_info_modified.num_channels = None

    input_config_modified = deepcopy(input_config)
    input_config_modified.input_type_info = input_type_info_modified

    loaded_object = set_up_computed_image_input_object(
        input_config=input_config_modified,
        normalization_stats=normalization_stats,
    )

    return loaded_object
