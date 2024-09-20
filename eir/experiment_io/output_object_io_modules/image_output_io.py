import json
from copy import deepcopy
from pathlib import Path

from eir.experiment_io.io_utils import load_dataclass
from eir.experiment_io.output_object_io_modules.output_io_utils import (
    load_output_config_from_yaml,
)
from eir.setup import schemas
from eir.setup.output_setup_modules.image_output_setup import (
    ComputedImageOutputInfo,
    DiffusionConfig,
    ImageNormalizationStats,
    set_up_image_output,
)


def load_image_output_object(
    serialized_output_folder: Path,
) -> ComputedImageOutputInfo:
    config_path = serialized_output_folder / "output_config.yaml"
    normalization_stats_path = serialized_output_folder / "normalization_stats.json"
    num_channels_path = serialized_output_folder / "num_channels.json"
    diffusion_path = serialized_output_folder / "diffusion.json"

    output_config = load_output_config_from_yaml(output_config_path=config_path)
    output_type_info_modified = deepcopy(output_config.output_type_info)
    assert isinstance(output_type_info_modified, schemas.ImageOutputTypeConfig)

    normalization_stats = load_dataclass(
        cls=ImageNormalizationStats, file_path=normalization_stats_path
    )

    image_mode = output_type_info_modified.mode
    if not image_mode:
        num_channels = json.loads(num_channels_path.read_text())["num_channels"]
        output_type_info_modified.num_channels = num_channels
    else:
        output_type_info_modified.num_channels = None

    output_config_modified = deepcopy(output_config)
    output_config_modified.output_type_info = output_type_info_modified

    diffusion_config = None
    if diffusion_path.exists():
        diffusion_config = load_dataclass(
            cls=DiffusionConfig,
            file_path=diffusion_path,
        )

    loaded_object = set_up_image_output(
        output_config=output_config_modified,
        normalization_stats=normalization_stats,
        diffusion_config=diffusion_config,
    )

    return loaded_object
