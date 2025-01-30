import json
from pathlib import Path

import numpy as np

from eir.experiment_io.io_utils import load_dataclass
from eir.experiment_io.output_object_io_modules.output_io_utils import (
    load_output_config_from_yaml,
)
from eir.setup.output_setup_modules.array_output_setup import (
    ArrayNormalizationStats,
    ComputedArrayOutputInfo,
    DataDimensions,
    DiffusionConfig,
    set_up_array_output,
)


def load_array_output_object(serialized_output_folder: Path) -> ComputedArrayOutputInfo:
    config_path = serialized_output_folder / "output_config.yaml"
    normalization_stats_path = serialized_output_folder / "normalization_stats.json"
    dtype_path = serialized_output_folder / "dtype.json"
    data_dimensions_path = serialized_output_folder / "data_dimensions.json"
    diffusion_path = serialized_output_folder / "diffusion.json"

    output_config = load_output_config_from_yaml(output_config_path=config_path)

    normalization_stats: ArrayNormalizationStats | None = None
    if normalization_stats_path.exists():
        normalization_stats = load_dataclass(
            cls=ArrayNormalizationStats,
            file_path=normalization_stats_path,
        )

    dtype_str = json.loads(dtype_path.read_text())
    dtype = np.dtype(dtype_str)

    data_dimensions = load_dataclass(
        cls=DataDimensions,
        file_path=data_dimensions_path,
    )

    diffusion_config = None
    if diffusion_path.exists():
        diffusion_config = load_dataclass(
            cls=DiffusionConfig,
            file_path=diffusion_path,
        )

    loaded_object = set_up_array_output(
        output_config=output_config,
        normalization_stats=normalization_stats,
        data_dimensions=data_dimensions,
        dtype=dtype,
        diffusion_config=diffusion_config,
    )

    return loaded_object
