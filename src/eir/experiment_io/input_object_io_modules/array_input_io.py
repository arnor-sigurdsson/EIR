import json
from pathlib import Path

import numpy as np

from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.experiment_io.io_utils import load_dataclass
from eir.setup.input_setup_modules.setup_array import (
    ArrayNormalizationStats,
    ComputedArrayInputInfo,
    DataDimensions,
    set_up_array_input_object,
)


def load_array_input_object(serialized_input_folder: Path) -> ComputedArrayInputInfo:
    config_path = serialized_input_folder / "input_config.yaml"
    normalization_stats_path = serialized_input_folder / "normalization_stats.json"
    dtype_path = serialized_input_folder / "dtype.json"
    data_dimensions_path = serialized_input_folder / "data_dimensions.json"

    input_config = load_input_config_from_yaml(input_config_path=config_path)

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

    loaded_object = set_up_array_input_object(
        input_config=input_config,
        normalization_stats=normalization_stats,
        data_dimensions=data_dimensions,
        dtype=dtype,
    )

    return loaded_object
