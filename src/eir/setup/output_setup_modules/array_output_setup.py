from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eir.data_load.label_setup import get_file_path_iterator
from eir.setup.input_setup_modules.common import DataDimensions
from eir.setup.input_setup_modules.setup_array import (
    ArrayNormalizationStats,
    get_array_normalization_values,
    get_data_dimension_from_data_source,
    get_dtype_from_data_source,
)
from eir.setup.schemas import ArrayOutputTypeConfig, OutputConfig
from eir.train_utils.step_modules.diffusion import (
    DiffusionConfig,
    initialize_diffusion_config,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class ComputedArrayOutputInfo:
    output_config: OutputConfig
    data_dimensions: DataDimensions
    dtype: np.dtype
    normalization_stats: ArrayNormalizationStats | None = None
    diffusion_config: DiffusionConfig | None = None
    num_classes: int | None = None


def set_up_array_output(
    output_config: OutputConfig,
    normalization_stats: ArrayNormalizationStats | None = None,
    data_dimensions: DataDimensions | None = None,
    diffusion_config: DiffusionConfig | None = None,
    dtype: np.dtype | None = None,
    num_classes: int | None = None,
    *args,
    **kwargs,
) -> ComputedArrayOutputInfo:
    if data_dimensions is None:
        data_dimensions = get_data_dimension_from_data_source(
            data_source=Path(output_config.output_info.output_source),
        )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, ArrayOutputTypeConfig)

    is_categorical = output_type_info.loss == "categorical"

    if is_categorical:
        normalization_stats = None
        if num_classes is None:
            num_classes = get_array_num_classes(
                source=output_config.output_info.output_source,
                max_samples=output_type_info.adaptive_normalization_max_samples,
            )
            logger.info(
                "Auto-discovered %d classes for categorical array output '%s'.",
                num_classes,
                output_config.output_info.output_name,
            )
    else:
        if normalization_stats is None and output_type_info.normalization is not None:
            normalization_stats = get_array_normalization_values(
                source=output_config.output_info.output_source,
                normalization=output_type_info.normalization,
                data_dimensions=data_dimensions,
                max_samples=output_type_info.adaptive_normalization_max_samples,
            )

    if dtype is None:
        dtype = get_dtype_from_data_source(
            data_source=Path(output_config.output_info.output_source),
        )

    if diffusion_config is None and output_type_info.loss == "diffusion":
        time_steps = output_type_info.diffusion_time_steps
        if time_steps is None:
            raise ValueError(
                "Diffusion loss requires specifying the number of time steps."
                "Please set `diffusion_time_steps` in the output config."
            )
        beta_schedule = output_type_info.diffusion_beta_schedule
        diffusion_config = initialize_diffusion_config(
            time_steps=time_steps,
            beta_schedule=beta_schedule,
        )

    array_output_object = ComputedArrayOutputInfo(
        output_config=output_config,
        data_dimensions=data_dimensions,
        normalization_stats=normalization_stats,
        dtype=dtype,
        diffusion_config=diffusion_config,
        num_classes=num_classes,
    )

    return array_output_object


def get_array_num_classes(
    source: str,
    max_samples: int | None,
) -> int:
    file_iterator = get_file_path_iterator(data_source=Path(source))

    global_max = -1
    for count, path in enumerate(file_iterator):
        if max_samples is not None and count >= max_samples:
            break
        arr = np.load(str(path))
        cur_max = int(arr.max())
        if cur_max > global_max:
            global_max = cur_max

    if global_max < 0:
        raise ValueError(
            f"Could not determine num_classes from data source '{source}'. "
            f"No valid data files found or all arrays are empty."
        )

    return global_max + 1
