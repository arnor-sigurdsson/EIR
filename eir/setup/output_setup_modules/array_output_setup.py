from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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
    normalization_stats: Optional[ArrayNormalizationStats] = None
    diffusion_config: Optional[DiffusionConfig] = None


def set_up_array_output(
    output_config: OutputConfig, *args, **kwargs
) -> ComputedArrayOutputInfo:
    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(output_config.output_info.output_source),
        deeplake_inner_key=output_config.output_info.output_inner_key,
    )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, ArrayOutputTypeConfig)

    normalization_stats: Optional[ArrayNormalizationStats] = None
    if output_type_info.normalization is not None:
        normalization_stats = get_array_normalization_values(
            source=output_config.output_info.output_source,
            inner_key=output_config.output_info.output_inner_key,
            normalization=output_type_info.normalization,
            data_dimensions=data_dimensions,
            max_samples=output_type_info.adaptive_normalization_max_samples,
        )

    dtype = get_dtype_from_data_source(
        data_source=Path(output_config.output_info.output_source),
        deeplake_inner_key=output_config.output_info.output_inner_key,
    )

    diffusion_config = None
    if output_type_info.loss == "diffusion":
        time_steps = output_type_info.diffusion_time_steps
        if time_steps is None:
            raise ValueError(
                "Diffusion loss requires specifying the number of time steps."
                "Please set `diffusion_time_steps` in the output config."
            )
        diffusion_config = initialize_diffusion_config(time_steps=time_steps)

    array_output_object = ComputedArrayOutputInfo(
        output_config=output_config,
        data_dimensions=data_dimensions,
        normalization_stats=normalization_stats,
        dtype=dtype,
        diffusion_config=diffusion_config,
    )

    return array_output_object
