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
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class ComputedArrayOutputInfo:
    output_config: OutputConfig
    data_dimensions: DataDimensions
    dtype: np.dtype
    normalization_stats: Optional[ArrayNormalizationStats] = None


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

    array_output_object = ComputedArrayOutputInfo(
        output_config=output_config,
        data_dimensions=data_dimensions,
        normalization_stats=normalization_stats,
        dtype=dtype,
    )

    return array_output_object
