from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

from eir.data_load.data_source_modules.deeplake_ops import (
    get_deeplake_input_source_iterable,
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.setup import schemas
from eir.setup.input_setup_modules.common import (
    DataDimensions,
    get_data_dimension_from_data_source,
    get_dtype_from_data_source,
)
from eir.setup.setup_utils import (
    ChannelBasedRunningStatistics,
    ElementBasedRunningStatistics,
    al_collector_classes,
    collect_stats,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class ComputedArrayInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"
    dtype: np.dtype
    normalization_stats: Optional["ArrayNormalizationStats"] = None


def set_up_array_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ComputedArrayInputInfo:
    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source),
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.ArrayInputDataConfig)

    normalization_stats: Optional[ArrayNormalizationStats] = None
    if input_type_info.normalization is not None:
        normalization_stats = get_array_normalization_values(
            source=input_config.input_info.input_source,
            inner_key=input_config.input_info.input_inner_key,
            normalization=input_type_info.normalization,
            data_dimensions=data_dimensions,
            max_samples=input_type_info.adaptive_normalization_max_samples,
        )

    dtype = get_dtype_from_data_source(
        data_source=Path(input_config.input_info.input_source),
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    array_input_info = ComputedArrayInputInfo(
        input_config=input_config,
        data_dimensions=data_dimensions,
        dtype=dtype,
        normalization_stats=normalization_stats,
    )

    return array_input_info


@dataclass
class ArrayNormalizationStats:
    shape: tuple
    means: torch.Tensor
    stds: torch.Tensor
    type: Optional[Literal["element", "channel"]]


def get_array_normalization_values(
    source: str,
    inner_key: Optional[str],
    normalization: Optional[Literal["element", "channel"]],
    data_dimensions: DataDimensions,
    max_samples: Optional[int],
) -> ArrayNormalizationStats:
    input_source = source
    deeplake_inner_key = inner_key

    if is_deeplake_dataset(data_source=input_source):
        deeplake_ds = load_deeplake_dataset(data_source=input_source)
        assert deeplake_inner_key is not None
        ds_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        tensor_iterator = (torch.from_numpy(i.numpy()).float() for i in ds_iter)
    else:
        file_iterator = Path(input_source).rglob("*")
        np_iterator = (np.load(str(i)) for i in file_iterator)
        tensor_iterator = (torch.from_numpy(i).float() for i in np_iterator)

    tensor_iterator = (i.reshape(data_dimensions.full_shape()) for i in tensor_iterator)

    collector_class: al_collector_classes
    if normalization == "channel":
        collector_class = ChannelBasedRunningStatistics
    elif normalization == "element":
        collector_class = ElementBasedRunningStatistics
    else:
        raise ValueError(
            f"Invalid normalization type: {normalization}. "
            f"Must be one of ['element', 'channel']"
        )

    gathered_stats = collect_stats(
        tensor_iterable=tensor_iterator,
        collector_class=collector_class,
        shape=data_dimensions.full_shape(),
        max_samples=max_samples,
        name=source,
    )

    means = _add_extra_dims_if_needed(
        normalization_tensor=gathered_stats.mean,
        data_shape=data_dimensions.full_shape(),
        normalization_type=normalization,
    )

    stds = _add_extra_dims_if_needed(
        normalization_tensor=gathered_stats.std,
        data_shape=data_dimensions.full_shape(),
        normalization_type=normalization,
    )

    normalization_stats = ArrayNormalizationStats(
        shape=data_dimensions.full_shape(),
        means=means,
        stds=stds,
        type=normalization,
    )

    _check_normalization_stats(source=source, normalization_stats=normalization_stats)

    return normalization_stats


def _add_extra_dims_if_needed(
    normalization_tensor: torch.Tensor,
    data_shape: tuple,
    normalization_type: Optional[Literal["element", "channel"]],
) -> torch.Tensor:
    if normalization_type == "element":
        assert normalization_tensor.shape == data_shape
        return normalization_tensor

    elif normalization_type == "channel":
        assert normalization_tensor.dim() == 1
        assert normalization_tensor.shape[0] == data_shape[0]

        num_extra_dims = len(data_shape) - 1
        new_shape = (data_shape[0],) + (1,) * num_extra_dims
        reshaped_tensor = normalization_tensor.view(*new_shape)

        return reshaped_tensor

    else:
        raise ValueError(
            f"Invalid normalization type: {normalization_type}. "
            f"Must be one of ['element', 'channel']"
        )


def _check_normalization_stats(
    source: str, normalization_stats: ArrayNormalizationStats, epsilon: float = 1e-10
) -> None:
    if torch.any(normalization_stats.stds < epsilon):
        num_zero_stds = torch.sum(normalization_stats.stds < epsilon).item()
        logger.warning(
            f"In source {source}, "
            f"{num_zero_stds} elements have zero or near-zero standard deviation. "
            "This may lead to large values when normalizing. "
            "Consider handling these cases specifically.",
        )
