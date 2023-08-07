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
)
from eir.setup.setup_utils import (
    ChannelBasedRunningStatistics,
    ElementBasedRunningStatistics,
    al_collector_classes,
    collect_stats,
)


@dataclass
class ComputedArrayInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"
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
    if input_type_info.normalization == "element":
        normalization_stats = get_array_normalization_values(
            input_config=input_config,
            data_dimensions=data_dimensions,
        )

    array_input_info = ComputedArrayInputInfo(
        input_config=input_config,
        data_dimensions=data_dimensions,
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
    input_config: schemas.InputConfig,
    data_dimensions: DataDimensions,
) -> ArrayNormalizationStats:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.ArrayInputDataConfig)

    input_source = input_config.input_info.input_source
    deeplake_inner_key = input_config.input_info.input_inner_key

    if is_deeplake_dataset(data_source=input_source):
        deeplake_ds = load_deeplake_dataset(data_source=input_source)
        assert deeplake_inner_key is not None
        ds_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        tensor_iterator = (torch.from_numpy(i.numpy()) for i in ds_iter)
    else:
        file_iterator = Path(input_source).rglob("*")
        np_iterator = (np.load(str(i)) for i in file_iterator)
        tensor_iterator = (torch.from_numpy(i).float() for i in np_iterator)

    collector_class: al_collector_classes = ElementBasedRunningStatistics
    if input_type_info.normalization == "channel":
        collector_class = ChannelBasedRunningStatistics

    gathered_stats = collect_stats(
        tensor_iterable=tensor_iterator,
        collector_class=collector_class,
        shape=data_dimensions.full_shape(),
    )

    normalization_stats = ArrayNormalizationStats(
        shape=data_dimensions.full_shape(),
        means=gathered_stats.mean,
        stds=gathered_stats.std,
        type=input_type_info.normalization,
    )

    return normalization_stats
