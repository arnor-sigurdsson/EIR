from dataclasses import dataclass
from pathlib import Path

from eir.setup import schemas
from eir.setup.input_setup_modules.common import (
    DataDimensions,
    get_data_dimension_from_data_source,
)


@dataclass
class ComputedArrayInputInfo:
    input_config: schemas.InputConfig
    data_dimensions: "DataDimensions"


def set_up_array_input(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ComputedArrayInputInfo:
    data_dimensions = get_data_dimension_from_data_source(
        data_source=Path(input_config.input_info.input_source),
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    array_input_info = ComputedArrayInputInfo(
        input_config=input_config,
        data_dimensions=data_dimensions,
    )

    return array_input_info
