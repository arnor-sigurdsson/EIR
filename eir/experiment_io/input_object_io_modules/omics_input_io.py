from copy import deepcopy
from pathlib import Path

from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.experiment_io.io_utils import load_dataclass
from eir.setup.input_setup_modules.setup_omics import (
    ComputedOmicsInputInfo,
    DataDimensions,
    set_up_omics_input,
)
from eir.setup.schemas import OmicsInputDataConfig


def load_omics_input_object(serialized_input_folder: Path) -> ComputedOmicsInputInfo:
    config_path = serialized_input_folder / "input_config.yaml"
    data_dimensions_path = serialized_input_folder / "data_dimensions.json"
    snps_path = serialized_input_folder / "snps.bim"
    subset_snps_file_path = serialized_input_folder / "subset_snps_file.txt"

    input_config = load_input_config_from_yaml(input_config_path=config_path)
    input_config_copy = deepcopy(input_config)

    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, OmicsInputDataConfig)

    if snps_path.exists():
        input_type_info.snp_file = str(snps_path)

    if subset_snps_file_path.exists():
        input_type_info.subset_snps_file = str(subset_snps_file_path)

    input_config_copy.input_type_info = input_type_info

    data_dimensions = load_dataclass(
        cls=DataDimensions,
        file_path=data_dimensions_path,
    )

    loaded_object = set_up_omics_input(
        input_config=input_config_copy,
        data_dimensions=data_dimensions,
    )

    return loaded_object
