from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from eir.setup.config_setup_modules.output_config_initialization import (
    init_output_config,
)

if TYPE_CHECKING:
    from eir.setup.schemas import OutputConfig


def load_output_config_from_yaml(output_config_path: Path) -> "OutputConfig":
    with open(output_config_path, "r") as infile:
        output_config_dict = yaml.safe_load(stream=infile)

    output_config = init_output_config(yaml_config_as_dict=output_config_dict)

    return output_config
