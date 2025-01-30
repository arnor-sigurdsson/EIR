from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from eir.setup.config_setup_modules.input_config_initialization import init_input_config

if TYPE_CHECKING:
    from eir.setup.schemas import InputConfig


def load_input_config_from_yaml(input_config_path: Path) -> "InputConfig":
    with open(input_config_path) as infile:
        input_config_dict = yaml.safe_load(stream=infile)

    input_config = init_input_config(yaml_config_as_dict=input_config_dict)

    return input_config
