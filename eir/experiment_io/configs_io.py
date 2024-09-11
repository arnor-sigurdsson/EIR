from pathlib import Path

import yaml

from eir.setup.config import (
    Configs,
    get_global_config,
    get_input_configs,
    load_fusion_configs,
    load_output_configs,
    validate_train_configs,
)


def load_configs(configs_root_folder: Path) -> Configs:
    """
    Note we do not add the extra sequence output configs here as they
    are already added to the configs when they are saved.
    """
    global_path = configs_root_folder / "global_config.yaml"
    input_path = configs_root_folder / "input_configs.yaml"
    fusion_path = configs_root_folder / "fusion_config.yaml"
    output_path = configs_root_folder / "output_configs.yaml"

    with open(global_path, "r") as infile:
        global_config_list = [yaml.safe_load(stream=infile)]
    global_config = get_global_config(global_configs=global_config_list)

    with open(input_path, "r") as infile:
        input_config_list = yaml.safe_load(stream=infile)
    input_configs = get_input_configs(input_configs=input_config_list)

    with open(fusion_path, "r") as infile:
        fusion_config_list = [yaml.safe_load(stream=infile)]
    fusion_config = load_fusion_configs(fusion_configs=fusion_config_list)

    with open(output_path, "r") as infile:
        output_config_list = yaml.safe_load(stream=infile)
    output_configs = load_output_configs(output_configs=output_config_list)

    aggregate_config = Configs(
        global_config=global_config,
        input_configs=input_configs,
        fusion_config=fusion_config,
        output_configs=output_configs,
    )

    validate_train_configs(configs=aggregate_config)

    return aggregate_config
