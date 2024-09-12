from copy import deepcopy
from pathlib import Path
from typing import Sequence

import yaml

from eir.models.input.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
)
from eir.setup.config import (
    Configs,
    get_global_config,
    get_input_configs,
    load_fusion_configs,
    load_output_configs,
    validate_train_configs,
)
from eir.setup.schemas import InputConfig


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
    input_configs_patched = _patch_sequence_input_configs_linked_from_output(
        input_configs=input_configs
    )

    with open(fusion_path, "r") as infile:
        fusion_config_list = [yaml.safe_load(stream=infile)]
    fusion_config = load_fusion_configs(fusion_configs=fusion_config_list)

    with open(output_path, "r") as infile:
        output_config_list = yaml.safe_load(stream=infile)
    output_configs = load_output_configs(output_configs=output_config_list)

    aggregate_config = Configs(
        global_config=global_config,
        input_configs=input_configs_patched,
        fusion_config=fusion_config,
        output_configs=output_configs,
    )

    validate_train_configs(configs=aggregate_config)

    return aggregate_config


def _patch_sequence_input_configs_linked_from_output(
    input_configs: Sequence[InputConfig],
) -> Sequence[InputConfig]:
    """
    We need this as during training when configs are set up, we call
    get_configs_object_with_seq_output_configs(configs=aggregate_config) to build
    relevant sequence input configs from the sequence output configs.

    A part of the logic there initializes the model_init_config of the input config
    to be a BasicTransformerFeatureExtractorModelConfig object.

    However, we do not reuse the get_configs_object_with_seq_output_configs function
    here as the configs have already been built (during training) and saved,
    hence we are only loading them here. Calling that function results in a
    collision / error. However, we still need to patch the model_init_config
    here to ensure the proper BasicTransformerFeatureExtractorModelConfig object
    is initialized.
    """
    parsed_configs = []

    for input_config in input_configs:
        input_type = input_config.input_info.input_type
        if input_type != "sequence":
            parsed_configs.append(input_config)
            continue

        model_type = input_config.model_config.model_type
        if not model_type.startswith("eir-input-sequence-from-linked-output-"):
            parsed_configs.append(input_config)
            continue

        input_config_copy = deepcopy(input_config)

        init_kwargs_as_dict = input_config.model_config.model_init_config
        assert isinstance(init_kwargs_as_dict, dict)

        model_init_config = BasicTransformerFeatureExtractorModelConfig(
            **init_kwargs_as_dict
        )

        input_config_copy.model_config.model_init_config = model_init_config

        parsed_configs.append(input_config_copy)

    return parsed_configs
