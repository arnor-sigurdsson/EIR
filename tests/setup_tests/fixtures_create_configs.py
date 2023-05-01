import logging
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Sequence, Tuple, Union

import pytest
from aislib.misc_utils import ensure_path_exists

from eir.setup import config
from eir.setup.config import recursive_dict_replace
from eir.train_utils.utils import configure_global_eir_logging, get_run_folder
from tests.setup_tests.fixtures_create_data import TestDataConfig
from tests.setup_tests.setup_test_eir_configs import (
    get_test_base_global_init,
    get_test_inputs_inits,
    get_test_base_fusion_init,
    get_test_outputs_inits,
)


@dataclass
class TestConfigInits:
    global_configs: Sequence[dict]
    input_configs: Sequence[dict]
    fusion_configs: Sequence[dict]
    output_configs: Sequence[dict]


@pytest.fixture
def create_test_config_init_base(
    request, create_test_data: "TestDataConfig"
) -> Tuple[TestConfigInits, "TestDataConfig"]:
    injections = {}
    if hasattr(request, "param"):
        assert "injections" in request.param.keys()
        injections = request.param["injections"]

        injections_keys = set(injections.keys())
        expected_keys = set(TestConfigInits.__dataclass_fields__.keys())
        assert injections_keys.issubset(expected_keys)

    test_global_init = get_test_base_global_init()
    test_global_init = general_sequence_inject(
        sequence=test_global_init, inject_dict=injections.get("global_configs", {})
    )

    test_input_init = get_test_inputs_inits(
        test_path=create_test_data.scoped_tmp_path,
        input_config_dicts=injections.get("input_configs", {}),
        split_to_test=create_test_data.request_params.get("split_to_test", False),
        source=create_test_data.source,
        extra_kwargs=create_test_data.extras,
    )

    model_type = injections.get("fusion_configs", {}).get("model_type", "default")
    test_fusion_init = get_test_base_fusion_init(model_type=model_type)

    test_fusion_init = general_sequence_inject(
        sequence=test_fusion_init,
        inject_dict=injections.get("fusion_configs", {}),
    )

    test_output_init = get_test_outputs_inits(
        test_path=create_test_data.scoped_tmp_path,
        output_configs_dicts=injections.get("output_configs", {}),
        split_to_test=create_test_data.request_params.get("split_to_test", False),
    )

    test_config = TestConfigInits(
        global_configs=test_global_init,
        input_configs=test_input_init,
        fusion_configs=test_fusion_init,
        output_configs=test_output_init,
    )

    return test_config, create_test_data


def general_sequence_inject(
    sequence: Sequence[dict], inject_dict: dict
) -> Sequence[dict]:
    injected = []

    for dict_ in sequence:
        dict_injected = recursive_dict_replace(dict_=dict_, dict_to_inject=inject_dict)
        injected.append(dict_injected)

    return injected


@pytest.fixture()
def create_test_config(
    create_test_config_init_base: Tuple[TestConfigInits, "TestDataConfig"],
    keep_outputs: bool,
) -> config.Configs:
    test_init, test_data_config = copy(create_test_config_init_base)

    test_configs, output_folder = build_test_output_folder(
        test_configs=test_init, test_data_config=test_data_config
    )
    configure_global_eir_logging(output_folder=output_folder, log_level="DEBUG")

    test_global_config = config.get_global_config(
        global_configs=test_init.global_configs
    )

    test_input_configs = config.get_input_configs(input_configs=test_init.input_configs)
    test_fusion_configs = config.load_fusion_configs(
        fusion_configs=test_init.fusion_configs
    )

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    test_output_configs = config.load_output_configs(
        output_configs=test_init.output_configs,
        dynamic_output_setup=tabular_output_setup,
    )

    test_configs = config.Configs(
        global_config=test_global_config,
        input_configs=test_input_configs,
        fusion_config=test_fusion_configs,
        output_configs=test_output_configs,
    )

    run_folder = get_run_folder(output_folder=output_folder)

    if run_folder.exists():
        cleanup(run_path=run_folder)

    ensure_path_exists(path=run_folder, is_folder=True)

    yield test_configs

    teardown_logger()

    if not keep_outputs:
        cleanup(run_path=run_folder)


def build_test_output_folder(
    test_configs: TestConfigInits, test_data_config: TestDataConfig
) -> Tuple[TestConfigInits, str]:
    """
    This is done after in case tests modify output_folder.
    """

    test_configs_copy = copy(test_configs)

    output_folder_base = test_configs_copy.global_configs[0]["output_folder"]
    input_model_types = "_".join(
        i["model_config"]["model_type"] for i in test_configs_copy.input_configs
    )
    output_model_type = test_configs_copy.output_configs[0]["model_config"].get(
        "model_type", "default"
    )
    task_type = test_data_config.request_params["task_type"]

    output_folder = (
        output_folder_base
        + "_"
        + input_model_types
        + "_"
        + output_model_type
        + "_"
        + task_type
    )

    if not output_folder.startswith("runs/"):
        output_folder = "runs/" + output_folder

    for gc in test_configs_copy.global_configs:
        gc["output_folder"] = output_folder

    return test_configs_copy, output_folder


def teardown_logger():
    root_logger = logging.getLogger("")
    for handler in root_logger.handlers:
        match handler:
            case logging.FileHandler():
                if "logging_history.log" in handler.baseFilename:
                    handler.close()
                    root_logger.removeHandler(handler)


def cleanup(run_path: Union[Path, str]) -> None:
    rmtree(path=run_path)
