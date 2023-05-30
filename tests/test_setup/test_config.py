import argparse
import os
import tempfile
from argparse import Namespace
from copy import copy
from pathlib import Path
from typing import Dict, List, Mapping, Any

import pytest
import yaml
from hypothesis import given, strategies

from eir.setup import config
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_iterator_with_injections,
    convert_cl_str_to_dict,
)
from tests.setup_tests.fixtures_create_configs import TestConfigInits


@pytest.fixture()
def create_cl_args_config_files(
    create_test_config_init_base: TestConfigInits, tmp_path
) -> Dict[str, List[str]]:
    test_init_base = create_test_config_init_base[0]

    config_file_paths = {}
    for config_name in test_init_base.__dataclass_fields__.keys():
        cur_paths = []
        for idx, cur_config in enumerate(getattr(test_init_base, config_name)):
            cur_outpath = tmp_path / f"{config_name.split('_')[0]}_{idx}.yaml"
            with open(cur_outpath, "w") as out_yaml:
                yaml.dump(data=cur_config, stream=out_yaml)
            cur_paths.append(str(cur_outpath))
        config_file_paths[config_name] = cur_paths

    return config_file_paths


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_generate_aggregated_config_basic(
    create_cl_args_config_files: Dict[str, List[str]],
):
    test_cl_args = Namespace(**create_cl_args_config_files)

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    aggregated_config = config.generate_aggregated_config(
        cl_args=test_cl_args, dynamic_output_setup=tabular_output_setup
    )
    assert aggregated_config.global_config.output_folder == "runs/test_run"

    assert len(aggregated_config.input_configs) == 1
    assert aggregated_config.input_configs[0].input_info.input_name == "test_genotype"
    assert aggregated_config.input_configs[0].input_info.input_type == "omics"

    assert len(aggregated_config.output_configs) == 1
    assert aggregated_config.output_configs[0].output_type_info.target_cat_columns == [
        "Origin"
    ]


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_generate_aggregated_config_fail(
    create_cl_args_config_files: Dict[str, List[str]],
):
    input_file = create_cl_args_config_files["input_configs"][0]
    with open(input_file, "r") as infile:
        original_config = yaml.load(stream=infile, Loader=yaml.FullLoader)

    original_config["input_info"]["input_name"] = "test_output"

    with open(input_file, "w") as outfile:
        yaml.dump(data=original_config, stream=outfile)

    test_cl_args = Namespace(**create_cl_args_config_files)

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    with pytest.raises(ValueError):
        config.generate_aggregated_config(
            cl_args=test_cl_args, dynamic_output_setup=tabular_output_setup
        )


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_generate_aggregated_config_with_overload(create_cl_args_config_files):
    test_cl_args = Namespace(**create_cl_args_config_files)

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    aggregated_config = config.generate_aggregated_config(
        cl_args=test_cl_args,
        dynamic_output_setup=tabular_output_setup,
        extra_cl_args_overload=["--input_0.input_info.input_source=test_value"],
    )
    assert aggregated_config.global_config.output_folder == "runs/test_run"
    assert len(aggregated_config.input_configs) == 1
    assert aggregated_config.input_configs[0].input_info.input_source == "test_value"


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "binary",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_get_yaml_iterator_with_injections(create_cl_args_config_files):
    test_cl_args = Namespace(**create_cl_args_config_files)

    input_yaml_files = test_cl_args.input_configs
    assert len(input_yaml_files) == 1
    assert Path(input_yaml_files[0]).stem == "input_0"

    extra_cl_args_overload = ["--input_0.input_info.input_source=test_value"]

    yaml_iter_with_injections = get_yaml_iterator_with_injections(
        yaml_config_files=input_yaml_files, extra_cl_args=extra_cl_args_overload
    )
    overloaded_config = next(yaml_iter_with_injections)
    assert overloaded_config["input_info"]["input_source"] == "test_value"


def test_convert_cl_str_to_dict():
    test_str = "gln_input.input_info.input_source=test_value"
    test_dict = convert_cl_str_to_dict(str_=test_str)
    assert test_dict == {"gln_input": {"input_info": {"input_source": "test_value"}}}


def test_get_output_folder_and_log_level_from_cl_args():
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir.name)
    test_config = {
        "output_folder": "expected_output_folder",
        "log_level": "expected_log_level",
    }
    with open(temp_file.name, "w") as f:
        yaml.safe_dump(test_config, f)

    main_cl_args = argparse.Namespace(global_configs=[temp_file.name])
    extra_cl_args: List[str] = []

    output_folder, log_level = config.get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args,
        extra_cl_args=extra_cl_args,
    )

    assert output_folder == test_config["output_folder"]
    assert log_level == test_config["log_level"]

    extra_cl_args = ["output_folder=new_output_folder"]
    output_folder, log_level = config.get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args,
        extra_cl_args=extra_cl_args,
    )

    assert output_folder == "new_output_folder"
    assert log_level == test_config["log_level"]

    test_config = {"log_level": "expected_log_level", "output_folder": None}
    with open(temp_file.name, "w") as f:
        yaml.safe_dump(test_config, f)

    with pytest.raises(ValueError, match="Output folder not found in global configs."):
        config.get_output_folder_and_log_level_from_cl_args(
            main_cl_args=main_cl_args,
            extra_cl_args=[],
        )

    os.unlink(temp_file.name)
    temp_dir.cleanup()


@given(
    strategies.dictionaries(strategies.text(), strategies.integers()),
    strategies.integers(),
)
def test_recursive_search(dict_: Mapping, target: Any):
    paths_and_values = list(
        config._recursive_search(
            dict_=dict_,
            target=target,
        )
    )

    for path, value in paths_and_values:
        dict_copy = copy(dict_)
        for key in path:
            dict_copy = dict_copy[key]
        assert dict_copy == target
