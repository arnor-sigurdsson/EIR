import argparse
import os
import tempfile
from argparse import Namespace
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings, strategies

from eir.setup import config
from eir.setup.config_setup_modules.config_setup_utils import (
    convert_cl_str_to_dict,
    get_yaml_iterator_with_injections,
    recursive_dict_inject,
    validate_keys_against_dataclass,
)
from tests.setup_tests.fixtures_create_configs import TestConfigInits


@pytest.fixture()
def create_cl_args_config_files(
    create_test_config_init_base: TestConfigInits, tmp_path
) -> dict[str, list[str]]:
    test_init_base = create_test_config_init_base[0]

    config_file_paths = {}
    for config_name in test_init_base.__dataclass_fields__:
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
                        "output_info": {"output_name": "test_output_tabular"},
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
    create_cl_args_config_files: dict[str, list[str]],
):
    test_cl_args = Namespace(**create_cl_args_config_files)

    aggregated_config = config.generate_aggregated_config(
        cl_args=test_cl_args,
    )
    assert aggregated_config.gc.be.output_folder == "runs/test_run"

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
                        "output_info": {"output_name": "test_output_tabular"},
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
    create_cl_args_config_files: dict[str, list[str]],
):
    input_file = create_cl_args_config_files["input_configs"][0]
    with open(input_file) as infile:
        original_config = yaml.load(stream=infile, Loader=yaml.FullLoader)

    original_config["input_info"]["input_name"] = "test_output_tabular"

    with open(input_file, "w") as outfile:
        yaml.dump(data=original_config, stream=outfile)

    test_cl_args = Namespace(**create_cl_args_config_files)

    with pytest.raises(ValueError):
        config.generate_aggregated_config(
            cl_args=test_cl_args,
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
                        "output_info": {"output_name": "test_output_tabular"},
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

    aggregated_config = config.generate_aggregated_config(
        cl_args=test_cl_args,
        extra_cl_args_overload=["--input_0.input_info.input_source=test_value"],
    )
    assert aggregated_config.gc.be.output_folder == "runs/test_run"
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
                        "output_info": {"output_name": "test_output_tabular"},
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
        "basic_experiment": {"output_folder": "expected_output_folder"},
        "visualization_logging": {"log_level": "expected_log_level"},
    }
    with open(temp_file.name, "w") as f:
        yaml.safe_dump(test_config, f)

    main_cl_args = argparse.Namespace(global_configs=[temp_file.name])
    extra_cl_args: list[str] = []

    output_folder, log_level = config.get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args,
        extra_cl_args=extra_cl_args,
    )

    assert output_folder == test_config["basic_experiment"]["output_folder"]
    assert log_level == test_config["visualization_logging"]["log_level"]

    extra_cl_args = ["basic_experiment.output_folder=new_output_folder"]
    output_folder, log_level = config.get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args,
        extra_cl_args=extra_cl_args,
    )

    assert output_folder == "new_output_folder"
    assert log_level == test_config["visualization_logging"]["log_level"]

    test_config = {
        "basic_experiment": {"output_folder": None},
        "visualization_logging": {"log_level": "expected_log_level"},
    }
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
@settings(deadline=500)
def test_recursive_search(dict_: Mapping, target: Any):
    paths_and_values = list(
        config._recursive_search(
            dict_=dict_,
            target=target,
        )
    )

    for path, _value in paths_and_values:
        dict_copy = copy(dict_)
        for key in path:
            dict_copy = dict_copy[key]
        assert dict_copy == target


@dataclass
class MockDataclass:
    field1: str
    field2: int


def test_validate_keys_against_dataclass():
    valid_input_dict = {"field1": "value1", "field2": 42}
    validate_keys_against_dataclass(
        input_dict=valid_input_dict, dataclass_type=MockDataclass
    )

    invalid_input_dict = {"field1": "value1", "unexpected_field": 42}
    with pytest.raises(KeyError, match="Unexpected keys found"):
        validate_keys_against_dataclass(
            input_dict=invalid_input_dict, dataclass_type=MockDataclass
        )

    non_dataclass_type = int
    with pytest.raises(TypeError, match="Provided type int is not a dataclass"):
        validate_keys_against_dataclass(
            input_dict=valid_input_dict, dataclass_type=non_dataclass_type
        )


def test_basic_replacement():
    d = {"key1": "value1"}
    inject = {"key1": "updated", "key2": {"subkey": "subvalue"}}
    expected = {"key1": "updated", "key2": {"subkey": "subvalue"}}
    result = recursive_dict_inject(d, inject)
    assert result == expected


def test_nested_replacement():
    d = {"nested": {"key1": "value1"}}
    inject = {"nested": {"key1": "updated"}}
    expected = {"nested": {"key1": "updated"}}
    result = recursive_dict_inject(d, inject)
    assert result == expected


def test_adding_dict_to_none():
    d = {"key1": None}
    inject = {"key1": {"newkey": "newvalue"}}
    expected = {"key1": {"newkey": "newvalue"}}
    result = recursive_dict_inject(d, inject)
    assert result == expected


def test_adding_none_to_dict():
    d = {"key1": {"newkey": "newvalue"}}
    inject = {"key1": None}
    expected = {"key1": None}
    result = recursive_dict_inject(d, inject)
    assert result == expected


def test_mixed_types():
    d = {"key1": {"subkey": "value"}, "key2": "simple"}
    inject = {"key1": "newvalue", "key2": {"newsubkey": "newsubvalue"}}
    expected = {"key1": "newvalue", "key2": {"newsubkey": "newsubvalue"}}
    result = recursive_dict_inject(d, inject)
    assert result == expected
