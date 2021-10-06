from argparse import Namespace

from eir.setup import config
from eir.setup.presets.gln import PRESET as GLN_PRESET


def test_add_preset_to_cl_args():
    test_cl_args = Namespace(preset="gln")
    cl_args_with_preset_configs = config.add_preset_to_cl_args(cl_args=test_cl_args)
    assert len(cl_args_with_preset_configs.__dict__) == 5


def test_prepare_preset_tmp_dir():
    preset_yaml_files = config.prepare_preset_tmp_dir(
        preset_dict=GLN_PRESET, preset_name="gln"
    )
    assert len(GLN_PRESET) == len(preset_yaml_files)
    assert set(preset_yaml_files.keys()) == {
        "global_configs",
        "input_configs",
        "predictor_configs",
        "target_configs",
    }

    for config_name, yaml_file_paths in preset_yaml_files.items():
        assert len(yaml_file_paths) == 1
        yaml_file_path = yaml_file_paths[0]
        loaded_dict_from_yaml = config.load_yaml_config(config_path=yaml_file_path)

        preset_dict_configs = GLN_PRESET[config_name]
        assert len(preset_dict_configs) == 1
        preset_config = list(preset_dict_configs.values())[0]

        assert loaded_dict_from_yaml == preset_config


def test_generate_aggregated_config_basic():
    test_cl_args = Namespace(preset="gln")
    cl_args_with_preset_configs = config.add_preset_to_cl_args(cl_args=test_cl_args)

    aggregated_config = config.generate_aggregated_config(
        cl_args=cl_args_with_preset_configs
    )
    assert aggregated_config.global_config.run_name == "gln_run"

    assert len(aggregated_config.input_configs) == 1
    assert aggregated_config.input_configs[0].input_info.input_source == "MUST_FILL"

    assert len(aggregated_config.target_configs) == 1
    assert aggregated_config.target_configs[0].target_cat_columns == "MUST_FILL"


def test_generate_aggregated_config_with_overload():
    test_cl_args = Namespace(preset="gln")
    cl_args_with_preset_configs = config.add_preset_to_cl_args(cl_args=test_cl_args)

    aggregated_config = config.generate_aggregated_config(
        cl_args=cl_args_with_preset_configs,
        extra_cl_args_overload=["--gln_input.input_info.input_source=test_value"],
    )
    assert aggregated_config.global_config.run_name == "gln_run"
    assert len(aggregated_config.input_configs) == 1
    assert aggregated_config.input_configs[0].input_info.input_source == "test_value"


def test_get_yaml_iterator_with_injections():
    test_cl_args = Namespace(preset="gln")
    cl_args_with_preset_configs = config.add_preset_to_cl_args(cl_args=test_cl_args)

    input_yaml_files = cl_args_with_preset_configs.input_configs
    assert len(input_yaml_files) == 1
    assert input_yaml_files[0].stem == "gln_input"

    extra_cl_args_overload = ["--gln_input.input_info.input_source=test_value"]

    yaml_iter_with_injections = config.get_yaml_iterator_with_injections(
        yaml_config_files=input_yaml_files, extra_cl_args=extra_cl_args_overload
    )
    overloaded_config = next(yaml_iter_with_injections)
    assert overloaded_config["input_info"]["input_source"] == "test_value"


def test_convert_cl_str_to_dict():
    test_str = "gln_input.input_info.input_source=test_value"
    test_dict = config.convert_cl_str_to_dict(str_=test_str)
    assert test_dict == {"gln_input": {"input_info": {"input_source": "test_value"}}}
