from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Union, Sequence, Mapping

import pytest
import yaml
from aislib.misc_utils import ensure_path_exists

import eir.predict_modules
from eir.predict_modules.predict_config import (
    get_named_pred_dict_iterators,
    get_train_predict_matched_config_generator,
)
import eir.setup
from eir.setup import schemas, config
from eir.setup.config import object_to_primitives


def test_get_named_pred_dict_iterators(tmp_path: Path) -> None:

    keys = {"global_configs", "input_configs", "fusion_configs", "output_configs"}
    paths = {}

    for k in keys:
        test_yaml_data = {f"key_{k}": f"value_{k}"}
        cur_outpath = (tmp_path / k).with_suffix(".yaml")
        ensure_path_exists(path=cur_outpath)

        with open(cur_outpath, "w") as out_yaml:
            yaml.dump(data=test_yaml_data, stream=out_yaml)

        paths.setdefault(k, []).append(cur_outpath)

    test_predict_cl_args = Namespace(**paths)

    named_iterators = get_named_pred_dict_iterators(
        predict_cl_args=test_predict_cl_args
    )

    for key, key_iter in named_iterators.items():
        for dict_ in key_iter:
            assert dict_ == {f"key_{key}": f"value_{key}"}


al_config_instances = Union[
    schemas.GlobalConfig,
    schemas.InputConfig,
    schemas.OutputConfig,
    schemas.TabularModelOutputConfig,
]


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
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
        },
    ],
    indirect=True,
)
def test_get_train_predict_matched_config_generator(create_test_config, tmp_path: Path):
    test_configs = create_test_config

    test_predict_cl_args = setup_test_namespace_for_matched_config_test(
        test_configs=test_configs,
        predict_cl_args_save_path=tmp_path,
        do_inject_test_values=True,
    )

    named_test_iterators = get_named_pred_dict_iterators(
        predict_cl_args=test_predict_cl_args
    )

    matched_iterator = get_train_predict_matched_config_generator(
        train_configs=test_configs, named_dict_iterators=named_test_iterators
    )

    # TODO: Note that these conditions currently come from
    #       _overload_test_yaml_object_for_predict. Later we can configure this
    #       further.
    for name, train_config_dict, predict_config_dict_to_inject in matched_iterator:
        if name == "input_configs":
            assert train_config_dict != predict_config_dict_to_inject
            assert (
                predict_config_dict_to_inject["input_info"]["input_source"]
                == "predict_input_source_overloaded"
            )
        else:
            assert train_config_dict == predict_config_dict_to_inject


def setup_test_namespace_for_matched_config_test(
    test_configs: config.Configs,
    predict_cl_args_save_path: Path,
    do_inject_test_values: bool = True,
    monkeypatch_train_to_test_paths: bool = False,
) -> Namespace:
    keys = ("global_configs", "input_configs", "fusion_configs", "output_configs")
    name_to_attr_map = {
        "global_configs": "global_config",
        "fusion_configs": "fusion_config",
    }
    paths = {}
    for k in keys:
        attr_name = name_to_attr_map.get(k, k)
        test_yaml_obj = getattr(test_configs, attr_name)

        obj_as_primitives = _overload_test_yaml_object_for_predict(
            test_yaml_obj=test_yaml_obj,
            cur_key=k,
            do_inject_test_values=do_inject_test_values,
        )

        if isinstance(obj_as_primitives, Sequence):
            name_object_iterator = enumerate(obj_as_primitives)
        elif isinstance(obj_as_primitives, Mapping):
            name_object_iterator = enumerate([obj_as_primitives])
        else:
            raise ValueError()

        for idx, obj_primitive_to_dump in name_object_iterator:
            cur_outpath = (predict_cl_args_save_path / f"{k}_{idx}").with_suffix(
                ".yaml"
            )
            ensure_path_exists(path=cur_outpath)

            if monkeypatch_train_to_test_paths:
                obj_primitive_to_dump = _recursive_dict_str_value_replace(
                    dict_=obj_primitive_to_dump, old="train", new="test"
                )

            with open(cur_outpath, "w") as out_yaml:
                yaml.dump(data=obj_primitive_to_dump, stream=out_yaml)

            paths.setdefault(k, []).append(cur_outpath)

    test_predict_cl_args = Namespace(**paths)

    return test_predict_cl_args


def _recursive_dict_str_value_replace(dict_: dict, old: str, new: str):
    for key, value in dict_.items():
        if isinstance(value, dict):
            _recursive_dict_str_value_replace(dict_=value, old=old, new=new)
        elif isinstance(value, str):
            if old in value:
                dict_[key] = value.replace(old, new)

    return dict_


def _overload_test_yaml_object_for_predict(
    test_yaml_obj: al_config_instances, cur_key: str, do_inject_test_values: bool = True
):
    test_yaml_obj_copy = deepcopy(test_yaml_obj)
    obj_as_primitives = object_to_primitives(obj=test_yaml_obj_copy)
    if cur_key == "input_configs":
        for idx, input_dict in enumerate(obj_as_primitives):
            if do_inject_test_values:
                input_dict = eir.setup.config.recursive_dict_replace(
                    dict_=input_dict,
                    dict_to_inject={
                        "input_info": {
                            "input_source": "predict_input_source_overloaded"
                        }
                    },
                )
            obj_as_primitives[idx] = input_dict

    return obj_as_primitives


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary"},
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
        },
    ],
    indirect=True,
)
def test_overload_train_configs_for_predict(
    create_test_config: config.Configs, tmp_path: Path
) -> None:

    test_configs = create_test_config

    test_predict_cl_args = setup_test_namespace_for_matched_config_test(
        test_configs=test_configs,
        predict_cl_args_save_path=tmp_path,
        do_inject_test_values=True,
    )

    named_test_iterators = (
        eir.predict_modules.predict_config.get_named_pred_dict_iterators(
            predict_cl_args=test_predict_cl_args
        )
    )

    matched_iterator = (
        eir.predict_modules.predict_config.get_train_predict_matched_config_generator(
            train_configs=test_configs, named_dict_iterators=named_test_iterators
        )
    )

    overloaded_train_config = (
        eir.predict_modules.predict_config.overload_train_configs_for_predict(
            matched_dict_iterator=matched_iterator
        )
    )

    # TODO: Note that these conditions currently come from
    #       _overload_test_yaml_object_for_predict. Later we can configure this
    #       further.
    for input_config in overloaded_train_config.input_configs:
        assert input_config.input_info.input_source == "predict_input_source_overloaded"