import ast
import json
import operator
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, DefaultDict, Dict, Generator, Iterable, MutableMapping, Optional

import yaml

from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def get_yaml_iterator_with_injections(
    yaml_config_files: Iterable[str], extra_cl_args: Optional[list[str]]
) -> Generator[Dict, None, None]:
    if not extra_cl_args:
        yield from get_yaml_to_dict_iterator(yaml_config_files=yaml_config_files)
        return

    for yaml_config_file in yaml_config_files:
        loaded_yaml = load_yaml_config(config_path=yaml_config_file)

        yaml_file_path_object = Path(yaml_config_file)
        for extra_arg in extra_cl_args:
            extra_arg_parsed = extra_arg.lstrip("--")
            target_file, str_to_inject = extra_arg_parsed.split(".", 1)

            if target_file == yaml_file_path_object.stem:
                dict_to_inject = convert_cl_str_to_dict(str_=str_to_inject)

                logger.debug("Injecting %s into %s", dict_to_inject, loaded_yaml)
                loaded_yaml = recursive_dict_replace(
                    dict_=loaded_yaml, dict_to_inject=dict_to_inject
                )

        yield loaded_yaml


def convert_cl_str_to_dict(str_: str) -> dict:
    def _infinite_dict() -> DefaultDict:
        return defaultdict(_infinite_dict)

    infinite_dict = _infinite_dict()

    keys, final_value = str_.split("=", 1)
    keys_split = keys.split(".")

    try:
        final_value_parsed = ast.literal_eval(final_value)
    except (ValueError, SyntaxError):
        final_value_parsed = final_value

    inner_most_dict = reduce(operator.getitem, keys_split[:-1], infinite_dict)
    inner_most_dict[keys_split[-1]] = final_value_parsed

    dict_primitive = object_to_primitives(obj=infinite_dict)
    return dict_primitive


def get_yaml_to_dict_iterator(
    yaml_config_files: Iterable[str],
) -> Generator[dict, None, None]:
    for yaml_config in yaml_config_files:
        yield load_yaml_config(config_path=yaml_config)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as yaml_file:
        config_as_dict = yaml.load(stream=yaml_file, Loader=yaml.FullLoader)

    return config_as_dict


def recursive_dict_replace(
    dict_: MutableMapping,
    dict_to_inject: MutableMapping,
) -> dict:
    for cur_key, cur_value in dict_to_inject.items():
        if cur_key not in dict_:
            dict_[cur_key] = {}

        old_dict_value = dict_.get(cur_key)
        cur_is_dict = isinstance(cur_value, MutableMapping)
        old_is_dict = isinstance(old_dict_value, MutableMapping)
        if cur_is_dict and old_is_dict:
            assert isinstance(cur_value, MutableMapping)
            assert isinstance(old_dict_value, MutableMapping)
            recursive_dict_replace(dict_=old_dict_value, dict_to_inject=cur_value)
        else:
            dict_[cur_key] = cur_value

    return dict(dict_)


def object_to_primitives(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))
