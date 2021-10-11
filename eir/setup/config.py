import argparse
import ast
import atexit
import json
import operator
import shutil
import sys
import tempfile
import types
from argparse import Namespace
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import (
    Sequence,
    Iterable,
    Dict,
    List,
    Union,
    Tuple,
    Type,
    Any,
    Callable,
    overload,
    Generator,
    Mapping,
)

import configargparse
import torch
import yaml
from aislib.misc_utils import get_logger

import eir.models.tabular.tabular
from eir.models.fusion import FusionModelConfig
from eir.models.fusion_linear import LinearFusionModelConfig
from eir.models.fusion_mgmoe import MGMoEModelConfig
from eir.models.omics.omics_models import get_omics_config_dataclass_mapping
from eir.models.sequence.transformer_basic import BasicTransformerModelConfig
from eir.setup import schemas
from eir.setup.presets import gln

al_input_types = Union[
    schemas.OmicsInputDataConfig,
    schemas.TabularInputDataConfig,
    schemas.SequenceInputDataConfig,
]

logger = get_logger(name=__name__)


def get_configs():
    main_cl_args, extra_cl_args = get_main_cl_args()
    configs = generate_aggregated_config(
        cl_args=main_cl_args, extra_cl_args_overload=extra_cl_args
    )
    return configs


def get_main_cl_args() -> Tuple[argparse.Namespace, List[str]]:
    parser_ = get_main_parser()
    cl_args, extra_cl_args = parser_.parse_known_args()

    cl_args = add_preset_to_cl_args(cl_args=cl_args)

    return cl_args, extra_cl_args


def get_main_parser() -> configargparse.ArgumentParser:
    parser_ = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    config_nargs = "*" if "--preset" in sys.argv else "+"
    config_required = False if "--preset" in sys.argv else True

    parser_.add_argument(
        "--preset",
        type=str,
        choices=["gln"],
        default=None,
        help="Whether and which preset to use that is built into the framework.",
    )

    parser_.add_argument(
        "--global_configs",
        nargs=config_nargs,
        type=str,
        required=config_required,
        help="Global .yaml configurations for the experiment.",
    )

    parser_.add_argument(
        "--input_configs",
        type=str,
        nargs=config_nargs,
        required=config_required,
        help="Input feature extraction .yaml configurations. "
        "Each configuration represents one input.",
    )

    parser_.add_argument(
        "--predictor_configs",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Predictor .yaml configurations.",
    )

    parser_.add_argument(
        "--target_configs",
        type=str,
        nargs=config_nargs,
        required=config_required,
        help="Target .yaml configurations.",
    )

    return parser_


def add_preset_to_cl_args(cl_args: Namespace) -> Namespace:

    if not cl_args.preset:
        return cl_args

    preset_map = _get_preset_map()
    preset_dict = preset_map.get(cl_args.preset)
    preset_dir_dict = prepare_preset_tmp_dir(
        preset_dict=preset_dict, preset_name=cl_args.preset
    )

    cl_args_copy = copy(cl_args)
    for key, value in preset_dir_dict.items():
        setattr(cl_args_copy, key, value)

    return cl_args_copy


def prepare_preset_tmp_dir(preset_dict: Dict, preset_name: str) -> Dict[str, List[str]]:
    expected_keys = {
        "global_configs",
        "input_configs",
        "target_configs",
        "predictor_configs",
    }

    def _cleanup(tmp_dir: Path):
        if tmp_dir.exists():
            shutil.rmtree(path=tmp_dir)

    temp_dir = tempfile.mkdtemp()
    temp_dir_path_object = Path(str(temp_dir))

    remove_func = partial(_cleanup, tmp_dir=temp_dir_path_object)
    atexit.register(remove_func)

    overload_names = []
    preset_yaml_files = {}
    log_targets = []

    for main_key, inner_dicts in preset_dict.items():
        assert main_key in expected_keys
        preset_yaml_files[main_key] = []

        for inner_key, inner_dict in inner_dicts.items():
            overload_names.append(inner_key)

            for match, *_ in _recursive_search(dict_=inner_dict, target="MUST_FILL"):
                match_str = ".".join(match)
                log_targets.append(f"{inner_key}.{match_str}")

            tmp_yaml_outpath = temp_dir_path_object / f"{inner_key}.yaml"
            with open(tmp_yaml_outpath, "w") as out_yaml:
                yaml.dump(data=inner_dict, stream=out_yaml)
            preset_yaml_files[main_key].append(tmp_yaml_outpath)

    logger.info("Preset keys for overloading are: %s", overload_names)

    log_str = (
        f"Following keys will have to be overloaded when using "
        f"'{[preset_name]}' preset:"
    )

    logger.info(f"{log_str} {log_targets}")
    return preset_yaml_files


def _get_preset_map() -> Dict[str, Dict]:
    preset_map = {"gln": gln.PRESET}

    return preset_map


def _recursive_search(
    dict_: Mapping, target: Any, path=None
) -> Generator[Tuple[str, Any], None, None]:
    if not path:
        path = []

    for key, value in dict_.items():
        local_path = copy(path)
        local_path.append(key)
        if isinstance(value, Mapping):
            yield from _recursive_search(dict_=value, target=target, path=local_path)
        else:
            if value == target:
                yield local_path, value


@dataclass
class Configs:
    global_config: schemas.GlobalConfig
    input_configs: Sequence[schemas.InputConfig]
    predictor_config: schemas.PredictorConfig
    target_configs: Sequence[schemas.TargetConfig]


def generate_aggregated_config(
    cl_args: Union[argparse.Namespace, types.SimpleNamespace],
    extra_cl_args_overload: Union[List[str], None] = None,
) -> Configs:
    """
    We manually assume and use only first element of predictor configs, as for now we
    will only support one configuration. Later we can use different predictor settings
    per target.
    """

    global_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.global_configs, extra_cl_args=extra_cl_args_overload
    )
    global_config = get_global_config(global_configs=global_config_iter)

    inputs_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.input_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    input_configs = get_input_configs(input_configs=inputs_config_iter)

    predictor_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.predictor_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    predictor_config = load_predictor_config(predictor_configs=predictor_config_iter)

    target_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.target_configs, extra_cl_args=extra_cl_args_overload
    )
    target_configs = load_configs_general(
        config_dict_iterable=target_config_iter, cls=schemas.TargetConfig
    )

    aggregated_configs = Configs(
        global_config=global_config,
        input_configs=input_configs,
        predictor_config=predictor_config,
        target_configs=target_configs,
    )

    return aggregated_configs


def get_global_config(global_configs: Iterable[dict]) -> schemas.GlobalConfig:
    combined_config = combine_dicts(dicts=global_configs)

    combined_config_namespace = Namespace(**combined_config)

    global_config_object = schemas.GlobalConfig(**combined_config_namespace.__dict__)

    global_config_object_mod = modify_global_config(global_config=global_config_object)

    return global_config_object_mod


def modify_global_config(global_config: schemas.GlobalConfig) -> schemas.GlobalConfig:
    gc_copy = copy(global_config)

    if gc_copy.valid_size > 1.0:
        gc_copy.valid_size = int(gc_copy.valid_size)

    gc_copy.device = "cuda:" + gc_copy.gpu_num if torch.cuda.is_available() else "cpu"

    # benchmark breaks if we run it with multiple GPUs
    if not gc_copy.multi_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        logger.debug("Setting device to cuda:0 since running with multiple GPUs.")
        gc_copy.device = "cuda:0"

    return gc_copy


def get_input_configs(
    input_configs: Iterable[dict],
) -> Sequence[schemas.InputConfig]:
    config_objects = []

    for config_dict in input_configs:
        config_object = init_input_config(yaml_config_as_dict=config_dict)
        config_objects.append(config_object)

    return config_objects


def init_input_config(yaml_config_as_dict: Dict[str, Any]) -> schemas.InputConfig:
    cfg = yaml_config_as_dict

    input_info_object = schemas.InputDataConfig(**cfg["input_info"])

    input_schema_map = get_inputs_schema_map()
    input_type_info_class = input_schema_map[input_info_object.input_type]
    input_type_info_object = input_type_info_class(**cfg["input_type_info"])

    model_config = set_up_model_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
    )

    interpretation_config = set_up_interpretation_config(
        input_type=input_info_object.input_type,
        interpretation_config_dict=cfg.get("interpretation_config", None),
    )

    input_config = schemas.InputConfig(
        input_info=input_info_object,
        input_type_info=input_type_info_object,
        model_config=model_config,
        interpretation_config=interpretation_config,
    )

    return input_config


def get_inputs_schema_map() -> Dict[
    str,
    Union[
        Type[schemas.OmicsInputDataConfig],
        Type[schemas.TabularInputDataConfig],
        Type[schemas.SequenceInputDataConfig],
    ],
]:
    mapping = {
        "omics": schemas.OmicsInputDataConfig,
        "tabular": schemas.TabularInputDataConfig,
        "sequence": schemas.SequenceInputDataConfig,
    }

    return mapping


def set_up_model_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: al_input_types,
    model_init_kwargs_base: Union[None, dict],
):
    if not model_init_kwargs_base:
        model_init_kwargs_base = {}

    init_kwargs = copy(model_init_kwargs_base)

    cur_setup_hook = get_model_config_setup_hook(
        input_type=input_info_object.input_type
    )
    if cur_setup_hook:
        init_kwargs = cur_setup_hook(
            init_kwargs=init_kwargs,
            input_info_object=input_info_object,
            input_type_info_object=input_type_info_object,
        )

    model_config_map = get_model_config_map()
    model_config_class = model_config_map[input_type_info_object.model_type]

    model_config = model_config_class(**init_kwargs)

    return model_config


@overload
def get_model_config_setup_hook(
    input_type: str,
) -> Callable[[dict, schemas.InputDataConfig, al_input_types], dict]:
    ...


@overload
def get_model_config_setup_hook(input_type: None) -> None:
    ...


def get_model_config_setup_hook(input_type):
    model_config_setup_map = get_model_config_setup_hook_map()

    return model_config_setup_map.get(input_type, None)


def get_model_config_setup_hook_map():
    mapping = {
        "omics": set_up_config_object_init_kwargs_identity,
        "tabular": set_up_config_object_init_kwargs_identity,
        "sequence": set_up_config_object_init_kwargs_identity,
    }

    return mapping


def set_up_config_object_init_kwargs_identity(
    init_kwargs: dict, *args, **kwargs
) -> dict:
    return init_kwargs


def get_model_config_map() -> Dict[str, Type]:
    mapping = get_omics_config_dataclass_mapping()
    mapping = {
        **mapping,
        **{
            "tabular": eir.models.tabular.tabular.TabularModelConfig,
            "sequence-default": BasicTransformerModelConfig,
        },
    }

    return mapping


def set_up_interpretation_config(
    input_type: str, interpretation_config_dict: Union[None, Dict[str, Any]]
) -> Union[None, schemas.SequenceInterpretationConfig]:

    config_class = get_interpretation_config_class(input_type=input_type)
    if config_class is None:
        return None

    if interpretation_config_dict is None:
        interpretation_config_dict = {}

    config_object = config_class(**interpretation_config_dict)

    return config_object


def get_interpretation_config_class(input_type: str):
    mapping = get_interpretation_config_schema_map()

    return mapping.get(input_type, None)


def get_interpretation_config_schema_map():
    mapping = {
        "sequence": schemas.SequenceInterpretationConfig,
    }

    return mapping


def load_predictor_config(predictor_configs: Iterable[dict]) -> schemas.PredictorConfig:
    combined_config = combine_dicts(dicts=predictor_configs)

    combined_config.setdefault("model_type", "default")
    combined_config.setdefault("model_config", {})

    fusion_model_type = combined_config["model_type"]

    model_dataclass_config_class = FusionModelConfig
    if fusion_model_type == "mgmoe":
        model_dataclass_config_class = MGMoEModelConfig
    elif fusion_model_type == "linear":
        model_dataclass_config_class = LinearFusionModelConfig

    fusion_config_kwargs = combined_config["model_config"]
    fusion_config = model_dataclass_config_class(**fusion_config_kwargs)

    predictor_config = schemas.PredictorConfig(
        model_type=combined_config["model_type"], model_config=fusion_config
    )

    return predictor_config


def load_configs_general(config_dict_iterable: Iterable[dict], cls: Type):
    config_objects = []

    for config_dict in config_dict_iterable:
        config_object = cls(**config_dict)
        config_objects.append(config_object)

    return config_objects


def get_target_configs(target_configs: Iterable[str]) -> Sequence[schemas.TargetConfig]:
    config_objects = []

    for config_path in target_configs:
        config_as_dict = load_yaml_config(config_path=config_path)
        config_object = schemas.TargetConfig(**config_as_dict)
        config_objects.append(config_object)

    return config_objects


@dataclass
class Targets:
    con_targets: List[str]
    cat_targets: List[str]

    @property
    def all_targets(self):
        return self.con_targets + self.cat_targets

    def __len__(self):
        return len(self.all_targets)


def get_all_targets(targets_configs: Iterable[schemas.TargetConfig]) -> Targets:
    con_targets = []
    cat_targets = []

    for target_config in targets_configs:
        con_targets += target_config.target_con_columns
        cat_targets += target_config.target_cat_columns

    targets = Targets(con_targets=con_targets, cat_targets=cat_targets)
    return targets


def combine_dicts(dicts: Iterable[dict]) -> dict:
    combined_dict = {}

    for dict_ in dicts:
        combined_dict = {**combined_dict, **dict_}

    return combined_dict


def get_yaml_iterator_with_injections(
    yaml_config_files: Iterable[str], extra_cl_args: List[str]
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
    def _infinite_dict():
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
) -> Generator[Dict, None, None]:
    for yaml_config in yaml_config_files:
        yield load_yaml_config(config_path=yaml_config)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as yaml_file:
        config_as_dict = yaml.load(stream=yaml_file, Loader=yaml.FullLoader)

    return config_as_dict


def recursive_dict_replace(dict_: dict, dict_to_inject: dict) -> dict:
    for cur_key, cur_value in dict_to_inject.items():

        if cur_key not in dict_:
            dict_[cur_key] = {}

        old_dict_value = dict_.get(cur_key)
        if isinstance(cur_value, Mapping):
            assert isinstance(old_dict_value, Mapping)
            recursive_dict_replace(old_dict_value, cur_value)
        else:
            dict_[cur_key] = cur_value

    return dict_


def object_to_primitives(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))
