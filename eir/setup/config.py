import argparse
import types
from argparse import Namespace
from copy import copy
from dataclasses import dataclass
from typing import (
    Sequence,
    Iterable,
    Dict,
    List,
    Union,
    Type,
    Any,
    Callable,
    overload,
    Generator,
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
from eir.setup import schemas

al_input_types = Union[schemas.OmicsInputDataConfig, schemas.TabularInputDataConfig]

logger = get_logger(name=__name__)


def get_configs():
    main_cl_args = get_main_cl_args()
    configs = generate_aggregated_config(cl_args=main_cl_args)

    return configs


def get_main_cl_args() -> argparse.Namespace:

    parser_ = get_main_parser()
    cl_args = parser_.parse_args()

    return cl_args


def get_main_parser() -> configargparse.ArgumentParser:
    parser_ = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser_.add_argument(
        "--global_configs", nargs="+", type=str, required=True, help=""
    )

    parser_.add_argument("--input_configs", type=str, nargs="+", required=True, help="")

    parser_.add_argument(
        "--predictor_configs", type=str, nargs="*", required=False, help=""
    )

    parser_.add_argument(
        "--target_configs", type=str, nargs="+", required=True, help=""
    )

    return parser_


@dataclass
class Configs:

    global_config: schemas.GlobalConfig
    input_configs: Sequence[schemas.InputConfig]
    predictor_config: schemas.PredictorConfig
    target_configs: Sequence[schemas.TargetConfig]


def generate_aggregated_config(
    cl_args: Union[argparse.Namespace, types.SimpleNamespace]
) -> Configs:
    """
    We manually assume and use only first element of predictor configs, as for now we
    will only support one configuration. Later we can use different predictor settings
    per target.
    """

    global_config_iter = get_yaml_to_dict_iterator(configs=cl_args.global_configs)
    global_config = get_global_config(global_configs=global_config_iter)

    inputs_config_iter = get_yaml_to_dict_iterator(configs=cl_args.input_configs)
    input_configs = get_input_configs(input_configs=inputs_config_iter)

    predictor_config_iter = get_yaml_to_dict_iterator(configs=cl_args.predictor_configs)
    predictor_config = load_predictor_config(predictor_configs=predictor_config_iter)

    target_config_iter = get_yaml_to_dict_iterator(configs=cl_args.target_configs)
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
    input_type_info_class = input_schema_map.get(input_info_object.input_type)
    input_type_info_object = input_type_info_class(**cfg["input_type_info"])

    model_config = set_up_model_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
    )

    input_config = schemas.InputConfig(
        input_info=input_info_object,
        input_type_info=input_type_info_object,
        model_config=model_config,
    )

    return input_config


def get_inputs_schema_map() -> Dict[
    str, Union[Type[schemas.OmicsInputDataConfig], Type[schemas.TabularInputDataConfig]]
]:
    mapping = {
        "omics": schemas.OmicsInputDataConfig,
        "tabular": schemas.TabularInputDataConfig,
    }

    return mapping


def set_up_model_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: al_input_types,
    model_init_kwargs_base: dict,
):
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
    model_config_class = model_config_map.get(input_type_info_object.model_type)
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
        "omics": set_up_omics_config_object_init_kwargs,
        "tabular": set_up_tabular_config_object_init_kwargs,
    }

    return mapping


def set_up_omics_config_object_init_kwargs(init_kwargs: dict, *args, **kwargs) -> dict:

    return init_kwargs


def set_up_tabular_config_object_init_kwargs(
    init_kwargs: dict, *args, **kwargs
) -> dict:
    return init_kwargs


def get_model_config_map() -> Dict[str, Type]:
    mapping = get_omics_config_dataclass_mapping()
    mapping = {**mapping, **{"tabular": eir.models.tabular.tabular.TabularModelConfig}}

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


def get_yaml_to_dict_iterator(configs: Iterable[str]) -> Generator[Dict, None, None]:
    for yaml_config in configs:
        yield load_yaml_config(config_path=yaml_config)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as yaml_file:
        config_as_dict = yaml.load(stream=yaml_file, Loader=yaml.FullLoader)

    return config_as_dict
