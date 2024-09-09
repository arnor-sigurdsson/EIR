import argparse
import types
from collections import Counter
from copy import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    Type,
    Union,
)

import configargparse
import yaml

from eir.models.fusion.fusion_identity import IdentityConfig
from eir.models.fusion.fusion_mgmoe import MGMoEModelConfig
from eir.models.input.array.array_models import LCLModelConfig
from eir.models.layers.mlp_layers import ResidualMLPConfig
from eir.models.output.array.array_output_modules import (
    ArrayOutputModuleConfig,
    CNNUpscaleModelConfig,
    LCLOutputModelConfig,
)
from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig,
    TransformerSequenceOutputModuleConfig,
)
from eir.models.output.tabular.linear import LinearOutputModuleConfig
from eir.models.output.tabular.mlp_residual import ResidualMLPOutputModuleConfig
from eir.models.output.tabular.tabular_output_modules import TabularOutputModuleConfig
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_iterator_with_injections,
    validate_keys_against_dataclass,
)
from eir.setup.config_setup_modules.input_config_initialization import init_input_config
from eir.setup.config_setup_modules.output_config_setup_sequence import (
    get_configs_object_with_seq_output_configs,
)
from eir.setup.config_validation import validate_train_configs
from eir.setup.schema_modules.latent_analysis_schemas import LatentSamplingConfig
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.setup.schemas import (
    AttributionAnalysisConfig,
    BasicExperimentConfig,
    EvaluationCheckpointConfig,
    GlobalConfig,
    GlobalModelConfig,
    LRScheduleConfig,
    OptimizationConfig,
    SupervisedMetricsConfig,
    TrainingControlConfig,
    VisualizationLoggingConfig,
)
from eir.setup.tensor_broker_setup import set_up_tensor_broker_config
from eir.train_utils.utils import configure_global_eir_logging
from eir.utils.logging import get_logger

al_input_types = Union[
    schemas.OmicsInputDataConfig,
    schemas.TabularInputDataConfig,
    schemas.SequenceInputDataConfig,
    schemas.ByteInputDataConfig,
]


al_output_types_schema_map = dict[
    str,
    Union[
        Type[TabularOutputTypeConfig],
        Type,
    ],
]

al_output_module_config_class_getter = (
    Callable[[str], Union[schemas.al_output_module_configs_classes, Any]],
)

al_output_model_config_classes = (
    Type[ResidualMLPOutputModuleConfig]
    | Type[LinearOutputModuleConfig]
    | Type[TransformerSequenceOutputModuleConfig]
    | Type[LCLModelConfig]
    | Type[CNNUpscaleModelConfig]
)

al_output_model_configs = (
    ResidualMLPOutputModuleConfig
    | LinearOutputModuleConfig
    | TransformerSequenceOutputModuleConfig
    | LCLModelConfig
    | CNNUpscaleModelConfig
)
al_output_model_init_map = dict[str, dict[str, al_output_model_config_classes]]

logger = get_logger(name=__name__)


def get_configs():
    main_cl_args, extra_cl_args = get_main_cl_args()

    output_folder, log_level = get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args, extra_cl_args=extra_cl_args
    )

    configure_global_eir_logging(output_folder=output_folder, log_level=log_level)

    configs = generate_aggregated_config(
        cl_args=main_cl_args,
        extra_cl_args_overload=extra_cl_args,
    )

    configs_with_seq_outputs = get_configs_object_with_seq_output_configs(
        configs=configs,
    )
    validate_train_configs(configs=configs)

    return configs_with_seq_outputs


def get_main_cl_args() -> Tuple[argparse.Namespace, List[str]]:
    parser_ = get_main_parser()
    cl_args, extra_cl_args = parser_.parse_known_args()

    return cl_args, extra_cl_args


def get_output_folder_and_log_level_from_cl_args(
    main_cl_args: argparse.Namespace,
    extra_cl_args: List[str],
) -> Tuple[str, str]:
    global_configs = main_cl_args.global_configs

    output_folder = None
    for config in global_configs:
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)

            output_folder = config_dict.get("basic_experiment", {}).get("output_folder")
            log_level = config_dict.get("visualization_logging", {}).get(
                "log_level", "info"
            )

            if output_folder and log_level:
                break

    for arg in extra_cl_args:
        if "output_folder=" in arg:
            output_folder = arg.split("=")[1]
            break

    if output_folder is None:
        raise ValueError("Output folder not found in global configs.")

    return output_folder, log_level


def get_main_parser(
    global_nargs: Literal["+", "*"] = "+", output_nargs: Literal["+", "*"] = "+"
) -> configargparse.ArgumentParser:
    parser_ = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    global_required = True if global_nargs == "+" else False
    output_required = True if output_nargs == "+" else False

    parser_.add_argument(
        "--global_configs",
        nargs=global_nargs,
        type=str,
        required=global_required,
        help="Global .yaml configurations for the experiment.",
    )

    parser_.add_argument(
        "--input_configs",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Input feature extraction .yaml configurations. "
        "Each configuration represents one input.",
    )

    parser_.add_argument(
        "--fusion_configs",
        type=str,
        nargs="*",
        required=False,
        default=[],
        help="Fusion .yaml configurations.",
    )

    parser_.add_argument(
        "--output_configs",
        type=str,
        nargs=output_nargs,
        required=output_required,
        help="Output .yaml configurations.",
    )

    return parser_


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
    fusion_config: schemas.FusionConfig
    output_configs: Sequence[schemas.OutputConfig]

    gc: schemas.GlobalConfig = field(init=False, repr=False)

    def __post_init__(self):
        self.gc = self.global_config


def generate_aggregated_config(
    cl_args: Union[argparse.Namespace, types.SimpleNamespace],
    extra_cl_args_overload: Union[List[str], None] = None,
    strict: bool = True,
) -> Configs:
    global_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.global_configs, extra_cl_args=extra_cl_args_overload
    )
    global_config = get_global_config(global_configs=global_config_iter)

    inputs_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.input_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    input_configs = get_input_configs(input_configs=inputs_config_iter)

    fusion_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.fusion_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    fusion_config = load_fusion_configs(fusion_configs=fusion_config_iter)

    output_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.output_configs,
        extra_cl_args=extra_cl_args_overload,
    )
    output_configs = load_output_configs(output_configs=output_config_iter)

    if strict:
        _check_input_and_output_config_names(
            input_configs=input_configs, output_configs=output_configs
        )

    aggregated_configs = Configs(
        global_config=global_config,
        input_configs=input_configs,
        fusion_config=fusion_config,
        output_configs=output_configs,
    )

    return aggregated_configs


def get_global_config(global_configs: Iterable[dict]) -> GlobalConfig:
    combined_config = combine_dicts(dicts=global_configs)

    basic_experiment_config = BasicExperimentConfig(
        **combined_config.get("basic_experiment", {})
    )
    model_config = GlobalModelConfig(**combined_config.get("model", {}))
    optimization_config = OptimizationConfig(**combined_config.get("optimization", {}))
    lr_schedule_config = LRScheduleConfig(**combined_config.get("lr_schedule", {}))
    training_control_config = TrainingControlConfig(
        **combined_config.get("training_control", {})
    )
    evaluation_checkpoint_config = EvaluationCheckpointConfig(
        **combined_config.get("evaluation_checkpoint", {})
    )
    attribution_analysis_config = AttributionAnalysisConfig(
        **combined_config.get("attribution_analysis", {})
    )
    metrics_config = SupervisedMetricsConfig(**combined_config.get("metrics", {}))
    visualization_logging_config = VisualizationLoggingConfig(
        **combined_config.get("visualization_logging", {})
    )

    latent_sampling = None
    if "latent_sampling" in combined_config and isinstance(
        combined_config["latent_sampling"], dict
    ):
        latent_sampling = LatentSamplingConfig(**combined_config["latent_sampling"])

    global_config = GlobalConfig(
        basic_experiment=basic_experiment_config,
        model=model_config,
        optimization=optimization_config,
        lr_schedule=lr_schedule_config,
        training_control=training_control_config,
        evaluation_checkpoint=evaluation_checkpoint_config,
        attribution_analysis=attribution_analysis_config,
        metrics=metrics_config,
        visualization_logging=visualization_logging_config,
        latent_sampling=latent_sampling,
    )

    return modify_global_config(global_config)


def modify_global_config(global_config: GlobalConfig) -> GlobalConfig:
    gc_copy = copy(global_config)

    if gc_copy.basic_experiment.valid_size > 1.0:
        gc_copy.basic_experiment.valid_size = int(gc_copy.basic_experiment.valid_size)

    return gc_copy


def _maybe_add_latent_sampling_to_combined_config(combined_config: dict) -> dict:
    if "latent_sampling" in combined_config and isinstance(
        combined_config["latent_sampling"], dict
    ):
        combined_config["latent_sampling"] = LatentSamplingConfig(
            **combined_config["latent_sampling"]
        )

    return combined_config


def get_input_configs(
    input_configs: Iterable[dict],
) -> Sequence[schemas.InputConfig]:
    config_objects = []

    for config_dict in input_configs:
        config_object = init_input_config(yaml_config_as_dict=config_dict)
        config_objects.append(config_object)

    _check_input_config_names(input_configs=config_objects)
    return config_objects


def _check_input_config_names(input_configs: Iterable[schemas.InputConfig]) -> None:
    names = []
    for config in input_configs:
        names.append(config.input_info.input_name)

    if len(set(names)) != len(names):
        counts = Counter(names)
        raise ValueError(
            f"Found duplicates of input names in input configs: {counts}. "
            f"Please make sure each input source has a unique input_name."
        )


def load_fusion_configs(fusion_configs: Iterable[dict]) -> schemas.FusionConfig:
    combined_config = combine_dicts(dicts=fusion_configs)

    validate_keys_against_dataclass(
        input_dict=combined_config, dataclass_type=schemas.FusionConfig
    )

    combined_config.setdefault("model_type", "mlp-residual")
    combined_config.setdefault("model_config", {})

    fusion_model_type = combined_config["model_type"]

    model_dataclass_config_class: (
        Type[ResidualMLPConfig] | Type[MGMoEModelConfig] | Type[IdentityConfig]
    ) = ResidualMLPConfig
    if fusion_model_type == "mgmoe":
        model_dataclass_config_class = MGMoEModelConfig
    elif fusion_model_type == "linear":
        model_dataclass_config_class = IdentityConfig

    fusion_config_kwargs = combined_config["model_config"]
    fusion_model_config = model_dataclass_config_class(**fusion_config_kwargs)

    tensor_broker_config = set_up_tensor_broker_config(
        tensor_broker_config=combined_config.get("tensor_broker_config", {})
    )

    fusion_config = schemas.FusionConfig(
        model_type=combined_config["model_type"],
        model_config=fusion_model_config,
        tensor_broker_config=tensor_broker_config,
    )

    return fusion_config


def load_output_configs(
    output_configs: Iterable[dict],
) -> Sequence[schemas.OutputConfig]:
    output_config_objects = []

    for config_dict in output_configs:
        config_object = init_output_config(
            yaml_config_as_dict=config_dict,
        )
        output_config_objects.append(config_object)

    _check_output_config_names(output_configs=output_config_objects)
    return output_config_objects


def _check_output_config_names(output_configs: Iterable[schemas.OutputConfig]) -> None:
    names = []
    for config in output_configs:
        names.append(config.output_info.output_name)

    if len(set(names)) != len(names):
        counts = Counter(names)
        raise ValueError(
            f"Found duplicates of input names in input configs: {counts}. "
            f"Please make sure each input source has a unique input_name."
        )


def init_output_config(
    yaml_config_as_dict: Dict[str, Any],
) -> schemas.OutputConfig:
    cfg = yaml_config_as_dict

    validate_keys_against_dataclass(
        input_dict=cfg,
        dataclass_type=schemas.OutputConfig,
        name=cfg.get("output_info", {}).get("output_name", ""),
    )

    output_info_object = schemas.OutputInfoConfig(**cfg["output_info"])

    output_schema_map = get_outputs_types_schema_map()
    output_type_info_class = output_schema_map[output_info_object.output_type]

    output_type_info_class_init_kwargs = cfg.get("output_type_info", {})
    output_type_info_object = output_type_info_class(
        **output_type_info_class_init_kwargs
    )

    model_config = set_up_output_module_config(
        output_info_object=output_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
    )

    sampling_config = _set_up_basic_sampling_config(
        output_type_config=output_type_info_object,
        sampling_config=cfg.get("sampling_config", {}),
    )

    tensor_broker_config = set_up_tensor_broker_config(
        tensor_broker_config=cfg.get("tensor_broker_config", {})
    )

    output_config = schemas.OutputConfig(
        output_info=output_info_object,
        output_type_info=output_type_info_object,
        model_config=model_config,
        sampling_config=sampling_config,
        tensor_broker_config=tensor_broker_config,
    )

    return output_config


def _set_up_basic_sampling_config(
    output_type_config: schemas.al_output_type_configs, sampling_config: dict
) -> dict | schemas.ArrayOutputSamplingConfig | schemas.ImageOutputSamplingConfig:
    """
    Note that the sequence sampling config currently has it's own logic
    in output_config_setup_sequence.py.
    """
    sampling_config_object: (
        dict | schemas.ArrayOutputSamplingConfig | schemas.ImageOutputSamplingConfig
    )
    match output_type_config:
        case schemas.ArrayOutputTypeConfig():
            sampling_config_object = schemas.ArrayOutputSamplingConfig(
                **sampling_config
            )
        case schemas.ImageOutputTypeConfig():
            sampling_config_object = schemas.ImageOutputSamplingConfig(
                **sampling_config
            )

        case schemas.TabularOutputTypeConfig() | schemas.SequenceOutputTypeConfig():
            sampling_config_object = sampling_config
        case _:
            raise ValueError(f"Unknown output type config '{output_type_config}'.")

    return sampling_config_object


def get_outputs_types_schema_map() -> Dict[
    str,
    Type[schemas.TabularOutputTypeConfig]
    | Type[schemas.SequenceOutputTypeConfig]
    | Type[schemas.ArrayOutputTypeConfig]
    | Type[schemas.ImageOutputTypeConfig],
]:
    mapping = {
        "tabular": schemas.TabularOutputTypeConfig,
        "sequence": schemas.SequenceOutputTypeConfig,
        "array": schemas.ArrayOutputTypeConfig,
        "image": schemas.ImageOutputTypeConfig,
    }

    return mapping


def get_output_module_config_class(
    output_type: str,
) -> schemas.al_output_module_configs_classes:
    model_config_setup_map = get_output_module_config_class_map()

    return model_config_setup_map[output_type]


def get_output_module_config_class_map() -> (
    Dict[str, schemas.al_output_module_configs_classes]
):
    mapping = {
        "tabular": TabularOutputModuleConfig,
        "sequence": SequenceOutputModuleConfig,
        "array": ArrayOutputModuleConfig,
        "image": ArrayOutputModuleConfig,
    }

    return mapping


def set_up_output_module_config(
    output_info_object: schemas.OutputInfoConfig,
    model_init_kwargs_base: dict,
) -> schemas.al_output_module_configs:
    output_type = output_info_object.output_type

    model_config_class = get_output_module_config_class(output_type=output_type)

    model_type = None
    if model_init_kwargs_base:
        model_type = model_init_kwargs_base.get("model_type", None)

    if not model_type:
        try:
            model_type = getattr(model_config_class, "model_type")
        except AttributeError:
            raise AttributeError(
                "Not model type specified in model config and could not find default "
                "value for '%s'.",
                output_type,
            )

        logger.info(
            "Output model type not specified in model configuration with name '%s', "
            "attempting to grab default value.",
            output_info_object.output_name,
        )

    output_module_init_class_map = get_output_config_type_init_callable_map()

    model_type_config = set_up_output_module_init_config(
        model_init_kwargs_base=model_init_kwargs_base.get("model_init_config", {}),
        output_type=output_type,
        model_type=model_type,
        output_module_init_class_map=output_module_init_class_map,
    )

    common_kwargs = {"model_type": model_type, "model_init_config": model_type_config}
    other_specific_kwargs = {
        k: v for k, v in model_init_kwargs_base.items() if k not in common_kwargs
    }
    model_config_kwargs = {**common_kwargs, **other_specific_kwargs}
    model_config = model_config_class(**model_config_kwargs)

    return model_config


def set_up_output_module_init_config(
    model_init_kwargs_base: Union[None, dict],
    output_type: Literal["tabular", "sequence", "array"],
    model_type: str,
    output_module_init_class_map: al_output_model_init_map,
) -> al_output_model_configs:
    if not model_init_kwargs_base:
        model_init_kwargs_base = {}

    model_init_kwargs = copy(model_init_kwargs_base)

    model_init_config_callable = output_module_init_class_map[output_type][model_type]

    model_init_config = model_init_config_callable(**model_init_kwargs)

    return model_init_config


def get_output_config_type_init_callable_map() -> al_output_model_init_map:
    mapping: al_output_model_init_map = {
        "tabular": {
            "mlp_residual": ResidualMLPOutputModuleConfig,
            "linear": LinearOutputModuleConfig,
        },
        "sequence": {
            "sequence": TransformerSequenceOutputModuleConfig,
        },
        "array": {
            "lcl": LCLOutputModelConfig,
            "cnn": CNNUpscaleModelConfig,
        },
        "image": {
            "lcl": LCLOutputModelConfig,
            "cnn": CNNUpscaleModelConfig,
        },
    }

    return mapping


@dataclass
class TabularTargets:
    con_targets: Dict[str, Sequence[str]]
    cat_targets: Dict[str, Sequence[str]]

    def __len__(self):
        all_con = sum(len(i) for i in self.con_targets.values())
        all_cat = sum(len(i) for i in self.cat_targets.values())
        return all_con + all_cat


def get_all_tabular_targets(
    output_configs: Iterable[schemas.OutputConfig],
) -> TabularTargets:
    con_targets = {}
    cat_targets = {}

    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            continue

        output_name = output_config.output_info.output_name
        output_type_info = output_config.output_type_info
        assert isinstance(output_type_info, TabularOutputTypeConfig)
        con_targets[output_name] = output_type_info.target_con_columns
        cat_targets[output_name] = output_type_info.target_cat_columns

    targets = TabularTargets(con_targets=con_targets, cat_targets=cat_targets)
    return targets


def _check_input_and_output_config_names(
    input_configs: Sequence[schemas.InputConfig],
    output_configs: Sequence[schemas.OutputConfig],
    skip_keys: Sequence[str] = ("sequence", "array", "image"),
) -> None:
    """
    We allow for the same name to be used for sequence inputs and outputs,
    as it's used for specifying e.g. different model settings for the input feature
    extractor vs. the output module.
    """
    input_names = set(i.input_info.input_name for i in input_configs)
    output_names = set(
        i.output_info.output_name
        for i in output_configs
        if i.output_info.output_type not in skip_keys
    )

    intersection = output_names.intersection(input_names)
    if len(intersection) > 0:
        raise ValueError(
            "Found common names in input and output configs. Please ensure"
            " that there are no common names between the input and output configs."
            " Input config names: '%s'.\n"
            " Output config names: '%s'.\n",
            " Common names: '%s'.\n",
            input_names,
            output_names,
            intersection,
        )


def combine_dicts(dicts: Iterable[dict]) -> dict:
    combined_dict: dict = {}

    for dict_ in dicts:
        combined_dict = {**combined_dict, **dict_}

    return combined_dict
