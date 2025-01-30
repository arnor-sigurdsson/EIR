import argparse
import types
from collections import Counter
from copy import copy
from dataclasses import dataclass, field
from typing import (
    Any,
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
from eir.models.layers.mlp_layers import ResidualMLPConfig
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_iterator_with_injections,
    validate_keys_against_dataclass,
)
from eir.setup.config_setup_modules.input_config_initialization import init_input_config
from eir.setup.config_setup_modules.output_config_initialization import (
    init_output_config,
)
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

logger = get_logger(name=__name__)


def get_configs():
    main_cl_args, extra_cl_args = get_main_cl_args()

    output_folder, log_level = get_output_folder_and_log_level_from_cl_args(
        main_cl_args=main_cl_args,
        extra_cl_args=extra_cl_args,
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
        input_name = config.input_info.input_name

        model_name = config.model_config.model_type
        if model_name.startswith("eir-input-sequence-from-linked-output-"):
            logger.info(
                "Skipping input config name check for sequence input config "
                "(name='%s') as it was automatically generated from a sequence output.",
                input_name,
            )
            continue

        names.append(input_name)

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
