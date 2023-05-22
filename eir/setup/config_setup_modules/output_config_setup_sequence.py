from argparse import Namespace
from typing import Sequence, Iterable, Dict, Any, Optional, TYPE_CHECKING

from aislib.misc_utils import get_logger

from eir.models.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
)
from eir.setup.config_setup_modules.config_setup_utils import (
    get_yaml_iterator_with_injections,
)

if TYPE_CHECKING:
    from eir.setup.config import Configs

from eir.setup import config, schemas

logger = get_logger(name=__name__)


def get_configs_object_with_seq_output_configs(
    eir_configs: "Configs", cl_args: Namespace, extra_cl_args: list[str]
) -> "Configs":
    seq_config_iter = get_yaml_iterator_with_injections(
        yaml_config_files=cl_args.output_configs, extra_cl_args=extra_cl_args
    )
    seq_configs = get_seq_output_configs(
        seq_configs=seq_config_iter,
        base_output_configs=eir_configs.output_configs,
    )

    input_configs = eir_configs.input_configs
    extra_input_configs = converge_sequence_output_configs_to_input_configs(
        seq_output_configs=seq_configs,
        input_configs=input_configs,
    )
    # TODO: Filter the duplicated ones where we have original *and* extra
    seq_input_configs = list(input_configs) + list(extra_input_configs)

    seq_config_object_kwargs = eir_configs.__dict__
    seq_config_object_kwargs["output_configs"] = seq_configs
    seq_config_object_kwargs["input_configs"] = seq_input_configs

    configs_object = config.Configs(**seq_config_object_kwargs)

    return configs_object


def get_seq_output_configs(
    seq_configs: Iterable[dict],
    base_output_configs: Iterable[schemas.OutputConfig],
) -> Sequence[schemas.OutputConfig]:
    config_objects = []

    for config_dict in seq_configs:
        cur_base_output_config = next(
            i
            for i in base_output_configs
            if i.output_info.output_name == config_dict["output_info"]["output_name"]
        )

        config_object = init_seq_output_config(
            yaml_config_as_dict=config_dict, base_output_config=cur_base_output_config
        )
        config_objects.append(config_object)

    return config_objects


def init_seq_output_config(
    yaml_config_as_dict: Dict[str, Any],
    base_output_config: schemas.OutputConfig,
) -> schemas.OutputConfig:
    """
    We include the input type here for more fine grained logic later when setting up
    the target and input configs.
    """

    seq_sampling_config_kwargs = yaml_config_as_dict.get("sampling_config", {})
    seq_sampling_config = schemas.SequenceOutputSamplingConfig(
        **seq_sampling_config_kwargs
    )

    seq_config = schemas.OutputConfig(
        output_info=base_output_config.output_info,
        output_type_info=base_output_config.output_type_info,
        model_config=base_output_config.model_config,
        sampling_config=seq_sampling_config,
    )

    return seq_config


def converge_sequence_output_configs_to_input_configs(
    seq_output_configs: Sequence[schemas.OutputConfig],
    input_configs: Optional[Sequence[schemas.InputConfig]] = None,
) -> Sequence[schemas.InputConfig]:
    converged_input_configs = []

    for seq_output_config in seq_output_configs:
        if seq_output_config.output_info.output_type != "sequence":
            continue

        matched_input_config = None
        if input_configs:
            matched_input_configs = [
                i
                for i in input_configs
                if i.input_info.input_name == seq_output_config.output_info.output_name
            ]
            if matched_input_configs:
                assert len(matched_input_configs) == 1
                matched_input_config = matched_input_configs[0]
                logger.info(
                    "Found input config with matching name for SSL config: '%s'. "
                    "Will overload matching keys.",
                    seq_output_config.output_info.output_name,
                )

        cur_converged_input_config = _build_sequence_input_config_from_output(
            sequence_output_config=seq_output_config,
            input_config_to_overload=matched_input_config,
        )
        converged_input_configs.append(cur_converged_input_config)

    return converged_input_configs


def _build_sequence_input_config_from_output(
    sequence_output_config: schemas.OutputConfig,
    input_config_to_overload: Optional[schemas.InputConfig] = None,
) -> schemas.InputConfig:
    output_info = sequence_output_config.output_info
    input_info = schemas.InputDataConfig(
        input_source=output_info.output_source,
        input_type=output_info.output_type,
        input_name=output_info.output_name,
    )

    output_type_info = sequence_output_config.output_type_info
    output_type_info_kwargs = output_type_info.__dict__

    input_type_info_keys = schemas.SequenceInputDataConfig.__dataclass_fields__.keys()
    matched_kwargs = {
        k: v for k, v in output_type_info_kwargs.items() if k in input_type_info_keys
    }

    input_type_info = schemas.SequenceInputDataConfig(**matched_kwargs)

    # TODO: Extend
    model_init_config_kwargs = (
        sequence_output_config.model_config.model_init_config.__dict__
    )
    model_init_config = BasicTransformerFeatureExtractorModelConfig(
        **model_init_config_kwargs
    )

    if input_config_to_overload:
        model_init_config = input_config_to_overload.model_config.model_init_config
        rest = {
            k: v
            for k, v in input_config_to_overload.model_config.__dict__.items()
            if k not in {"model_type", "model_init_config"}
        }
        pretrained_config = input_config_to_overload.pretrained_config
    else:
        rest = {}
        pretrained_config = None

    overloaded_model_config = schemas.SequenceModelConfig(
        model_type="eir-sequence-output-linked-default",
        model_init_config=model_init_config,
        **rest,
    )
    converged_input_config = schemas.InputConfig(
        input_info=input_info,
        input_type_info=input_type_info,
        model_config=overloaded_model_config,
        pretrained_config=pretrained_config,
    )

    return converged_input_config
