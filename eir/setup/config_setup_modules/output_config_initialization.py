from copy import copy
from typing import Any, Dict, Literal, Type, Union

from eir.models.input.array.models_locally_connected import LCLModelConfig
from eir.models.output.array.array_output_modules import (
    ArrayOutputModuleConfig,
    LCLOutputModelConfig,
)
from eir.models.output.array.output_array_models_cnn import CNNUpscaleModelConfig
from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig,
    TransformerSequenceOutputModuleConfig,
)
from eir.models.output.tabular.linear import LinearOutputModuleConfig
from eir.models.output.tabular.mlp_residual import ResidualMLPOutputModuleConfig
from eir.models.output.tabular.shared_mlp_residual import (
    SharedResidualMLPOutputModuleConfig,
)
from eir.models.output.tabular.tabular_output_modules import TabularOutputModuleConfig
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import (
    validate_keys_against_dataclass,
)
from eir.setup.tensor_broker_setup import set_up_tensor_broker_config
from eir.utils.logging import get_logger

al_output_model_config_classes = (
    Type[ResidualMLPOutputModuleConfig]
    | Type[LinearOutputModuleConfig]
    | Type[SharedResidualMLPOutputModuleConfig]
    | Type[TransformerSequenceOutputModuleConfig]
    | Type[LCLModelConfig]
    | Type[CNNUpscaleModelConfig]
)
al_output_model_configs = (
    ResidualMLPOutputModuleConfig
    | LinearOutputModuleConfig
    | SharedResidualMLPOutputModuleConfig
    | TransformerSequenceOutputModuleConfig
    | LCLModelConfig
    | CNNUpscaleModelConfig
)
al_output_model_init_map = dict[str, dict[str, al_output_model_config_classes]]

logger = get_logger(name=__name__)


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

        case (
            schemas.TabularOutputTypeConfig()
            | schemas.SequenceOutputTypeConfig()
            | schemas.SurvivalOutputTypeConfig()
        ):
            sampling_config_object = sampling_config
        case _:
            raise ValueError(f"Unknown output type config '{output_type_config}'.")

    return sampling_config_object


def get_outputs_types_schema_map() -> Dict[
    str,
    Type[schemas.TabularOutputTypeConfig]
    | Type[schemas.SequenceOutputTypeConfig]
    | Type[schemas.ArrayOutputTypeConfig]
    | Type[schemas.ImageOutputTypeConfig]
    | Type[schemas.SurvivalOutputTypeConfig],
]:
    mapping = {
        "tabular": schemas.TabularOutputTypeConfig,
        "sequence": schemas.SequenceOutputTypeConfig,
        "array": schemas.ArrayOutputTypeConfig,
        "image": schemas.ImageOutputTypeConfig,
        "survival": schemas.SurvivalOutputTypeConfig,
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
        "survival": TabularOutputModuleConfig,
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
            "shared_mlp_residual": SharedResidualMLPOutputModuleConfig,
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
        "survival": {
            "mlp_residual": ResidualMLPOutputModuleConfig,
            "linear": LinearOutputModuleConfig,
            "shared_mlp_residual": SharedResidualMLPOutputModuleConfig,
        },
    }

    return mapping
