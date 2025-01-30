from typing import TYPE_CHECKING, Any

from eir.models.input.array.array_models import (
    ArrayModelConfig,
    get_array_config_dataclass_mapping,
)
from eir.models.input.image.image_models import ImageModelConfig
from eir.models.input.omics.omics_models import (
    OmicsModelConfig,
    get_omics_config_dataclass_mapping,
)
from eir.models.input.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
    SequenceModelConfig,
)
from eir.models.input.tabular.tabular import (
    SimpleTabularModelConfig,
    TabularModelConfig,
)
from eir.setup import schemas
from eir.setup.config_setup_modules.config_setup_utils import (
    validate_keys_against_dataclass,
)
from eir.setup.tensor_broker_setup import set_up_tensor_broker_config
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


if TYPE_CHECKING:
    from eir.setup.config import al_input_types


def init_input_config(yaml_config_as_dict: dict[str, Any]) -> schemas.InputConfig:
    cfg = yaml_config_as_dict

    validate_keys_against_dataclass(
        input_dict=cfg,
        dataclass_type=schemas.InputConfig,
        name=cfg.get("input_info", {}).get("input_name", ""),
    )

    input_info_object = schemas.InputDataConfig(**cfg["input_info"])

    input_schema_map = get_inputs_schema_map()
    input_type_info_class = input_schema_map[input_info_object.input_type]
    input_type_info_kwargs = cfg.get("input_type_info", {})
    input_type_info_object = input_type_info_class(**input_type_info_kwargs)

    _validate_input_type_info_object(
        input_type_info_object=input_type_info_object,
        input_source=input_info_object.input_source,
    )

    model_config = set_up_input_feature_extractor_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs_base=cfg.get("model_config", {}),
    )

    pretrained_config = set_up_pretrained_config(
        pretrained_config_dict=cfg.get("pretrained_config", None)
    )

    interpretation_config = set_up_interpretation_config(
        input_type=input_info_object.input_type,
        interpretation_config_dict=cfg.get("interpretation_config", None),
    )

    tensor_broker_config = set_up_tensor_broker_config(
        tensor_broker_config=cfg.get("tensor_broker_config", {})
    )

    input_config = schemas.InputConfig(
        input_info=input_info_object,
        input_type_info=input_type_info_object,
        pretrained_config=pretrained_config,
        model_config=model_config,
        interpretation_config=interpretation_config,
        tensor_broker_config=tensor_broker_config,
    )

    return input_config


def get_inputs_schema_map() -> dict[
    str,
    type[schemas.OmicsInputDataConfig]
    | type[schemas.TabularInputDataConfig]
    | type[schemas.SequenceInputDataConfig]
    | type[schemas.ByteInputDataConfig],
]:
    mapping = {
        "omics": schemas.OmicsInputDataConfig,
        "tabular": schemas.TabularInputDataConfig,
        "sequence": schemas.SequenceInputDataConfig,
        "bytes": schemas.ByteInputDataConfig,
        "image": schemas.ImageInputDataConfig,
        "array": schemas.ArrayInputDataConfig,
    }

    return mapping


def _validate_input_type_info_object(
    input_type_info_object: "al_input_types",
    input_source: str,
) -> None:
    ito = input_type_info_object
    match ito:
        case schemas.TabularInputDataConfig():
            con = ito.input_con_columns
            cat = ito.input_cat_columns
            common = set(con).intersection(set(cat))
            if common:
                raise ValueError(
                    f"Found columns passed in both continuous and categorical inputs: "
                    f"{common}. In input source: {input_source}."
                    f"Please make sure that each column is only in one of the two, "
                    f"or create differently named copies of the columns."
                )


def set_up_input_feature_extractor_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: "al_input_types",
    model_init_kwargs_base: dict,
) -> schemas.al_feature_extractor_configs:
    input_type = input_info_object.input_type

    model_config_class = get_input_feature_extractor_config_class(input_type=input_type)

    model_type = model_init_kwargs_base.get("model_type")
    if not model_type:
        try:
            model_type = model_config_class.model_type
        except AttributeError as e:
            logger.error(
                "Not model type specified in model config and "
                "could not find default value for '%s'.",
                input_type,
            )
            raise e

        logger.info(
            "Input model type not specified in model configuration for input name "
            "'%s', attempting to grab default value.",
            input_info_object.input_name,
        )

    model_type_init_config = set_up_feature_extractor_init_config(
        input_info_object=input_info_object,
        input_type_info_object=input_type_info_object,
        model_init_kwargs=model_init_kwargs_base.get("model_init_config", {}),
        model_type=model_type,
    )

    common_kwargs = {
        "model_type": model_type,
        "model_init_config": model_type_init_config,
    }
    other_specific_kwargs = {
        k: v for k, v in model_init_kwargs_base.items() if k not in common_kwargs
    }
    model_config_kwargs = {**common_kwargs, **other_specific_kwargs}
    model_config = model_config_class(**model_config_kwargs)

    return model_config


def get_input_feature_extractor_config_class(
    input_type: str,
) -> schemas.al_feature_extractor_configs_classes:
    model_config_setup_map = get_input_feature_extractor_config_init_class_map()

    return model_config_setup_map[input_type]


def get_input_feature_extractor_config_init_class_map() -> dict[
    str, schemas.al_feature_extractor_configs_classes
]:
    mapping = {
        "tabular": TabularModelConfig,
        "omics": OmicsModelConfig,
        "sequence": SequenceModelConfig,
        "bytes": SequenceModelConfig,
        "image": ImageModelConfig,
        "array": ArrayModelConfig,
    }

    return mapping


def set_up_feature_extractor_init_config(
    input_info_object: schemas.InputDataConfig,
    input_type_info_object: "al_input_types",
    model_init_kwargs: dict,
    model_type: str,
) -> dict:
    if getattr(input_type_info_object, "pretrained_model", None):
        return {}

    not_from_eir = get_is_not_eir_model_condition(
        input_info_object=input_info_object,
        model_type=model_type,
    )
    if not_from_eir:
        return model_init_kwargs

    if not model_init_kwargs:
        model_init_kwargs = {}

    model_config_map = get_feature_extractor_config_type_init_callable_map()
    model_config_callable = model_config_map[model_type]

    model_config = model_config_callable(**model_init_kwargs)

    return model_config


def get_is_not_eir_model_condition(
    input_info_object: schemas.InputDataConfig, model_type: str
) -> bool:
    is_possibly_external = input_info_object.input_type in (
        "sequence",
        "bytes",
        "image",
    )
    is_unknown_model = model_type not in ("sequence-default", "cnn", "lcl")
    not_from_eir = is_possibly_external and is_unknown_model
    return not_from_eir


def get_feature_extractor_config_type_init_callable_map() -> dict[str, type]:
    omics_mapping = get_omics_config_dataclass_mapping()
    array_mapping = get_array_config_dataclass_mapping()
    other_mapping = {
        "tabular": SimpleTabularModelConfig,
        "sequence-default": BasicTransformerFeatureExtractorModelConfig,
    }

    mapping = omics_mapping | array_mapping | other_mapping
    return mapping


def set_up_pretrained_config(
    pretrained_config_dict: None | dict[str, Any],
) -> None | schemas.BasicPretrainedConfig:
    if pretrained_config_dict is None:
        return None

    config_class = get_pretrained_config_class()
    if config_class is None:
        return None

    config_object = config_class(**pretrained_config_dict)

    return config_object


def get_pretrained_config_class() -> type[schemas.BasicPretrainedConfig]:
    return schemas.BasicPretrainedConfig


def set_up_interpretation_config(
    input_type: str, interpretation_config_dict: None | dict[str, Any]
) -> None | schemas.BasicInterpretationConfig:
    config_class = get_interpretation_config_class(input_type=input_type)
    if config_class is None:
        return None

    if interpretation_config_dict is None:
        interpretation_config_dict = {}

    config_object = config_class(**interpretation_config_dict)

    return config_object


def get_interpretation_config_class(
    input_type: str,
) -> None | type[schemas.BasicInterpretationConfig]:
    mapping = get_interpretation_config_schema_map()

    return mapping.get(input_type, None)


def get_interpretation_config_schema_map() -> dict[
    str, type[schemas.BasicInterpretationConfig]
]:
    mapping = {
        "sequence": schemas.BasicInterpretationConfig,
        "image": schemas.BasicInterpretationConfig,
    }

    return mapping
