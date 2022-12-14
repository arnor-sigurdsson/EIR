import math
import reprlib
import typing
from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Union,
    Callable,
    Dict,
    Any,
    Sequence,
    Set,
    Type,
    Tuple,
    Literal,
    Optional,
    TYPE_CHECKING,
)

import timm
import torch
from aislib.misc_utils import get_logger
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoConfig,
)

import eir.models.fusion.fusion
import eir.models.meta.meta
from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    load_serialized_train_experiment,
)
from eir.models import al_fusion_models
from eir.models.image.image_models import ImageWrapperModel, ImageModelConfig
from eir.models.models_base import (
    get_output_dimensions_for_input,
)
from eir.models.omics.omics_models import (
    al_omics_model_configs,
    get_model_class,
    get_omics_model_init_kwargs,
)
from eir.models.output.tabular_output import (
    TabularMLPResidualModelConfig,
    TabularOutputModule,
)
from eir.models.sequence.transformer_models import (
    SequenceModelConfig,
    BasicTransformerFeatureExtractorModelConfig,
    TransformerWrapperModel,
    get_embedding_dim_for_sequence_model,
    TransformerFeatureExtractor,
    PerceiverIOModelConfig,
    PerceiverIOFeatureExtractor,
)
from eir.models.tabular.tabular import (
    get_unique_values_from_transformers,
    SimpleTabularModel,
    SimpleTabularModelConfig,
)
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict, DataDimensions
from eir.setup.setup_utils import get_unsupported_hf_models
from eir.train_utils.distributed import maybe_make_model_distributed

if TYPE_CHECKING:
    from eir.setup.output_setup import (
        al_output_objects_as_dict,
        al_num_outputs_per_target,
    )

al_fusion_class_callable = Callable[[str], Type[nn.Module]]
al_data_dimensions = Dict[
    str,
    Union[
        DataDimensions,
        "OmicsDataDimensions",
        "SequenceDataDimensions",
    ],
]
al_model_registry = Dict[str, Callable[[str], Type[nn.Module]]]

logger = get_logger(name=__name__)


def get_default_meta_class(
    meta_model_type: str,
) -> Type[nn.Module]:
    if meta_model_type == "default":
        return eir.models.meta.meta.MetaModel
    raise ValueError(f"Unrecognized meta model type: {meta_model_type}.")


def get_model(
    global_config: schemas.GlobalConfig,
    inputs_as_dict: al_input_objects_as_dict,
    fusion_config: schemas.FusionConfig,
    outputs_as_dict: "al_output_objects_as_dict",
    model_registry_per_input_type: al_model_registry,
    model_registry_per_output_type: al_model_registry,
    meta_class_getter: al_fusion_class_callable = get_default_meta_class,
) -> Union[nn.Module, nn.DataParallel]:

    meta_class, meta_kwargs = get_meta_model_class_and_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        model_registry_per_input_type=model_registry_per_input_type,
        model_registry_per_output_type=model_registry_per_output_type,
        meta_class_getter=meta_class_getter,
    )

    if global_config.pretrained_checkpoint:
        logger.info(
            "Loading pretrained checkpoint from '%s'.",
            global_config.pretrained_checkpoint,
        )
        loaded_meta_model = load_model(
            model_path=Path(global_config.pretrained_checkpoint),
            model_class=meta_class,
            model_init_kwargs=meta_kwargs,
            device=global_config.device,
            test_mode=False,
            strict_shapes=global_config.strict_pretrained_loading,
        )
        return loaded_meta_model

    input_modules = overload_fusion_model_feature_extractors_with_pretrained(
        input_modules=meta_kwargs["input_modules"],
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        model_registry_per_input_type=model_registry_per_input_type,
        model_registry_per_output_type=model_registry_per_output_type,
        meta_class_getter=meta_class_getter,
    )
    meta_kwargs["input_modules"] = input_modules

    meta_model = meta_class(**meta_kwargs)
    meta_model = meta_model.to(device=global_config.device)

    meta_model = maybe_make_model_distributed(
        device=global_config.device, model=meta_model
    )

    return meta_model


def get_output_modules(
    outputs_as_dict: "al_output_objects_as_dict",
    input_dimension: int,
    device: str,
    model_registry_per_output_type: Optional[al_model_registry] = None,
    in_features_per_input: Optional[Dict[str, DataDimensions]] = None,
    out_features_per_feature_extractor: Optional[Dict[str, int]] = None,
) -> nn.ModuleDict:

    output_modules = nn.ModuleDict()

    if model_registry_per_output_type is None:
        model_registry_per_output_type = {}

    for output_name, output_object in outputs_as_dict.items():
        output_type = output_object.output_config.output_info.output_type
        output_model_config = output_object.output_config.model_config

        if output_type == "tabular":

            tabular_output_module = get_tabular_output_module_from_model_config(
                model_init_config=output_model_config.model_init_config,
                input_dimension=input_dimension,
                num_outputs_per_target=output_object.num_outputs_per_target,
                device=device,
            )
            output_modules[output_name] = tabular_output_module

        elif output_type in model_registry_per_output_type:

            output_type_class_registry = model_registry_per_output_type[output_type]
            output_model_type = output_model_config.model_type
            output_type_class = output_type_class_registry(model_type=output_model_type)

            custom_output_module = output_type_class(
                output_object=output_object,
                output_name=output_name,
                input_dimension=input_dimension,
                in_features_per_feature_extractor=in_features_per_input,
                out_features_per_feature_extractor=out_features_per_feature_extractor,
                device=device,
            )
            output_modules[output_name] = custom_output_module
        else:
            raise NotImplementedError()

    return output_modules


def get_tabular_output_module_from_model_config(
    model_init_config: TabularMLPResidualModelConfig,
    input_dimension: int,
    num_outputs_per_target: "al_num_outputs_per_target",
    device: str,
) -> TabularOutputModule:
    output_module = TabularOutputModule(
        model_config=model_init_config,
        input_dimension=input_dimension,
        num_outputs_per_target=num_outputs_per_target,
    )
    output_module = output_module.to(device=device)

    return output_module


def get_input_modules(
    inputs_as_dict: al_input_objects_as_dict,
    model_registry_per_input_type: Dict[str, Callable[[str], Type[nn.Module]]],
    device: str,
) -> nn.ModuleDict:
    input_modules = nn.ModuleDict()

    for input_name, inputs_object in inputs_as_dict.items():
        input_type = inputs_object.input_config.input_info.input_type
        input_type_info = inputs_object.input_config.input_type_info
        input_model_config = inputs_object.input_config.model_config

        if input_type == "omics":
            cur_omics_model = get_omics_model_from_model_config(
                model_type=input_model_config.model_type,
                model_init_config=input_model_config.model_init_config,
                data_dimensions=inputs_object.data_dimensions,
            )

            input_modules[input_name] = cur_omics_model

        elif input_type == "tabular":

            transformers = inputs_object.labels.label_transformers
            cat_columns = input_type_info.input_cat_columns
            con_columns = input_type_info.input_con_columns

            unique_tabular_values = get_unique_values_from_transformers(
                transformers=transformers,
                keys_to_use=cat_columns,
            )

            tabular_model = get_tabular_model(
                model_init_config=input_model_config.model_init_config,
                cat_columns=cat_columns,
                con_columns=con_columns,
                device=device,
                unique_label_values=unique_tabular_values,
            )
            input_modules[input_name] = tabular_model

        elif input_type == "sequence":

            sequence_model_registry = model_registry_per_input_type["sequence"]

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                sequence_model_config=inputs_object.input_config.model_config,
                model_registry_lookup=sequence_model_registry,
                num_tokens=num_tokens,
                max_length=inputs_object.computed_max_length,
                embedding_dim=input_model_config.embedding_dim,
                device=device,
            )
            input_modules[input_name] = sequence_model

        elif input_type == "bytes":

            sequence_model_registry = model_registry_per_input_type["sequence"]

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                sequence_model_config=inputs_object.input_config.model_config,
                num_tokens=num_tokens,
                model_registry_lookup=sequence_model_registry,
                max_length=inputs_object.computed_max_length,
                embedding_dim=input_model_config.embedding_dim,
                device=device,
            )
            input_modules[input_name] = sequence_model

        elif input_type == "image":
            image_model_registry = model_registry_per_input_type["image"]
            image_model = get_image_model(
                model_config=input_model_config,
                input_channels=inputs_object.num_channels,
                model_registry_lookup=image_model_registry,
                device=device,
            )
            input_modules[input_name] = image_model

    return input_modules


def get_image_model(
    model_config: ImageModelConfig,
    input_channels: int,
    model_registry_lookup: Callable[[str], Type[nn.Module]],
    device: str,
) -> ImageWrapperModel:

    if model_config.model_type in timm.list_models():
        feature_extractor = timm.create_model(
            model_name=model_config.model_type,
            pretrained=model_config.pretrained_model,
            num_classes=model_config.num_output_features,
            in_chans=input_channels,
        ).to(device=device)

    else:
        feature_extractor = _meta_get_image_model_from_scratch(
            model_type=model_config.model_type,
            model_init_config=model_config.model_init_config,
            num_output_features=model_config.num_output_features,
        ).to(device=device)

    wrapper_model_class = model_registry_lookup(model_type="image-wrapper-default")
    model = wrapper_model_class(
        feature_extractor=feature_extractor, model_config=model_config
    )

    if model_config.freeze_pretrained_model:
        for param in model.parameters():
            param.requires_grad = False

    return model


def _meta_get_image_model_from_scratch(
    model_type: str, model_init_config: Dict, num_output_features: Union[None, int]
) -> nn.Module:
    """
    A kind of ridiculous way to initialize modules from scratch that are found in timm,
    but could not find a better way at first glance given how timm is set up.
    """

    feature_extractor_model_config = copy(model_init_config)
    if num_output_features:
        feature_extractor_model_config["num_classes"] = num_output_features

    logger.info(
        "Model '%s' not found among pretrained/external image model names, assuming "
        "module will be initialized from scratch using %s for initalization.",
        model_type,
        model_init_config,
    )

    feature_extractor_class = getattr(timm.models, model_type)
    parent_module = getattr(
        timm.models, feature_extractor_class.__module__.split(".")[-1]
    )
    found_modules = {
        k: getattr(parent_module, v)
        for k, v in feature_extractor_model_config.items()
        if isinstance(v, str) and getattr(parent_module, v, None)
    }
    feature_extractor_model_config_with_meta = {
        **feature_extractor_model_config,
        **found_modules,
    }

    feature_extractor = feature_extractor_class(
        **feature_extractor_model_config_with_meta
    )

    return feature_extractor


@dataclass
class SequenceModelConfigurationPrimitives:
    pretrained: bool
    pretrained_frozen: bool
    num_tokens: int
    max_length: int
    embedding_dim: int


@dataclass
class SequenceModelObjectsForWrapperModel:
    feature_extractor: Union[
        TransformerFeatureExtractor, PerceiverIOFeatureExtractor, nn.Module
    ]
    embeddings: Union[None, nn.Module]
    embedding_dim: int
    external: bool
    known_out_features: Union[None, int]


def get_sequence_model(
    sequence_model_config: SequenceModelConfig,
    model_registry_lookup: Callable[[str], Type[nn.Module]],
    num_tokens: int,
    max_length: int,
    embedding_dim: int,
    device: str,
) -> TransformerWrapperModel:

    feature_extractor_max_length = max_length
    num_chunks = 1
    if sequence_model_config.window_size:
        logger.info(
            "Using sliding model for sequence input as window size was set to %d.",
            sequence_model_config.window_size,
        )
        feature_extractor_max_length = sequence_model_config.window_size
        num_chunks = math.ceil(max_length / sequence_model_config.window_size)

    objects_for_wrapper = _get_sequence_feature_extractor_objects_for_wrapper_model(
        model_type=sequence_model_config.model_type,
        model_registry_lookup=model_registry_lookup,
        pretrained=sequence_model_config.pretrained_model,
        pretrained_frozen=sequence_model_config.freeze_pretrained_model,
        model_config=sequence_model_config.model_init_config,
        num_tokens=num_tokens,
        embedding_dim=embedding_dim,
        feature_extractor_max_length=feature_extractor_max_length,
        num_chunks=num_chunks,
        pool=sequence_model_config.pool,
    )

    wrapper_model_class = model_registry_lookup(model_type="sequence-wrapper-default")
    sequence_model = wrapper_model_class(
        feature_extractor=objects_for_wrapper.feature_extractor,
        external_feature_extractor=objects_for_wrapper.external,
        model_config=sequence_model_config,
        embedding_dim=objects_for_wrapper.embedding_dim,
        num_tokens=num_tokens,
        max_length=max_length,
        embeddings=objects_for_wrapper.embeddings,
        device=device,
        pre_computed_num_out_features=objects_for_wrapper.known_out_features,
    ).to(device=device)

    return sequence_model


def get_default_model_registry_per_input_type() -> al_model_registry:
    mapping = {"sequence": _sequence_model_registry, "image": _image_model_registry}

    return mapping


def _sequence_model_registry(model_type: str) -> Type[nn.Module]:
    if model_type == "sequence-default":
        return TransformerFeatureExtractor
    elif model_type == "sequence-wrapper-default":
        return TransformerWrapperModel
    else:
        raise ValueError()


def _image_model_registry(model_type: str) -> Type[nn.Module]:
    if model_type == "image-wrapper-default":
        return ImageWrapperModel
    else:
        raise ValueError()


def _get_sequence_feature_extractor_objects_for_wrapper_model(
    model_type: str,
    model_registry_lookup: Callable[[str], Type[nn.Module]],
    pretrained: bool,
    pretrained_frozen: bool,
    model_config: Union[
        BasicTransformerFeatureExtractorModelConfig, PerceiverIOModelConfig, Dict
    ],
    num_tokens: int,
    embedding_dim: int,
    feature_extractor_max_length: int,
    num_chunks: int,
    pool: Union[Literal["max"], Literal["avg"], None],
) -> SequenceModelObjectsForWrapperModel:

    if "sequence-default" in model_type or model_type.startswith("eir-"):
        model_class = model_registry_lookup(model_type=model_type)
        objects_for_wrapper = _get_basic_sequence_feature_extractor_objects(
            model_config=model_config,
            num_tokens=num_tokens,
            feature_extractor_max_length=feature_extractor_max_length,
            embedding_dim=embedding_dim,
            feature_extractor_class=model_class,
        )
    elif model_type == "perceiver":
        objects_for_wrapper = _get_perceiver_sequence_feature_extractor_objects(
            model_config=model_config,
            max_length=feature_extractor_max_length,
            num_chunks=num_chunks,
        )
    elif pretrained:
        objects_for_wrapper = _get_pretrained_hf_sequence_feature_extractor_objects(
            model_name=model_type,
            frozen=pretrained_frozen,
            feature_extractor_max_length=feature_extractor_max_length,
            num_chunks=num_chunks,
            num_tokens=num_tokens,
            pool=pool,
        )
    else:
        objects_for_wrapper = _get_hf_sequence_feature_extractor_objects(
            model_name=model_type,
            model_config=model_config,
            feature_extractor_max_length=feature_extractor_max_length,
            num_chunks=num_chunks,
            pool=pool,
        )

    return objects_for_wrapper


def _get_manual_out_features_for_external_feature_extractor(
    input_length: int,
    embedding_dim: int,
    num_chunks: int,
    feature_extractor: nn.Module,
    pool: Union[Literal["max"], Literal["avg"], None],
) -> int:
    input_shape = _get_sequence_input_dim(
        input_length=input_length,
        embedding_dim=embedding_dim,
    )
    out_feature_shape = get_output_dimensions_for_input(
        module=feature_extractor,
        input_shape=input_shape,
        hf_model=True,
        pool=pool,
    )
    manual_out_features = out_feature_shape.numel() * num_chunks

    return manual_out_features


def _get_sequence_input_dim(
    input_length: int, embedding_dim: int
) -> Tuple[int, int, int]:
    return 1, input_length, embedding_dim


def _get_pretrained_hf_sequence_feature_extractor_objects(
    model_name: str,
    num_tokens: int,
    frozen: bool,
    feature_extractor_max_length: int,
    num_chunks: int,
    pool: Union[Literal["max"], Literal["avg"], None],
) -> SequenceModelObjectsForWrapperModel:

    pretrained_model = _get_hf_pretrained_model(model_name=model_name)
    pretrained_model.resize_token_embeddings(new_num_tokens=num_tokens)
    pretrained_model_embeddings = pretrained_model.get_input_embeddings()
    feature_extractor = pretrained_model

    if frozen:
        logger.info("Freezing weights and embeddings of model '%s'.", model_name)
        for param in feature_extractor.parameters():
            param.requires_grad = False
        for param in pretrained_model_embeddings.parameters():
            param.requires_grad = False

    pretrained_embedding_dim = _pretrained_hf_model_embedding_dim(
        embeddings=pretrained_model_embeddings
    )
    known_out_features = _get_manual_out_features_for_external_feature_extractor(
        input_length=feature_extractor_max_length,
        embedding_dim=pretrained_embedding_dim,
        num_chunks=num_chunks,
        feature_extractor=feature_extractor,
        pool=pool,
    )
    objects_for_wrapper = SequenceModelObjectsForWrapperModel(
        feature_extractor=pretrained_model,
        embeddings=pretrained_model_embeddings,
        embedding_dim=pretrained_embedding_dim,
        external=True,
        known_out_features=known_out_features,
    )

    return objects_for_wrapper


def _get_hf_pretrained_model(model_name: str) -> PreTrainedModel:
    pretrained_model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    logger.info(
        "Loaded external pre-trained model '%s'. Note that this means that "
        "many configurations that might be set in input_type_info and model_config"
        "(e.g. 'embedding_dim') have no effect, as the default settings for "
        "the pre-trained model are used.",
        model_name,
    )
    return pretrained_model


def _get_hf_sequence_feature_extractor_objects(
    model_name: str,
    model_config: Dict[str, Any],
    feature_extractor_max_length: int,
    num_chunks: int,
    pool: Union[Literal["max"], Literal["avg"], None],
) -> SequenceModelObjectsForWrapperModel:

    feature_extractor = _get_hf_model(model_name=model_name, model_config=model_config)
    pretrained_model_embeddings = feature_extractor.get_input_embeddings()

    pretrained_embedding_dim = _pretrained_hf_model_embedding_dim(
        embeddings=pretrained_model_embeddings
    )
    known_out_features = _get_manual_out_features_for_external_feature_extractor(
        input_length=feature_extractor_max_length,
        embedding_dim=pretrained_embedding_dim,
        num_chunks=num_chunks,
        feature_extractor=feature_extractor,
        pool=pool,
    )
    objects_for_wrapper = SequenceModelObjectsForWrapperModel(
        feature_extractor=feature_extractor,
        embeddings=None,
        embedding_dim=pretrained_embedding_dim,
        external=True,
        known_out_features=known_out_features,
    )

    return objects_for_wrapper


def _pretrained_hf_model_embedding_dim(embeddings: nn.Module) -> int:
    if hasattr(embeddings, "embedding_dim"):
        return embeddings.embedding_dim
    elif hasattr(embeddings, "dim"):
        return embeddings.dim
    elif hasattr(embeddings, "emb_layers"):
        return embeddings.emb_layers[0].embedding_dim

    raise ValueError("Could not find embedding dimension.")


def _get_hf_model(model_name: str, model_config: Dict[str, Any]) -> nn.Module:
    config = AutoConfig.for_model(model_type=model_name, **model_config)
    model = AutoModel.from_config(config=config)
    logger.info(
        "Set up external (not using pre-trained weights) model '%s'. "
        "With configuration %s. Note that setting up external models ignores values "
        "for fields 'embedding_dim' and 'max_length' from input_type_info "
        "configuration. To configure these models, set the relevant values in the "
        "model_config field of the input configuration.",
        model_name,
        config,
    )
    return model


def _get_basic_sequence_feature_extractor_objects(
    model_config: BasicTransformerFeatureExtractorModelConfig,
    num_tokens: int,
    feature_extractor_max_length: int,
    embedding_dim: int,
    feature_extractor_class: Callable = TransformerFeatureExtractor,
) -> SequenceModelObjectsForWrapperModel:

    parsed_embedding_dim = get_embedding_dim_for_sequence_model(
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,
        num_heads=model_config.num_heads,
    )

    feature_extractor = feature_extractor_class(
        model_config=model_config,
        num_tokens=num_tokens,
        max_length=feature_extractor_max_length,
        embedding_dim=parsed_embedding_dim,
    )

    objects_for_wrapper = SequenceModelObjectsForWrapperModel(
        feature_extractor=feature_extractor,
        embeddings=None,
        embedding_dim=parsed_embedding_dim,
        external=False,
        known_out_features=0,
    )

    return objects_for_wrapper


def _get_perceiver_sequence_feature_extractor_objects(
    model_config: PerceiverIOModelConfig,
    max_length: int,
    num_chunks: int,
):

    feature_extractor = PerceiverIOFeatureExtractor(
        model_config=model_config,
        max_length=max_length,
    )

    known_out_features = feature_extractor.num_out_features * num_chunks

    objects_for_wrapper = SequenceModelObjectsForWrapperModel(
        feature_extractor=feature_extractor,
        embeddings=None,
        embedding_dim=model_config.dim,
        external=False,
        known_out_features=known_out_features,
    )

    return objects_for_wrapper


def get_tabular_model(
    model_init_config: SimpleTabularModelConfig,
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    device: str,
    unique_label_values: Dict[str, Set[str]],
) -> SimpleTabularModel:

    tabular_model = SimpleTabularModel(
        model_init_config=model_init_config,
        cat_columns=cat_columns,
        con_columns=con_columns,
        unique_label_values_per_column=unique_label_values,
        device=device,
    )

    return tabular_model


def get_omics_model_from_model_config(
    model_init_config: al_omics_model_configs,
    data_dimensions: DataDimensions,
    model_type: str,
):

    omics_model_class = get_model_class(model_type=model_type)
    model_init_kwargs = get_omics_model_init_kwargs(
        model_type=model_type,
        model_config=model_init_config,
        data_dimensions=data_dimensions,
    )
    omics_model = omics_model_class(**model_init_kwargs)

    if model_type == "cnn":
        assert omics_model.data_size_after_conv >= 8

    return omics_model


def get_meta_model_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    model_registry_per_input_type: al_model_registry,
    model_registry_per_output_type: al_model_registry,
) -> Dict[str, Any]:

    kwargs = {}
    input_modules = get_input_modules(
        inputs_as_dict=inputs_as_dict,
        device=global_config.device,
        model_registry_per_input_type=model_registry_per_input_type,
    )
    kwargs["input_modules"] = input_modules

    out_feature_per_feature_extractor = _get_feature_extractors_output_dimensions(
        input_modules=input_modules
    )
    fusion_module = eir.models.fusion.fusion.get_fusion_module(
        model_type=fusion_config.model_type,
        model_config=fusion_config.model_config,
        modules_to_fuse=input_modules,
        out_feature_per_feature_extractor=out_feature_per_feature_extractor,
    )
    kwargs["fusion_module"] = fusion_module

    in_features_per_input = _get_feature_extractors_input_dimensions_per_axis(
        inputs_as_dict=inputs_as_dict, input_modules=input_modules
    )
    output_modules = get_output_modules(
        outputs_as_dict=outputs_as_dict,
        model_registry_per_output_type=model_registry_per_output_type,
        input_dimension=fusion_module.num_out_features,
        device=global_config.device,
        in_features_per_input=in_features_per_input,
        out_features_per_feature_extractor=out_feature_per_feature_extractor,
    )
    kwargs["output_modules"] = output_modules

    return kwargs


def _get_feature_extractors_output_dimensions(
    input_modules: nn.ModuleDict,
) -> Dict[str, int]:
    fusion_in_dims = {name: i.num_out_features for name, i in input_modules.items()}
    return fusion_in_dims


@dataclass
class SequenceDataDimensions(DataDimensions):
    @property
    def max_length(self) -> int:
        return self.height

    @property
    def embedding_dim(self) -> int:
        return self.width


@dataclass
class OmicsDataDimensions(DataDimensions):
    @property
    def num_snps(self) -> int:
        return self.width

    @property
    def one_hot_encoding_dim(self) -> int:
        return self.height


def _get_feature_extractors_input_dimensions_per_axis(
    inputs_as_dict: al_input_objects_as_dict,
    input_modules: nn.ModuleDict,
) -> al_data_dimensions:

    fusion_in_dims = {}

    for name, input_object in inputs_as_dict.items():
        input_type = input_object.input_config.input_info.input_type
        input_type_info = input_object.input_config.input_type_info
        input_model_config = input_object.input_config.model_config

        if input_type in ("sequence", "bytes"):
            fusion_in_dims[name] = SequenceDataDimensions(
                channels=1,
                height=input_object.computed_max_length,
                width=input_model_config.embedding_dim,
            )
        elif input_type == "image":
            fusion_in_dims[name] = DataDimensions(
                channels=input_type_info.num_channels,
                height=input_type_info.size[0],
                width=input_type_info.size[-1],
            )
        elif input_type == "tabular":
            fusion_in_dims[name] = DataDimensions(
                channels=1,
                height=1,
                width=input_modules[name].input_dim,
            )
        elif input_type == "omics":
            fusion_in_dims[name] = OmicsDataDimensions(
                **input_object.data_dimensions.__dict__
            )
        else:
            raise ValueError(f"Unknown input type {input_type}.")

    return fusion_in_dims


def _warn_about_unsupported_hf_model(model_name: str) -> None:
    unsupported_models = get_unsupported_hf_models()
    if model_name in unsupported_models.keys():
        reason = unsupported_models[model_name]
        logger.warning(
            "Model '%s' has not been tested for compatibility with EIR due to "
            "reason: '%s'. It is very likely that it will not work straight out of "
            "the box with EIR.",
            model_name,
            reason,
        )


def overload_fusion_model_feature_extractors_with_pretrained(
    input_modules: nn.ModuleDict,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    model_registry_per_input_type: al_model_registry,
    model_registry_per_output_type: al_model_registry,
    meta_class_getter: al_fusion_class_callable = get_default_meta_class,
) -> nn.ModuleDict:

    """
    Note that `inputs_as_dict` here are coming from the current experiment, arguably
    it would be more robust / better to have them loaded from the pretrained experiment,
    but then we have to setup things from there such as hooks, valid_ids, train_ids,
    etc.

    For now, we will enforce that the feature extractor architecture that is set-up
    and then uses pre-trained weights from a previous experiment must match that of
    the feature extractor that did the pre-training. Simply put, we must ensure
    that all input setup parameters that have to do with architecture match exactly
    between the (a) pretrained input config and (b) the input config loading the
    pretrained model.
    """

    any_pretrained = any(
        i.input_config.pretrained_config for i in inputs_as_dict.values()
    )
    if not any_pretrained:
        return input_modules

    input_configs = tuple(i.input_config for i in inputs_as_dict.values())
    replace_pattern = _build_all_replacements_tuples_for_loading_pretrained_module(
        input_configs=input_configs
    )
    for input_name, input_object in inputs_as_dict.items():
        input_config = input_object.input_config

        pretrained_config = input_config.pretrained_config
        if not pretrained_config:
            continue

        load_model_path = Path(pretrained_config.model_path)
        load_run_folder = get_run_folder_from_model_path(
            model_path=str(load_model_path)
        )
        load_experiment = load_serialized_train_experiment(run_folder=load_run_folder)
        load_configs = load_experiment.configs

        func = get_meta_model_class_and_kwargs_from_configs
        meta_model_class, meta_model_kwargs = func(
            global_config=load_configs.global_config,
            fusion_config=load_configs.fusion_config,
            inputs_as_dict=inputs_as_dict,
            outputs_as_dict=outputs_as_dict,
            model_registry_per_input_type=model_registry_per_input_type,
            model_registry_per_output_type=model_registry_per_output_type,
            meta_class_getter=meta_class_getter,
        )

        pretrained_name = pretrained_config.load_module_name
        loaded_and_renamed_meta_model = load_model(
            model_path=load_model_path,
            model_class=meta_model_class,
            model_init_kwargs=meta_model_kwargs,
            device="cpu",
            test_mode=False,
            state_dict_key_rename=replace_pattern,
            state_dict_keys_to_keep=(pretrained_name,),
        )
        loaded_and_renamed_fusion_extractors = (
            loaded_and_renamed_meta_model.input_modules
        )

        module_name_to_load = pretrained_config.load_module_name
        module_to_overload = loaded_and_renamed_fusion_extractors[input_name]

        logger.info(
            "Replacing '%s' in current model with '%s' from %s.",
            input_name,
            module_name_to_load,
            load_model_path,
        )

        input_modules[input_name] = module_to_overload

    return input_modules


def get_meta_model_class_and_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    model_registry_per_input_type: al_model_registry,
    model_registry_per_output_type: al_model_registry,
    meta_class_getter: Callable[[str], Type[nn.Module]] = get_default_meta_class,
) -> Tuple[Type[nn.Module], Dict[str, Any]]:

    meta_model_class = meta_class_getter(meta_model_type="default")

    meta_model_kwargs = get_meta_model_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        model_registry_per_input_type=model_registry_per_input_type,
        model_registry_per_output_type=model_registry_per_output_type,
    )

    return meta_model_class, meta_model_kwargs


def _build_all_replacements_tuples_for_loading_pretrained_module(
    input_configs: Sequence[schemas.InputConfig],
) -> Sequence[Tuple[str, str]]:

    replacement_patterns = []
    for input_config in input_configs:
        if input_config.pretrained_config:
            cur_replacement = _build_replace_tuple_when_loading_pretrained_module(
                load_module_name=input_config.pretrained_config.load_module_name,
                current_input_name=input_config.input_info.input_name,
            )
            if cur_replacement:
                replacement_patterns.append(cur_replacement)

    return replacement_patterns


def _build_replace_tuple_when_loading_pretrained_module(
    load_module_name: str, current_input_name: str
) -> Union[None, Tuple[str, str]]:

    if load_module_name == current_input_name:
        return None

    load_module_name_parsed = f"modules_to_fuse.{load_module_name}."
    current_input_name_parsed = f"modules_to_fuse.{current_input_name}."

    replace_pattern = (load_module_name_parsed, current_input_name_parsed)

    return replace_pattern


def load_model(
    model_path: Path,
    model_class: Type[nn.Module],
    model_init_kwargs: Dict,
    device: str,
    test_mode: bool,
    state_dict_keys_to_keep: Union[None, Sequence[str]] = None,
    state_dict_key_rename: Union[None, Sequence[Tuple[str, str]]] = None,
    strict_shapes: bool = True,
) -> Union[al_fusion_models, nn.Module]:

    model = model_class(**model_init_kwargs)

    model = _load_model_weights(
        model=model,
        model_state_dict_path=model_path,
        device=device,
        state_dict_keys_to_keep=state_dict_keys_to_keep,
        state_dict_key_rename=state_dict_key_rename,
        strict_shapes=strict_shapes,
    )

    if test_mode:
        model.eval()

    return model


def _load_model_weights(
    model: nn.Module,
    model_state_dict_path: Path,
    device: str,
    state_dict_keys_to_keep: Union[None, Sequence[str]] = None,
    state_dict_key_rename: Union[None, Sequence[Tuple[str, str]]] = None,
    strict_shapes: bool = True,
) -> nn.Module:
    loaded_weights_state_dict = torch.load(model_state_dict_path, map_location=device)

    if state_dict_keys_to_keep:
        no_keys_before = len(loaded_weights_state_dict)
        loaded_weights_state_dict = _filter_state_dict_keys(
            state_dict=loaded_weights_state_dict, keys_to_keep=state_dict_keys_to_keep
        )
        logger.info(
            "Extracting %d/%d modules for feature extractors: '%s' from %s.",
            len(loaded_weights_state_dict),
            no_keys_before,
            state_dict_keys_to_keep,
            model_state_dict_path,
        )

    if state_dict_key_rename:
        for replace_tuple in state_dict_key_rename:
            logger.debug(
                "Renaming '%s' in pretrained model to '%s' in current model.",
                replace_tuple[0],
                replace_tuple[1],
            )
            loaded_weights_state_dict = _replace_dict_key_names(
                dict_=loaded_weights_state_dict, replace_pattern=replace_tuple
            )

    if not strict_shapes:
        model_state_dict = model.state_dict()
        loaded_weights_state_dict = _filter_incompatible_parameter_shapes_for_loading(
            source_state_dict=model_state_dict,
            destination_state_dict=loaded_weights_state_dict,
        )

    incompatible_keys = model.load_state_dict(
        state_dict=loaded_weights_state_dict, strict=False
    )

    no_missing = len(incompatible_keys.missing_keys)
    no_unexpected = len(incompatible_keys.unexpected_keys)
    no_incompatible_keys = no_missing + no_unexpected
    if no_incompatible_keys > 0:
        repr_object = reprlib.Repr()
        repr_object.maxother = 256
        repr_object.maxstring = 256
        logger.info(
            "Encountered incompatible modules when loading model from '%s'.\n"
            "Missing keys: \n%s\n"
            "Unexpected keys: \n%s\n"
            "This is expected if you are loading select modules from a saved model, "
            "which means you can ignore this message. If you are loading a pre-trained"
            " model as-is, then this is most likely an error and something unexpected"
            "has changed between the pre-training and setting up the current model"
            "from the pre-trained one.",
            model_state_dict_path,
            repr_object.repr(incompatible_keys.missing_keys),
            repr_object.repr(incompatible_keys.unexpected_keys),
        )

    model = model.to(device=device)

    return model


def _replace_dict_key_names(
    dict_: Dict[str, Any], replace_pattern: Tuple[str, str]
) -> OrderedDict:
    renamed_dict = OrderedDict()

    for key, value in dict_.items():
        new_key = key.replace(*replace_pattern)
        renamed_dict[new_key] = value

    return renamed_dict


def _filter_state_dict_keys(
    state_dict: typing.OrderedDict[str, torch.nn.Parameter], keys_to_keep: Sequence[str]
) -> typing.OrderedDict[str, torch.nn.Parameter]:

    filtered_state_dict = OrderedDict()

    for module_name, module_parameter in state_dict.items():
        if any(key in module_name for key in keys_to_keep):
            filtered_state_dict[module_name] = module_parameter

    return filtered_state_dict


def _filter_incompatible_parameter_shapes_for_loading(
    source_state_dict: Dict[str, Any], destination_state_dict: Dict[str, Any]
) -> Dict[str, Any]:

    destination_state_dict = deepcopy(destination_state_dict)

    for k in destination_state_dict:
        if k in source_state_dict:
            if destination_state_dict[k].shape != source_state_dict[k].shape:
                logger.info(
                    f"Skipping loading of parameter: {k} "
                    f"due to incompatible shapes. "
                    f"Source shape: {source_state_dict[k].shape}. "
                    f"Destination shape: {destination_state_dict[k].shape}."
                )
                destination_state_dict[k] = source_state_dict[k]

    return destination_state_dict
