import math
from copy import copy
from dataclasses import dataclass
from typing import Union, Dict, Any, Sequence, Set, Type, Tuple, TYPE_CHECKING

import timm
from aislib.misc_utils import get_logger
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoConfig,
)

from eir.models.fusion import fusion_linear, fusion_mgmoe, fusion_default
from eir.models.image.image_models import ImageWrapperModel, ImageModelConfig
from eir.models.models_base import get_output_dimensions_for_input
from eir.models.omics.omics_models import (
    al_omics_model_configs,
    get_model_class,
    get_omics_model_init_kwargs,
)
from eir.models.sequence.transformer_models import (
    TransformerWrapperModelConfig,
    BasicTransformerFeatureExtractorModelConfig,
    TransformerWrapperModel,
    get_embedding_dim_for_sequence_model,
    TransformerFeatureExtractor,
    PerceiverIOModelConfig,
    PerceiverIOFeatureExtractor,
    get_unsupported_hf_models,
)
from eir.models.tabular.tabular import (
    get_unique_values_from_transformers,
    SimpleTabularModel,
)
from eir.setup import schemas

if TYPE_CHECKING:
    from eir.setup.input_setup import al_input_objects_as_dict, DataDimensions
    from eir.train import al_num_outputs_per_target

logger = get_logger(name=__name__)


class GetAttrDelegatedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_model(
    inputs_as_dict: "al_input_objects_as_dict",
    predictor_config: schemas.PredictorConfig,
    global_config: schemas.GlobalConfig,
    num_outputs_per_target: "al_num_outputs_per_target",
) -> Union[nn.Module, nn.DataParallel]:

    fusion_class = get_fusion_class(fusion_model_type=predictor_config.model_type)
    fusion_kwargs = get_fusion_kwargs_from_cl_args(
        global_config=global_config,
        predictor_config=predictor_config,
        inputs=inputs_as_dict,
        num_outputs_per_target=num_outputs_per_target,
    )
    fusion_model = fusion_class(**fusion_kwargs)
    fusion_model = fusion_model.to(device=global_config.device)

    if global_config.multi_gpu:
        fusion_model = GetAttrDelegatedDataParallel(module=fusion_model)

    return fusion_model


def get_modules_to_fuse_from_inputs(
    inputs: "al_input_objects_as_dict", device: str
) -> nn.ModuleDict:
    models = nn.ModuleDict()

    for input_name, inputs_object in inputs.items():
        input_type_info = inputs_object.input_config.input_type_info

        if input_name.startswith("omics_"):
            cur_omics_model = get_omics_model_from_model_config(
                model_type=input_type_info.model_type,
                model_config=inputs_object.input_config.model_config,
                data_dimensions=inputs_object.data_dimensions,
            )

            models[input_name] = cur_omics_model

        elif input_name.startswith("tabular_"):

            transformers = inputs_object.labels.label_transformers
            cat_columns = input_type_info.extra_cat_columns
            con_columns = input_type_info.extra_con_columns

            unique_tabular_values = get_unique_values_from_transformers(
                transformers=transformers,
                keys_to_use=cat_columns,
            )

            tabular_model = get_tabular_model(
                cat_columns=cat_columns,
                con_columns=con_columns,
                device=device,
                unique_label_values=unique_tabular_values,
            )
            models[input_name] = tabular_model

        elif input_name.startswith("sequence_"):

            input_type_info = inputs_object.input_config.input_type_info
            sequence_wrapper_model_config = TransformerWrapperModelConfig(
                position=input_type_info.position,
                position_dropout=input_type_info.position_dropout,
                window_size=input_type_info.window_size,
            )

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                model_type=input_type_info.model_type,
                pretrained=input_type_info.pretrained_model,
                pretrained_frozen=input_type_info.freeze_pretrained_model,
                model_config=inputs_object.input_config.model_config,
                wrapper_model_config=sequence_wrapper_model_config,
                num_tokens=num_tokens,
                max_length=inputs_object.computed_max_length,
                embedding_dim=input_type_info.embedding_dim,
                device=device,
            )
            models[input_name] = sequence_model

        elif input_name.startswith("bytes_"):

            input_type_info = inputs_object.input_config.input_type_info
            sequence_wrapper_model_config = TransformerWrapperModelConfig(
                position=input_type_info.position,
                position_dropout=input_type_info.position_dropout,
                window_size=input_type_info.window_size,
            )

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                model_type=input_type_info.model_type,
                pretrained=False,
                pretrained_frozen=False,
                model_config=inputs_object.input_config.model_config,
                wrapper_model_config=sequence_wrapper_model_config,
                num_tokens=num_tokens,
                max_length=inputs_object.computed_max_length,
                embedding_dim=input_type_info.embedding_dim,
                device=device,
            )
            models[input_name] = sequence_model

        elif input_name.startswith("image_"):
            image_model = get_image_model(
                model_type=input_type_info.model_type,
                pretrained=input_type_info.pretrained_model,
                frozen=input_type_info.freeze_pretrained_model,
                model_config=inputs_object.input_config.model_config,
                input_channels=inputs_object.num_channels,
                device=device,
            )
            models[input_name] = image_model

    return models


def get_image_model(
    model_type: str,
    pretrained: bool,
    frozen: bool,
    model_config: Dict,
    input_channels: int,
    device: str,
) -> ImageWrapperModel:

    wrapper_kwargs = {
        k: v for k, v in model_config.items() if k == "num_output_features"
    }
    wrapper_model_config = ImageModelConfig(**wrapper_kwargs)

    if model_type in timm.list_models():
        feature_extractor = timm.create_model(
            model_name=model_type,
            pretrained=pretrained,
            num_classes=wrapper_model_config.num_output_features,
            in_chans=input_channels,
        ).to(device=device)
    else:

        if "num_output_features" not in model_config:
            n_output_feats = wrapper_model_config.num_output_features
            model_config["num_output_features"] = n_output_feats

        feature_extractor = _meta_get_image_model_from_scratch(
            model_type=model_type, model_config=model_config
        ).to(device=device)

    model = ImageWrapperModel(
        feature_extractor=feature_extractor, model_config=wrapper_model_config
    )

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    return model


def _meta_get_image_model_from_scratch(
    model_type: str, model_config: Dict
) -> nn.Module:
    """
    A kind of ridiculous way to initialize modules from scratch that are found in timm,
    but could not find a better way at first glance given how timm is set up.
    """

    feature_extractor_model_config = copy(model_config)
    if "num_output_features" in feature_extractor_model_config:
        num_classes = feature_extractor_model_config.pop("num_output_features")
        feature_extractor_model_config["num_classes"] = num_classes

    logger.info(
        "Model '%s' not found among pretrained/external image model names, assuming "
        "module will be initialized from scratch using %s for initalization.",
        model_type,
        model_config,
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
    model_type: str,
    pretrained: bool,
    pretrained_frozen: bool,
    model_config: Union[BasicTransformerFeatureExtractorModelConfig, Dict],
    wrapper_model_config: TransformerWrapperModelConfig,
    num_tokens: int,
    max_length: int,
    embedding_dim: int,
    device: str,
) -> TransformerWrapperModel:

    feature_extractor_max_length = max_length
    num_chunks = 1
    if wrapper_model_config.window_size:
        logger.info(
            "Using sliding model for sequence input as window size was set to %d.",
            wrapper_model_config.window_size,
        )
        feature_extractor_max_length = wrapper_model_config.window_size
        num_chunks = math.ceil(max_length / wrapper_model_config.window_size)

    objects_for_wrapper = _get_sequence_feature_extractor_objects_for_wrapper_model(
        model_type=model_type,
        pretrained=pretrained,
        pretrained_frozen=pretrained_frozen,
        model_config=model_config,
        num_tokens=num_tokens,
        embedding_dim=embedding_dim,
        feature_extractor_max_length=feature_extractor_max_length,
        num_chunks=num_chunks,
    )

    sequence_model = TransformerWrapperModel(
        feature_extractor=objects_for_wrapper.feature_extractor,
        external_feature_extractor=objects_for_wrapper.external,
        model_config=wrapper_model_config,
        embedding_dim=objects_for_wrapper.embedding_dim,
        num_tokens=num_tokens,
        max_length=max_length,
        embeddings=objects_for_wrapper.embeddings,
        device=device,
        pre_computed_num_out_features=objects_for_wrapper.known_out_features,
    ).to(device=device)

    return sequence_model


def _get_sequence_feature_extractor_objects_for_wrapper_model(
    model_type: str,
    pretrained: bool,
    pretrained_frozen: bool,
    model_config: Union[
        BasicTransformerFeatureExtractorModelConfig, PerceiverIOModelConfig, Dict
    ],
    num_tokens: int,
    embedding_dim: int,
    feature_extractor_max_length: int,
    num_chunks: int,
) -> SequenceModelObjectsForWrapperModel:

    if model_type == "sequence-default":
        objects_for_wrapper = _get_basic_sequence_feature_extractor_objects(
            model_config=model_config,
            num_tokens=num_tokens,
            feature_extractor_max_length=feature_extractor_max_length,
            embedding_dim=embedding_dim,
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
        )
    else:
        objects_for_wrapper = _get_hf_sequence_feature_extractor_objects(
            model_name=model_type,
            model_config=model_config,
            feature_extractor_max_length=feature_extractor_max_length,
            num_chunks=num_chunks,
        )

    return objects_for_wrapper


def _get_manual_out_features_for_external_feature_extractor(
    input_length: int, embedding_dim: int, num_chunks: int, feature_extractor: nn.Module
) -> int:
    input_shape = _get_sequence_input_dim(
        input_length=input_length,
        embedding_dim=embedding_dim,
    )
    out_feature_shape = get_output_dimensions_for_input(
        module=feature_extractor,
        input_shape=input_shape,
        hf_model=True,
    )
    manual_out_features = out_feature_shape.numel() * num_chunks

    return manual_out_features


def _get_sequence_input_dim(
    input_length: int, embedding_dim: int
) -> Tuple[int, int, int]:
    return 1, input_length, embedding_dim


def _get_pretrained_hf_sequence_feature_extractor_objects(
    model_name: str, frozen: bool, feature_extractor_max_length: int, num_chunks: int
) -> SequenceModelObjectsForWrapperModel:

    pretrained_model = _get_hf_pretrained_model(model_name=model_name)
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
) -> SequenceModelObjectsForWrapperModel:

    parsed_embedding_dim = get_embedding_dim_for_sequence_model(
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,
        num_heads=model_config.num_heads,
    )

    feature_extractor = TransformerFeatureExtractor(
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
        known_out_features=None,
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
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    device: str,
    unique_label_values: Dict[str, Set[str]],
) -> SimpleTabularModel:
    tabular_model = SimpleTabularModel(
        cat_columns=cat_columns,
        con_columns=con_columns,
        unique_label_values_per_column=unique_label_values,
        device=device,
    )

    return tabular_model


def get_omics_model_from_model_config(
    model_config: al_omics_model_configs,
    data_dimensions: "DataDimensions",
    model_type: str,
):

    omics_model_class = get_model_class(model_type=model_type)
    model_init_kwargs = get_omics_model_init_kwargs(
        model_type=model_type,
        model_config=model_config,
        data_dimensions=data_dimensions,
    )
    omics_model = omics_model_class(**model_init_kwargs)

    if model_type == "cnn":
        assert omics_model.data_size_after_conv >= 8

    return omics_model


def get_fusion_class(
    fusion_model_type: str,
) -> Type[nn.Module]:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "default":
        return fusion_default.FusionModel
    elif fusion_model_type == "linear":
        return fusion_linear.LinearFusionModel
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")


def get_fusion_kwargs_from_cl_args(
    global_config: schemas.GlobalConfig,
    predictor_config: schemas.PredictorConfig,
    inputs: "al_input_objects_as_dict",
    num_outputs_per_target: "al_num_outputs_per_target",
) -> Dict[str, Any]:

    kwargs = {}
    modules_to_fuse = get_modules_to_fuse_from_inputs(
        inputs=inputs, device=global_config.device
    )
    kwargs["modules_to_fuse"] = modules_to_fuse
    kwargs["num_outputs_per_target"] = num_outputs_per_target
    kwargs["model_config"] = predictor_config.model_config

    return kwargs


def _warn_abount_unsupported_hf_model(model_name: str) -> None:
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
