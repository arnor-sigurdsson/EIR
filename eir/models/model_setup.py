import math
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
import reprlib
import typing
from typing import Union, Dict, Any, Sequence, Set, Type, Tuple, Literal, TYPE_CHECKING

import timm
import torch
from aislib.misc_utils import get_logger
from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    load_serialized_train_experiment,
)
from eir.models import al_fusion_models
from eir.models.fusion import fusion_linear, fusion_mgmoe, fusion_default
from eir.models.image.image_models import ImageWrapperModel, ImageModelConfig
from eir.models.models_base import get_output_dimensions_for_input
from eir.models.omics.omics_models import (
    al_omics_model_configs,
    get_model_class,
    get_omics_model_init_kwargs,
)
from eir.models.sequence.transformer_models import (
    SequenceModelConfig,
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
    SimpleTabularModelConfig,
)
from eir.setup import schemas
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoConfig,
)

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

    fusion_class, fusion_kwargs = get_fusion_model_class_and_kwargs_from_configs(
        global_config=global_config,
        predictor_config=predictor_config,
        num_outputs_per_target=num_outputs_per_target,
        inputs_as_dicts=inputs_as_dict,
    )

    if global_config.pretrained_checkpoint:
        logger.info(
            "Loading pretrained checkpoint from '%s'.",
            global_config.pretrained_checkpoint,
        )
        loaded_fusion_model = load_model(
            model_path=Path(global_config.pretrained_checkpoint),
            model_class=fusion_class,
            model_init_kwargs=fusion_kwargs,
            device=global_config.device,
            test_mode=False,
        )
        return loaded_fusion_model

    modules_to_fuse = overload_fusion_model_feature_extractors_with_pretrained(
        modules_to_fuse=fusion_kwargs["modules_to_fuse"], inputs_as_dict=inputs_as_dict
    )
    fusion_kwargs["modules_to_fuse"] = modules_to_fuse

    fusion_model = fusion_class(**fusion_kwargs)
    fusion_model = fusion_model.to(device=global_config.device)

    if global_config.multi_gpu:
        fusion_model = GetAttrDelegatedDataParallel(module=fusion_model)

    return fusion_model


def get_modules_to_fuse_from_inputs(
    inputs_as_dict: "al_input_objects_as_dict", device: str
) -> nn.ModuleDict:
    models = nn.ModuleDict()

    for input_name, inputs_object in inputs_as_dict.items():
        input_type = inputs_object.input_config.input_info.input_type
        input_type_info = inputs_object.input_config.input_type_info
        model_config = inputs_object.input_config.model_config

        if input_type == "omics":
            cur_omics_model = get_omics_model_from_model_config(
                model_type=model_config.model_type,
                model_init_config=model_config.model_init_config,
                data_dimensions=inputs_object.data_dimensions,
            )

            models[input_name] = cur_omics_model

        elif input_type == "tabular":

            transformers = inputs_object.labels.label_transformers
            cat_columns = input_type_info.input_cat_columns
            con_columns = input_type_info.input_con_columns

            unique_tabular_values = get_unique_values_from_transformers(
                transformers=transformers,
                keys_to_use=cat_columns,
            )

            tabular_model = get_tabular_model(
                model_init_config=model_config.model_init_config,
                cat_columns=cat_columns,
                con_columns=con_columns,
                device=device,
                unique_label_values=unique_tabular_values,
            )
            models[input_name] = tabular_model

        elif input_type == "sequence":

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                sequence_model_config=inputs_object.input_config.model_config,
                num_tokens=num_tokens,
                max_length=inputs_object.computed_max_length,
                embedding_dim=model_config.embedding_dim,
                device=device,
            )
            models[input_name] = sequence_model

        elif input_type == "bytes":

            num_tokens = len(inputs_object.vocab)
            sequence_model = get_sequence_model(
                sequence_model_config=inputs_object.input_config.model_config,
                num_tokens=num_tokens,
                max_length=inputs_object.computed_max_length,
                embedding_dim=model_config.embedding_dim,
                device=device,
            )
            models[input_name] = sequence_model

        elif input_type == "image":
            image_model = get_image_model(
                model_config=model_config,
                input_channels=inputs_object.num_channels,
                device=device,
            )
            models[input_name] = image_model

    return models


def get_image_model(
    model_config: ImageModelConfig,
    input_channels: int,
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

    model = ImageWrapperModel(
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
        pretrained=sequence_model_config.pretrained_model,
        pretrained_frozen=sequence_model_config.freeze_pretrained_model,
        model_config=sequence_model_config.model_init_config,
        num_tokens=num_tokens,
        embedding_dim=embedding_dim,
        feature_extractor_max_length=feature_extractor_max_length,
        num_chunks=num_chunks,
        pool=sequence_model_config.pool,
    )

    sequence_model = TransformerWrapperModel(
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
    pool: Union[Literal["max"], Literal["avg"], None],
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
    data_dimensions: "DataDimensions",
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


def get_fusion_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    predictor_config: schemas.PredictorConfig,
    inputs_as_dict: "al_input_objects_as_dict",
    num_outputs_per_target: "al_num_outputs_per_target",
) -> Dict[str, Any]:

    kwargs = {}
    modules_to_fuse = get_modules_to_fuse_from_inputs(
        inputs_as_dict=inputs_as_dict, device=global_config.device
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


def overload_fusion_model_feature_extractors_with_pretrained(
    modules_to_fuse: nn.ModuleDict, inputs_as_dict: "al_input_objects_as_dict"
) -> nn.ModuleDict:

    """
    Note that `inputs_as_dict` here are coming from the current experiment, arguably
    it would be more robust / better to have them loaded from the pretrained experiment,
    but then we have to setup things from there such as hooks, valid_ids, train_ids,
    etc.

    For now we will enforce that the feature extractor architecture that is set-up
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
        return modules_to_fuse

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

        func = get_fusion_model_class_and_kwargs_from_configs
        fusion_model_class, fusion_model_kwargs = func(
            global_config=load_configs.global_config,
            predictor_config=load_configs.predictor_config,
            num_outputs_per_target=load_experiment.num_outputs_per_target,
            inputs_as_dicts=inputs_as_dict,
        )

        pretrained_name = pretrained_config.load_module_name
        loaded_and_renamed_fusion_model = load_model(
            model_path=load_model_path,
            model_class=fusion_model_class,
            model_init_kwargs=fusion_model_kwargs,
            device="cpu",
            test_mode=False,
            state_dict_key_rename=replace_pattern,
            state_dict_keys_to_keep=(pretrained_name,),
        )
        loaded_and_renamed_fusion_extractors = (
            loaded_and_renamed_fusion_model.modules_to_fuse
        )

        module_name_to_load = pretrained_config.load_module_name
        module_to_overload = loaded_and_renamed_fusion_extractors[input_name]

        logger.info(
            "Replacing '%s' in current model with '%s' from %s.",
            input_name,
            module_name_to_load,
            load_model_path,
        )

        modules_to_fuse[input_name] = module_to_overload

    return modules_to_fuse


def get_fusion_model_class_and_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    predictor_config: schemas.PredictorConfig,
    num_outputs_per_target: "al_num_outputs_per_target",
    inputs_as_dicts: "al_input_objects_as_dict",
) -> Tuple[Type[nn.Module], Dict[str, Any]]:

    fusion_model_class = get_fusion_class(fusion_model_type=predictor_config.model_type)

    fusion_model_kwargs = get_fusion_kwargs_from_configs(
        global_config=global_config,
        predictor_config=predictor_config,
        num_outputs_per_target=num_outputs_per_target,
        inputs_as_dict=inputs_as_dicts,
    )

    return fusion_model_class, fusion_model_kwargs


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
) -> Union[al_fusion_models, nn.Module]:

    model = model_class(**model_init_kwargs)

    model = _load_model_weights(
        model=model,
        model_state_dict_path=model_path,
        device=device,
        state_dict_keys_to_keep=state_dict_keys_to_keep,
        state_dict_key_rename=state_dict_key_rename,
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
) -> nn.Module:
    state_dict = torch.load(model_state_dict_path, map_location=device)

    if state_dict_keys_to_keep:
        no_keys_before = len(state_dict)
        state_dict = _filter_state_dict_keys(
            state_dict=state_dict, keys_to_keep=state_dict_keys_to_keep
        )
        logger.info(
            "Extracting %d/%d modules for feature extractors: '%s' from %s.",
            len(state_dict),
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
            state_dict = _replace_dict_key_names(
                dict_=state_dict, replace_pattern=replace_tuple
            )

    incompatible_keys = model.load_state_dict(state_dict=state_dict, strict=False)

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
