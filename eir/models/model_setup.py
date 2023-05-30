from dataclasses import dataclass
from pathlib import Path
from typing import (
    Union,
    Callable,
    Dict,
    Any,
    Protocol,
    Sequence,
    Type,
    Tuple,
    Optional,
    Literal,
    TYPE_CHECKING,
)

import torch
from aislib.misc_utils import get_logger
from torch import nn

from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    load_serialized_train_experiment,
)
from eir.models.fusion import fusion
from eir.models.image.image_models import get_image_model_class
from eir.models.meta import meta
from eir.models.model_setup_modules.input_model_setup_array import (
    get_array_model,
    get_array_feature_extractor,
)
from eir.models.model_setup_modules.input_model_setup_image import get_image_model
from eir.models.model_setup_modules.input_model_setup_omics import (
    get_omics_model_from_model_config,
)
from eir.models.model_setup_modules.input_model_setup_sequence import (
    get_sequence_model,
)
from eir.models.model_setup_modules.input_model_setup_tabular import get_tabular_model
from eir.models.model_setup_modules.model_io import load_model
from eir.models.model_setup_modules.output_model_setup import (
    get_tabular_output_module_from_model_config,
    get_sequence_output_module_from_model_config,
)
from eir.models.output.sequence.sequence_output_modules import (
    overload_embeddings_with_pretrained,
)
from eir.models.sequence.sequence_models import get_sequence_model_class
from eir.models.tabular.tabular import (
    get_unique_values_from_transformers,
)
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.common import DataDimensions
from eir.train_utils.distributed import maybe_make_model_distributed

if TYPE_CHECKING:
    from eir.setup.output_setup import (
        al_output_objects_as_dict,
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
        return meta.MetaModel
    raise ValueError(f"Unrecognized meta model type: {meta_model_type}.")


def get_model(
    global_config: schemas.GlobalConfig,
    inputs_as_dict: al_input_objects_as_dict,
    fusion_config: schemas.FusionConfig,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: al_fusion_class_callable = get_default_meta_class,
) -> Union[nn.Module, nn.DataParallel, Callable]:
    meta_class, meta_kwargs = get_meta_model_class_and_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
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

        loaded_meta_model = overload_embeddings_with_pretrained(
            model=loaded_meta_model,
            inputs=inputs_as_dict,
            pretrained_checkpoint=global_config.pretrained_checkpoint,
        )

        return loaded_meta_model

    input_modules = overload_fusion_model_feature_extractors_with_pretrained(
        input_modules=meta_kwargs["input_modules"],
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        meta_class_getter=meta_class_getter,
    )
    meta_kwargs["input_modules"] = input_modules

    meta_model = meta_class(**meta_kwargs)
    meta_model = meta_model.to(device=global_config.device)

    meta_model = maybe_make_model_distributed(
        device=global_config.device, model=meta_model
    )

    if global_config.compile_model:
        meta_model = torch.compile(model=meta_model)

    return meta_model


class MetaClassGetterCallable(Protocol):
    def __call__(
        self,
        meta_model_type: str,
    ) -> Type[nn.Module]:
        ...


def get_meta_model_class_and_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: MetaClassGetterCallable = get_default_meta_class,
) -> Tuple[Type[nn.Module], Dict[str, Any]]:
    meta_model_class = meta_class_getter(meta_model_type="default")

    meta_model_kwargs = get_meta_model_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
    )

    return meta_model_class, meta_model_kwargs


def get_meta_model_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
) -> Dict[str, Any]:
    kwargs = {}
    input_modules = get_input_modules(
        inputs_as_dict=inputs_as_dict,
        device=global_config.device,
    )
    kwargs["input_modules"] = input_modules

    out_feature_per_feature_extractor = _get_feature_extractors_num_output_dimensions(
        input_modules=input_modules
    )
    output_types = _get_output_types(outputs_as_dict=outputs_as_dict)

    fusion_modules = fusion.get_fusion_modules(
        fusion_model_type=fusion_config.model_type,
        model_config=fusion_config.model_config,
        modules_to_fuse=input_modules,
        out_feature_per_feature_extractor=out_feature_per_feature_extractor,
        output_types=output_types,
    )
    kwargs["fusion_modules"] = fusion_modules

    computed_out_dimension = _get_maybe_computed_out_dims(fusion_modules=fusion_modules)
    feature_dims_and_types = get_all_feature_extractor_dimensions_and_types(
        inputs_as_dict=inputs_as_dict, input_modules=input_modules
    )
    output_modules, output_types = get_output_modules(
        outputs_as_dict=outputs_as_dict,
        device=global_config.device,
        computed_out_dimensions=computed_out_dimension,
        feature_dimensions_and_types=feature_dims_and_types,
    )
    fusion_to_output_mapping = _match_fusion_outputs_to_output_types(
        output_types=output_types
    )
    kwargs["output_modules"] = output_modules
    kwargs["fusion_to_output_mapping"] = fusion_to_output_mapping

    return kwargs


def _get_output_types(
    outputs_as_dict: "al_output_objects_as_dict",
) -> dict[str, Literal["tabular", "sequence"]]:
    outputs_to_types_mapping = {}

    for output_name, output_object in outputs_as_dict.items():
        output_type = output_object.output_config.output_info.output_type
        outputs_to_types_mapping[output_name] = output_type

    return outputs_to_types_mapping


def _get_maybe_computed_out_dims(fusion_modules: nn.ModuleDict) -> Optional[int]:
    if "computed" in fusion_modules:
        return fusion_modules["computed"].num_out_features

    return None


def _match_fusion_outputs_to_output_types(
    output_types: dict[str, Literal["tabular", "sequence"]]
) -> dict[str, str]:
    output_name_to_fusion_output_type = {}

    for output_name, output_type in output_types.items():
        match output_type:
            case "tabular":
                output_name_to_fusion_output_type[output_name] = "computed"
            case "sequence":
                output_name_to_fusion_output_type[output_name] = "pass-through"
            case _:
                raise ValueError(f"Unknown output type '{output_type}'.")

    return output_name_to_fusion_output_type


@dataclass()
class FeatureExtractorInfo:
    input_dimension: Union[
        DataDimensions, "OmicsDataDimensions", "SequenceDataDimensions"
    ]
    output_dimension: int
    input_type: str


def get_all_feature_extractor_dimensions_and_types(
    inputs_as_dict: al_input_objects_as_dict,
    input_modules: nn.ModuleDict,
) -> Dict[str, FeatureExtractorInfo]:
    input_dimensionalities_and_types = {}

    out_feature_per_feature_extractor = _get_feature_extractors_num_output_dimensions(
        input_modules=input_modules
    )
    in_features_per_input = _get_feature_extractors_input_dimensions_per_axis(
        inputs_as_dict=inputs_as_dict, input_modules=input_modules
    )

    for input_name, input_object in inputs_as_dict.items():
        input_dimensionalities_and_types[input_name] = FeatureExtractorInfo(
            input_dimension=in_features_per_input[input_name],
            output_dimension=out_feature_per_feature_extractor[input_name],
            input_type=input_object.input_config.input_info.input_type,
        )

    return input_dimensionalities_and_types


def get_input_modules(
    inputs_as_dict: al_input_objects_as_dict,
    device: str,
) -> nn.ModuleDict:
    input_modules = nn.ModuleDict()

    for input_name, inputs_object in inputs_as_dict.items():
        input_type = inputs_object.input_config.input_info.input_type
        input_type_info = inputs_object.input_config.input_type_info
        input_model_config = inputs_object.input_config.model_config

        match input_type:
            case "omics":
                cur_omics_model = get_omics_model_from_model_config(
                    model_type=input_model_config.model_type,
                    model_init_config=input_model_config.model_init_config,
                    data_dimensions=inputs_object.data_dimensions,
                )

                input_modules[input_name] = cur_omics_model

            case "tabular":
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

            case "sequence" | "bytes":
                num_tokens = len(inputs_object.vocab)
                sequence_model = get_sequence_model(
                    sequence_model_config=inputs_object.input_config.model_config,
                    model_registry_lookup=get_sequence_model_class,
                    num_tokens=num_tokens,
                    max_length=inputs_object.computed_max_length,
                    embedding_dim=input_model_config.embedding_dim,
                    device=device,
                )
                input_modules[input_name] = sequence_model

            case "image":
                image_model = get_image_model(
                    model_config=input_model_config,
                    input_channels=inputs_object.num_channels,
                    model_registry_lookup=get_image_model_class,
                    device=device,
                )
                input_modules[input_name] = image_model

            case "array":
                array_feature_extractor = get_array_feature_extractor(
                    model_type=input_model_config.model_type,
                    model_init_config=input_model_config.model_init_config,
                    data_dimensions=inputs_object.data_dimensions,
                )
                cur_array_model = get_array_model(
                    array_feature_extractor=array_feature_extractor,
                    model_config=input_model_config,
                )

                input_modules[input_name] = cur_array_model

    return input_modules


def get_output_modules(
    outputs_as_dict: "al_output_objects_as_dict",
    device: str,
    computed_out_dimensions: Optional[int] = None,
    feature_dimensions_and_types: Optional[Dict[str, FeatureExtractorInfo]] = None,
) -> Tuple[nn.ModuleDict, Dict[str, Literal["tabular", "sequence"]]]:
    output_modules = nn.ModuleDict()
    output_types = {}

    for output_name, output_object in outputs_as_dict.items():
        output_type = output_object.output_config.output_info.output_type
        output_model_config = output_object.output_config.model_config
        output_types[output_name] = output_type

        match output_type:
            case "tabular":
                assert computed_out_dimensions is not None
                tabular_output_module = get_tabular_output_module_from_model_config(
                    output_model_config=output_model_config,
                    input_dimension=computed_out_dimensions,
                    num_outputs_per_target=output_object.num_outputs_per_target,
                    device=device,
                )
                output_modules[output_name] = tabular_output_module

            case "sequence":
                feat_dims = feature_dimensions_and_types
                assert feat_dims is not None
                sequence_output_module = get_sequence_output_module_from_model_config(
                    output_object=output_object,
                    feature_dimensionalities_and_types=feat_dims,
                    device=device,
                )
                output_modules[output_name] = sequence_output_module

            case _:
                raise NotImplementedError(
                    "Only tabular and sequence outputs are supported"
                )

    return output_modules, output_types


def get_default_model_registry_per_input_type() -> al_model_registry:
    mapping = {
        "sequence": get_sequence_model_class,
        "image": get_image_model_class,
    }

    return mapping


def _get_feature_extractors_num_output_dimensions(
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

        match input_type:
            case "sequence" | "bytes":
                fusion_in_dims[name] = SequenceDataDimensions(
                    channels=1,
                    height=input_object.computed_max_length,
                    width=input_model_config.embedding_dim,
                )

            case "image":
                fusion_in_dims[name] = DataDimensions(
                    channels=input_object.num_channels,
                    height=input_type_info.size[0],
                    width=input_type_info.size[-1],
                )
            case "tabular":
                fusion_in_dims[name] = DataDimensions(
                    channels=1,
                    height=1,
                    width=input_modules[name].input_dim,
                )
            case "omics":
                fusion_in_dims[name] = OmicsDataDimensions(
                    **input_object.data_dimensions.__dict__
                )
            case "array":
                fusion_in_dims[name] = input_object.data_dimensions

            case _:
                raise ValueError(f"Unknown input type {input_type}.")

    return fusion_in_dims


def overload_fusion_model_feature_extractors_with_pretrained(
    input_modules: nn.ModuleDict,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: MetaClassGetterCallable = get_default_meta_class,
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
