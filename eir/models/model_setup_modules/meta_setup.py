from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, Type, Union, cast

from torch import nn

from eir.models.fusion import fusion
from eir.models.input.array.array_models import CNNModelConfig
from eir.models.input.image.image_models import get_image_model_class
from eir.models.input.sequence.sequence_models import get_sequence_model_class
from eir.models.input.tabular.tabular import get_unique_values_from_transformers
from eir.models.meta import meta
from eir.models.meta.meta_utils import (
    al_fusion_modules,
    al_input_modules,
    al_output_modules,
)
from eir.models.model_setup_modules.input_model_setup.input_model_setup_array import (
    get_array_input_feature_extractor,
    get_array_model,
)
from eir.models.model_setup_modules.input_model_setup.input_model_setup_image import (
    get_image_model,
)
from eir.models.model_setup_modules.input_model_setup.input_model_setup_omics import (
    get_omics_model_from_model_config,
)
from eir.models.model_setup_modules.input_model_setup.input_model_setup_sequence import (  # noqa
    get_sequence_model,
)
from eir.models.model_setup_modules.input_model_setup.input_model_setup_tabular import (
    SimpleTabularModel,
    get_tabular_model,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_array import (  # noqa
    get_array_or_image_output_module_from_model_config,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_sequence import (  # noqa
    get_sequence_output_module_from_model_config,
)
from eir.models.model_setup_modules.output_model_setup_modules.output_model_setup_tabular import (  # noqa
    get_tabular_output_module_from_model_config,
)
from eir.models.output.sequence.sequence_output_modules import (
    SequenceOutputModuleConfig,
)
from eir.models.tensor_broker.tensor_broker import get_tensor_broker
from eir.predict_modules.predict_tabular_input_setup import (
    ComputedPredictTabularInputInfo,
)
from eir.serve_modules.serve_schemas import ComputedServeTabularInputInfo
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.common import DataDimensions
from eir.setup.input_setup_modules.setup_array import ComputedArrayInputInfo
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import ComputedImageInputInfo
from eir.setup.input_setup_modules.setup_omics import ComputedOmicsInputInfo
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.setup.input_setup_modules.setup_tabular import ComputedTabularInputInfo
from eir.setup.output_setup import al_output_objects_as_dict
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)
from eir.setup.schemas import ImageInputDataConfig, TabularInputDataConfig
from eir.train_utils.distributed import AttrDelegatedDistributedDataParallel
from eir.train_utils.optim import AttrDelegatedSWAWrapper

al_meta_model = Union[
    meta.MetaModel, AttrDelegatedDistributedDataParallel, AttrDelegatedSWAWrapper
]


al_data_dimensions = Dict[
    str,
    Union[
        DataDimensions,
        "OmicsDataDimensions",
        "SequenceDataDimensions",
    ],
]


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


def get_default_meta_class(
    meta_model_type: str,
) -> Type[al_meta_model]:
    if meta_model_type == "default":
        return meta.MetaModel
    raise ValueError(f"Unrecognized meta model type: {meta_model_type}.")


class MetaClassGetterCallable(Protocol):
    def __call__(
        self,
        meta_model_type: str,
    ) -> Type[al_meta_model]: ...


def get_meta_model_class_and_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: MetaClassGetterCallable = get_default_meta_class,
    strict: bool = True,
) -> Tuple[Type[al_meta_model], Dict[str, Any]]:
    meta_model_class = meta_class_getter(meta_model_type="default")

    meta_model_kwargs = get_meta_model_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        strict=strict,
    )

    return meta_model_class, meta_model_kwargs


def get_meta_model_kwargs_from_configs(
    global_config: schemas.GlobalConfig,
    fusion_config: schemas.FusionConfig,
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    strict: bool = True,
) -> Dict[str, Any]:

    kwargs: dict[str, Any] = {}
    input_modules = get_input_modules(
        inputs_as_dict=inputs_as_dict,
        device=global_config.device,
    )
    kwargs["input_modules"] = input_modules

    out_feature_per_feature_extractor = _get_feature_extractors_num_output_features(
        input_modules=input_modules
    )
    output_types = _get_output_types(outputs_as_dict=outputs_as_dict)
    diffusion_targets = _extract_diffusion_targets(outputs_as_dict=outputs_as_dict)

    fusion_modules = fusion.get_fusion_modules(
        fusion_model_type=fusion_config.model_type,
        model_config=fusion_config.model_config,
        modules_to_fuse=input_modules,
        out_feature_per_feature_extractor=out_feature_per_feature_extractor,
        output_types=output_types,
        any_diffusion=diffusion_targets != {},
        strict=strict,
    )
    kwargs["fusion_modules"] = fusion_modules

    computed_out_dimension = _get_maybe_computed_out_dims(fusion_modules=fusion_modules)
    feature_dims_and_types = get_all_feature_extractor_dimensions_and_types(
        inputs_as_dict=inputs_as_dict,
        input_modules=input_modules,
    )
    output_modules, output_types = get_output_modules(
        outputs_as_dict=outputs_as_dict,
        device=global_config.device,
        computed_out_dimensions=computed_out_dimension,
        feature_dimensions_and_types=feature_dims_and_types,
        fusion_model_type=fusion_config.model_type,
    )
    fusion_to_output_mapping = _match_fusion_outputs_to_output_types(
        output_types=output_types,
        diffusion_targets=diffusion_targets,
    )
    kwargs["output_modules"] = output_modules
    kwargs["fusion_to_output_mapping"] = fusion_to_output_mapping

    input_configs = [i.input_config for i in inputs_as_dict.values()]
    output_configs = [i.output_config for i in outputs_as_dict.values()]
    tensor_broker = get_tensor_broker(
        input_objects=inputs_as_dict,
        output_objects=outputs_as_dict,
        input_modules=input_modules,
        fusion_modules=fusion_modules,
        output_modules=output_modules,
        fusion_to_output_mapping=fusion_to_output_mapping,
        input_configs=input_configs,
        fusion_configs=[fusion_config],
        output_configs=output_configs,
        device=global_config.device,
    )
    kwargs["tensor_broker"] = tensor_broker

    return kwargs


def _get_output_types(
    outputs_as_dict: "al_output_objects_as_dict",
) -> dict[str, Literal["tabular", "sequence", "array"]]:
    outputs_to_types_mapping: dict[str, Literal["tabular", "sequence", "array"]] = {}

    for output_name, output_object in outputs_as_dict.items():
        output_type = output_object.output_config.output_info.output_type
        outputs_to_types_mapping[output_name] = output_type

    return outputs_to_types_mapping


def _extract_diffusion_targets(
    outputs_as_dict: "al_output_objects_as_dict",
) -> dict[str, bool]:
    diffusion_targets = {}
    for name, output in outputs_as_dict.items():
        match output:
            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                if output.diffusion_config is not None:
                    diffusion_targets[name] = True

    return diffusion_targets


def _get_maybe_computed_out_dims(fusion_modules: al_fusion_modules) -> Optional[int]:
    if "computed" in fusion_modules:
        return fusion_modules["computed"].num_out_features

    return None


def _match_fusion_outputs_to_output_types(
    output_types: dict[str, Literal["tabular", "sequence", "array"]],
    diffusion_targets: dict[str, bool],
) -> dict[str, Literal["computed", "pass-through"]]:
    output_name_to_fusion_output_type: dict[
        str, Literal["computed", "pass-through"]
    ] = {}

    for output_name, output_type in output_types.items():
        match output_type:
            case "tabular":
                output_name_to_fusion_output_type[output_name] = "computed"
            case "sequence":
                output_name_to_fusion_output_type[output_name] = "pass-through"
            case "array" | "image":
                if output_name in diffusion_targets:
                    output_name_to_fusion_output_type[output_name] = "pass-through"
                else:
                    output_name_to_fusion_output_type[output_name] = "computed"
            case _:
                raise ValueError(f"Unknown output type '{output_type}'.")

    return output_name_to_fusion_output_type


@dataclass()
class FeatureExtractorInfo:
    input_dimension: Union[
        DataDimensions,
        "OmicsDataDimensions",
        "SequenceDataDimensions",
    ]
    output_dimension: int
    input_type: str
    output_shape: Optional[Tuple[int, ...]] = None
    extras: dict = field(default_factory=dict)


def get_all_feature_extractor_dimensions_and_types(
    inputs_as_dict: al_input_objects_as_dict,
    input_modules: al_input_modules,
) -> Dict[str, FeatureExtractorInfo]:
    input_dimensionality_and_types = {}

    out_feature_per_feature_extractor = _get_feature_extractors_num_output_features(
        input_modules=input_modules
    )
    in_features_per_input = _get_feature_extractors_input_dimensions_per_axis(
        inputs_as_dict=inputs_as_dict, input_modules=input_modules
    )

    output_shapes = _maybe_get_feature_extractor_out_shapes(input_modules=input_modules)

    for input_name, input_object in inputs_as_dict.items():

        cur_output_shape = output_shapes.get(input_name)

        extras = {}
        match input_object:
            case ComputedArrayInputInfo() | ComputedImageInputInfo():
                cur_config = input_object.input_config.model_config
                assert isinstance(
                    cur_config, (schemas.ArrayModelConfig, schemas.ImageModelConfig)
                )
                mic = cur_config.model_init_config
                if isinstance(mic, CNNModelConfig):
                    extras["down_every_n_blocks"] = mic.down_sample_every_n_blocks

        input_dimensionality_and_types[input_name] = FeatureExtractorInfo(
            input_dimension=in_features_per_input[input_name],
            output_dimension=out_feature_per_feature_extractor[input_name],
            input_type=input_object.input_config.input_info.input_type,
            output_shape=cur_output_shape,
            extras=extras,
        )

    return input_dimensionality_and_types


def get_input_modules(
    inputs_as_dict: al_input_objects_as_dict,
    device: str,
) -> al_input_modules:
    input_modules = nn.ModuleDict()

    for input_name, inputs_object in inputs_as_dict.items():
        input_model_config = inputs_object.input_config.model_config

        match inputs_object:
            case ComputedOmicsInputInfo():
                assert isinstance(input_model_config, schemas.OmicsModelConfig)
                cur_omics_model = get_omics_model_from_model_config(
                    model_type=input_model_config.model_type,
                    model_init_config=input_model_config.model_init_config,
                    data_dimensions=inputs_object.data_dimensions,
                )
                input_modules[input_name] = cur_omics_model

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                transformers = inputs_object.labels.label_transformers
                input_type_info = inputs_object.input_config.input_type_info
                assert isinstance(input_type_info, TabularInputDataConfig)
                cat_columns = list(input_type_info.input_cat_columns)
                con_columns = list(input_type_info.input_con_columns)

                unique_tabular_values = get_unique_values_from_transformers(
                    transformers=transformers,
                    keys_to_use=cat_columns,
                )

                assert isinstance(input_model_config, schemas.TabularModelConfig)
                tabular_model = get_tabular_model(
                    model_init_config=input_model_config.model_init_config,
                    cat_columns=cat_columns,
                    con_columns=con_columns,
                    device=device,
                    unique_label_values=unique_tabular_values,
                )
                input_modules[input_name] = tabular_model

            case ComputedSequenceInputInfo() | ComputedBytesInputInfo():
                assert isinstance(input_model_config, schemas.SequenceModelConfig)
                num_tokens = len(inputs_object.vocab)
                sequence_model = get_sequence_model(
                    sequence_model_config=input_model_config,
                    model_registry_lookup=get_sequence_model_class,
                    num_tokens=num_tokens,
                    max_length=inputs_object.computed_max_length,
                    embedding_dim=input_model_config.embedding_dim,
                    device=device,
                )
                input_modules[input_name] = sequence_model

            case ComputedImageInputInfo():
                assert isinstance(input_model_config, schemas.ImageModelConfig)
                image_model = get_image_model(
                    model_config=input_model_config,
                    model_registry_lookup=get_image_model_class,
                    data_dimensions=inputs_object.data_dimensions,
                    device=device,
                )
                input_modules[input_name] = image_model

            case ComputedArrayInputInfo():
                assert isinstance(input_model_config, schemas.ArrayModelConfig)
                array_feature_extractor = get_array_input_feature_extractor(
                    model_type=input_model_config.model_type,
                    model_init_config=input_model_config.model_init_config,
                    data_dimensions=inputs_object.data_dimensions,
                )
                cur_array_model = get_array_model(
                    array_feature_extractor=array_feature_extractor,
                    model_config=input_model_config,
                )
                input_modules[input_name] = cur_array_model

            case _:
                raise ValueError(f"Unrecognized input type for object {inputs_object}")

    return cast(al_input_modules, input_modules)


def get_output_modules(
    outputs_as_dict: "al_output_objects_as_dict",
    device: str,
    fusion_model_type: str,
    computed_out_dimensions: Optional[int] = None,
    feature_dimensions_and_types: Optional[Dict[str, FeatureExtractorInfo]] = None,
) -> Tuple[al_output_modules, Dict[str, Literal["tabular", "sequence", "array"]]]:
    output_modules: al_output_modules = cast(al_output_modules, nn.ModuleDict())
    output_types = {}

    for output_name, output_object in outputs_as_dict.items():
        output_type = output_object.output_config.output_info.output_type
        output_model_config = output_object.output_config.model_config
        output_types[output_name] = output_type

        match output_object:
            case ComputedTabularOutputInfo():
                assert computed_out_dimensions is not None
                assert isinstance(
                    output_model_config, schemas.TabularOutputModuleConfig
                )
                tabular_output_module = get_tabular_output_module_from_model_config(
                    output_model_config=output_model_config,
                    input_dimension=computed_out_dimensions,
                    num_outputs_per_target=output_object.num_outputs_per_target,
                    device=device,
                )
                output_modules[output_name] = tabular_output_module

            case ComputedSequenceOutputInfo():
                assert isinstance(output_model_config, SequenceOutputModuleConfig)
                feat_dims = feature_dimensions_and_types
                assert feat_dims is not None
                sequence_output_module = get_sequence_output_module_from_model_config(
                    output_object=output_object,
                    feature_dimensionalities_and_types=feat_dims,
                    device=device,
                )
                output_modules[output_name] = sequence_output_module

            case ComputedArrayOutputInfo() | ComputedImageOutputInfo():
                assert isinstance(output_model_config, schemas.ArrayOutputModuleConfig)
                feat_dims = feature_dimensions_and_types

                is_diffusion = output_object.diffusion_config is not None
                setup_func = get_array_or_image_output_module_from_model_config
                match is_diffusion:
                    case True:
                        assert feat_dims is not None
                        output_module = setup_func(
                            output_object=output_object,
                            input_dimension=computed_out_dimensions,
                            fusion_model_type=fusion_model_type,
                            feature_extractor_infos=feat_dims,
                            device=device,
                        )
                    case False:
                        assert computed_out_dimensions is not None
                        output_module = setup_func(
                            output_object=output_object,
                            input_dimension=computed_out_dimensions,
                            fusion_model_type=fusion_model_type,
                            feature_extractor_infos=feat_dims,
                            device=device,
                        )
                    case _:
                        raise ValueError()

                output_modules[output_name] = output_module
            case _:
                raise NotImplementedError(
                    "Only tabular, sequence and array outputs are supported"
                )

    return output_modules, output_types


def _get_feature_extractors_num_output_features(
    input_modules: al_input_modules,
) -> Dict[str, int]:
    fusion_in_dims = {name: i.num_out_features for name, i in input_modules.items()}
    return fusion_in_dims


def _maybe_get_feature_extractor_out_shapes(
    input_modules: al_input_modules,
) -> dict[str, tuple[int, ...]]:
    out_shapes = {
        name: i.output_shape
        for name, i in input_modules.items()
        if hasattr(i, "output_shape")
    }

    return out_shapes


def _get_feature_extractors_input_dimensions_per_axis(
    inputs_as_dict: al_input_objects_as_dict,
    input_modules: al_input_modules,
) -> al_data_dimensions:
    fusion_in_dims: al_data_dimensions = {}

    for name, input_object in inputs_as_dict.items():
        input_model_config = input_object.input_config.model_config

        match input_object:
            case ComputedSequenceInputInfo() | ComputedBytesInputInfo():
                assert isinstance(input_model_config, schemas.SequenceModelConfig)
                fusion_in_dims[name] = SequenceDataDimensions(
                    channels=1,
                    height=input_object.computed_max_length,
                    width=input_model_config.embedding_dim,
                )

            case ComputedImageInputInfo():
                input_type_info = input_object.input_config.input_type_info
                assert isinstance(input_type_info, ImageInputDataConfig)
                fusion_in_dims[name] = DataDimensions(
                    channels=input_object.num_channels,
                    height=input_type_info.size[0],
                    width=input_type_info.size[-1],
                )

            case (
                ComputedTabularInputInfo()
                | ComputedPredictTabularInputInfo()
                | ComputedServeTabularInputInfo()
            ):
                input_model = input_modules[name]
                assert isinstance(input_model, SimpleTabularModel)
                fusion_in_dims[name] = DataDimensions(
                    channels=1,
                    height=1,
                    width=input_model.input_dim,
                )

            case ComputedOmicsInputInfo():
                fusion_in_dims[name] = OmicsDataDimensions(
                    **input_object.data_dimensions.__dict__
                )

            case ComputedArrayInputInfo():
                fusion_in_dims[name] = input_object.data_dimensions

            case _:
                raise ValueError(f"Unrecognized input object {input_object}.")

    return fusion_in_dims
