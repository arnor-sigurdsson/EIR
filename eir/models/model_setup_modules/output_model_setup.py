from typing import Type, Dict, TYPE_CHECKING

from eir.models.output.linear import LinearOutputModuleConfig, LinearOutputModule
from eir.models.output.mlp_residual import (
    ResidualMLPOutputModulelConfig,
    ResidualMLPOutputModule,
)
from eir.models.output.output_module_setup import (
    TabularOutputModuleConfig,
    SequenceOutputModuleConfig,
)
from eir.models.output.sequence.sequence_output_modules import SequenceOutputModule
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    al_num_outputs_per_target,
)

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo

al_output_module_init_configs = (
    ResidualMLPOutputModulelConfig
    | LinearOutputModuleConfig
    | SequenceOutputModuleConfig
)
al_output_module_classes = (
    Type[ResidualMLPOutputModule]
    | Type[LinearOutputModule]
    | Type[SequenceOutputModule]
)
al_output_modules = ResidualMLPOutputModule | LinearOutputModule | SequenceOutputModule


def get_sequence_output_module_from_model_config(
    output_object: ComputedSequenceOutputInfo,
    feature_dimensionalities_and_types: Dict[str, "FeatureExtractorInfo"],
    device: str,
) -> SequenceOutputModule:
    output_model_config = output_object.output_config.model_config
    output_module_type = output_model_config.model_type

    class_map = _get_sequence_output_module_type_class_map()
    cur_output_module_class = class_map[output_module_type]

    output_module = cur_output_module_class(
        output_object=output_object,
        output_name=output_object.output_config.output_info.output_name,
        feature_dimensionalities_and_types=feature_dimensionalities_and_types,
    )

    output_module = output_module.to(device=device)

    return output_module


def _get_sequence_output_module_type_class_map() -> dict[str, al_output_module_classes]:
    mapping = {
        "sequence": SequenceOutputModule,
    }

    return mapping


def get_tabular_output_module_from_model_config(
    output_model_config: TabularOutputModuleConfig,
    input_dimension: int,
    num_outputs_per_target: "al_num_outputs_per_target",
    device: str,
) -> al_output_modules:
    class_map = _get_supervised_output_module_type_class_map()

    output_module_type = output_model_config.model_type
    cur_output_module_class = class_map[output_module_type]

    output_module = cur_output_module_class(
        model_config=output_model_config.model_init_config,
        input_dimension=input_dimension,
        num_outputs_per_target=num_outputs_per_target,
    )

    output_module = output_module.to(device=device)

    return output_module


def _get_supervised_output_module_type_class_map() -> (
    dict[str, al_output_module_classes]
):
    mapping = {
        "mlp_residual": ResidualMLPOutputModule,
        "linear": LinearOutputModule,
    }

    return mapping
