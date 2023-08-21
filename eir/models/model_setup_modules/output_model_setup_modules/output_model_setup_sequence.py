from typing import TYPE_CHECKING, Dict, Type

import torch

from eir.models.output.sequence.sequence_output_modules import SequenceOutputModule
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)

al_sequence_module_classes = Type[SequenceOutputModule]

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import FeatureExtractorInfo


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

    torch_device = torch.device(device=device)
    output_module = output_module.to(device=torch_device)

    return output_module


def _get_sequence_output_module_type_class_map() -> (
    dict[str, al_sequence_module_classes]
):
    mapping = {
        "sequence": SequenceOutputModule,
    }

    return mapping
