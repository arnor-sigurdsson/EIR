from typing import Type

from torch import nn

from eir.models.sequence.transformer_models import (
    TransformerFeatureExtractor,
    TransformerWrapperModel,
    SequenceOutputTransformerFeatureExtractor,
)


def get_sequence_model_class(model_type: str) -> Type[nn.Module]:
    match model_type:
        case "sequence-default":
            return TransformerFeatureExtractor
        case "sequence-wrapper-default":
            return TransformerWrapperModel
        case "eir-input-sequence-from-linked-output-default":
            return SequenceOutputTransformerFeatureExtractor
        case _:
            raise ValueError("Invalid sequence model type.")
