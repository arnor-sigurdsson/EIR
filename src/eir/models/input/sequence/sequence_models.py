from typing import Protocol

from eir.models.input.sequence.transformer_models import (
    SequenceOutputTransformerFeatureExtractor,
    TransformerFeatureExtractor,
    TransformerWrapperModel,
)


class SequenceModelClassGetterFunction(Protocol):
    def __call__(
        self, model_type: str
    ) -> (
        type[TransformerFeatureExtractor]
        | type[TransformerWrapperModel]
        | type[SequenceOutputTransformerFeatureExtractor]
    ): ...


def get_sequence_model_class(
    model_type: str,
) -> (
    type[TransformerFeatureExtractor] | type[SequenceOutputTransformerFeatureExtractor]
):
    match model_type:
        case "sequence-default":
            return TransformerFeatureExtractor
        case "eir-input-sequence-from-linked-output-default":
            return SequenceOutputTransformerFeatureExtractor
        case _:
            raise ValueError("Invalid sequence model type.")
