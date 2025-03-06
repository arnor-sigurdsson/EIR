from typing import Protocol

from eir.models.input.sequence.transformer_models import (
    MaskedTransformerFeatureExtractor,
    TransformerFeatureExtractor,
    TransformerWrapperModel,
)


class SequenceModelClassGetterFunction(Protocol):
    def __call__(
        self,
        model_type: str,
        force_masked: bool = False,
    ) -> (
        type[TransformerFeatureExtractor]
        | type[TransformerWrapperModel]
        | type[MaskedTransformerFeatureExtractor]
    ): ...


def get_sequence_model_class(
    model_type: str,
    force_masked: bool = False,
) -> type[TransformerFeatureExtractor] | type[MaskedTransformerFeatureExtractor]:
    if force_masked:
        return MaskedTransformerFeatureExtractor

    match model_type:
        case "sequence-default":
            return TransformerFeatureExtractor
        case "eir-input-sequence-from-linked-output-default":
            return MaskedTransformerFeatureExtractor
        case _:
            raise ValueError("Invalid sequence model type.")
