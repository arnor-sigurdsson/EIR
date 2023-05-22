from dataclasses import dataclass
from typing import Literal, Union, Sequence, Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from eir.setup.schemas import al_max_sequence_length, al_tokenizer_choices

al_sequence_operations = Literal["autoregressive", "mlm"]


@dataclass
class SequenceOutputTypeConfig:
    vocab_file: Union[None, str] = None
    max_length: "al_max_sequence_length" = "average"
    sampling_strategy_if_longer: Literal["from_start", "uniform"] = "uniform"
    min_freq: int = 10
    split_on: str = " "
    tokenizer: "al_tokenizer_choices" = None
    tokenizer_language: Union[str, None] = None

    sequence_operation: al_sequence_operations = "autoregressive"


@dataclass
class SequenceOutputSamplingConfig:
    manual_inputs: Sequence[Dict[str, str]] = tuple()
    n_eval_inputs: int = 10

    generated_sequence_length: int = 64
    top_k: int = 20
    top_p: float = 0.9
