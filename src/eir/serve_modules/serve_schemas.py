from dataclasses import dataclass

from eir.data_load.label_setup import al_label_transformers
from eir.setup.schemas import InputConfig


@dataclass
class ComputedServeTabularInputInfo:
    labels: "ServeLabels"
    input_config: InputConfig


@dataclass
class ServeLabels:
    label_transformers: al_label_transformers
