from dataclasses import dataclass

from eir.data_load.label_setup import al_label_transformers
from eir.setup.schemas import InputConfig


@dataclass
class ComputedDeployTabularInputInfo:
    labels: "DeployLabels"
    input_config: InputConfig


@dataclass
class DeployLabels:
    label_transformers: al_label_transformers
