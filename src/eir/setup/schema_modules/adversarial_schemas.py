from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AdversarialConfig:
    name: str
    embedding_layer_path: str
    target_layer_path: str
    enabled: bool = True
    lambda_adv: float = 0.1
    fc_dim: int = 128
    layers: list[int] = field(default_factory=lambda: [2])
    dropout_p: float = 0.1
    stochastic_depth_p: float = 0.0
    projection_type: Literal[
        "linear", "lcl", "lcl_residual", "mlp_residual", "grouped_linear"
    ] = "linear"


@dataclass
class AdversarialTrainingConfig:
    adversarial_configs: Sequence[AdversarialConfig]
