from dataclasses import dataclass
from typing import Literal, Optional, Sequence


@dataclass
class TensorMessageConfig:
    name: str
    layer_path: str
    cache_tensor: bool = False
    use_from_cache: Optional[list[str]] = None
    cache_fusion_type: Literal["cross-attention", "sum", "cat+conv"] = "cat+conv"
    allow_projection: bool = True
    projection_type: Literal["lcl", "lcl_residual", "cnn", "linear", "pool"] = "linear"


@dataclass
class TensorBrokerConfig:
    message_configs: Sequence[TensorMessageConfig]