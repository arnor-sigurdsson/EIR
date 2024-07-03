from dataclasses import dataclass
from typing import Literal, Optional, Sequence

al_broker_projection_types = Literal[
    "lcl",
    "lcl_residual",
    "linear",
    "grouped_linear",
    "pool",
    "cnn",
    "interpolate",
]

al_broker_fusion_types = Literal["cross-attention", "sum", "cat+conv"]


@dataclass
class TensorMessageConfig:
    """
    :param name:
        Name of the message, used to identify the message in the broker, e.g. when
        a message is cached with a name, the name is used by other messages to
        use the cached tensor.

    :param layer_path:
        Path to the layer in the model that the message is extracted from.

    :param cache_tensor:
        Whether to cache the tensor in the broker.

    :param layer_cache_target:
        Whether to cache the input or output of the layer.

    :param use_from_cache:
        List of names of messages (i.e. from the 'name' field) that this message
        will use from the cache. Assumes that the messages have been cached earlier
        in the model flow.

    :param cache_fusion_type:
        Type of fusion to use when combining the cached tensors. Options are:

        - ``cross-attention``: Use cross-attention to combine the tensors.
        - ``sum``: Learnable gated sum to combine the tensors.
        - ``cat+conv``: Concatenate the tensors and apply a convolutional layer.

    :param projection_type:
        Type of projection to use when projecting the tensor to the target space.
        Options are:

        - ``lcl``: Locally connected layer.
        - ``lcl_residual``: Locally connected layer with residual connection.
        - ``cnn``: Convolutional layer, only supports down sampling for now.
        - ``linear``: Linear layer.
        - ``pool``: Adaptive average pooling layer.
        - ``grouped_linear``: Grouped linear layer (each dimension is projected
          separately with a learnable linear layer).

        If the tensor is already of the target size, no projection is performed.
    """

    name: str
    layer_path: str
    cache_tensor: bool = False
    layer_cache_target: Literal["input", "output"] = "output"
    use_from_cache: Optional[list[str]] = None
    cache_fusion_type: al_broker_fusion_types = "cat+conv"
    projection_type: al_broker_projection_types = "lcl"


@dataclass
class TensorBrokerConfig:
    """
    :param message_configs:
        List of message configurations for the broker.
    """

    message_configs: Sequence[TensorMessageConfig]
