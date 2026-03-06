from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

al_broker_projection_types = Literal[
    "lcl",
    "lcl_residual",
    "lcl+mlp_residual",
    "mlp_residual",
    "linear",
    "grouped_linear",
    "pool",
    "cnn",
    "interpolate",
]

al_broker_fusion_types = Literal["cross-attention", "sum", "cat+conv", "additive"]


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
        - ``additive``: Simple element-wise addition.

    :param projection_type:
        Type of projection to use when projecting the tensor to the target space.
        Options are:

        - ``lcl``: Locally connected layer.
        - ``lcl_residual``: Locally connected layer with residual connection.
        - ``lcl+mlp_residual``: Locally connected layer followed by MLP residual block.
        - ``mlp_residual``: MLP residual block.
        - ``cnn``: Convolutional layer, only supports down sampling for now.
        - ``linear``: Linear layer.
        - ``pool``: Adaptive average pooling layer.
        - ``grouped_linear``: Grouped linear layer (each dimension is projected
          separately with a learnable linear layer).
        - ``interpolate``: Interpolates the tensor to the target size.
        - ``sequence``: Project to a 2D sequence to be e.g. used with cross-attention.

        If the tensor is already of the target size, no projection is performed.

    :param kernel_width_divisible_by:
        For LCL-based projections, constrain kernel width to be divisible by this value.

    :param projection_intermediate_factor:
        For ``lcl+mlp_residual`` projections, multiply the target dimension by this
        factor to set the intermediate LCL output size. The MLP residual block then
        compresses from ``target * factor`` to the final target dimension.
        For example, a factor of 4 with target 512 would produce an intermediate size
        of 2048.

    :param cache_dropout_p:
        Probability of dropping cached tensor injection during training. When set to
        a value > 0, the cached tensor will be randomly skipped during forward pass
        with this probability during training mode. During evaluation, cache is always
        used if available. Useful for improving robustness when auxiliary features may
        be unavailable at inference time.
    """

    name: str
    layer_path: str
    cache_tensor: bool = False
    layer_cache_target: Literal["input", "output"] = "output"
    use_from_cache: list[str] | None = None
    cache_fusion_type: al_broker_fusion_types = "cat+conv"
    projection_type: al_broker_projection_types = "lcl"
    kernel_width_divisible_by: int | None = None
    projection_intermediate_factor: int | None = None
    cache_dropout_p: float = 0.0


@dataclass
class TensorBrokerConfig:
    """
    :param message_configs:
        List of message configurations for the broker.
    """

    message_configs: Sequence[TensorMessageConfig]
