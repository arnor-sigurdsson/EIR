from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AdversarialConfig:
    """
    :param name:
        Unique identifier for this adversarial configuration.

    :param embedding_layer_path:
        Module path to the layer whose output should be disentangled
        (e.g., 'input_modules.genotype').

    :param target_layer_path:
        Module path to the layer containing the information to disentangle from
        (e.g., 'input_modules.tabular'). With these to examples listed, we
        try to disentangle the tabular signal from the genotype layer.

    :param enabled:
        Whether this adversarial configuration is active.

    :param lambda_adv:
        Weight for the adversarial loss term. Higher values enforce stronger
        disentanglement.

    :param fc_dim:
        Hidden dimension for the adversarial discriminator network.

    :param layers:
        List where first element specifies number of residual blocks
        in the adversarial network.

    :param dropout_p:
        Dropout probability in the adversarial network.

    :param stochastic_depth_p:
        Stochastic depth probability for residual blocks in the adversarial network.

    :param projection_type:
        Type of projection layer to use before the adversarial discriminator.

    :param embedding_cache_target:
        Whether to cache 'input' or 'output' of the embedding layer.
        Default 'output' caches the layer's output activations.

    :param target_cache_target:
        Whether to cache 'input' or 'output' of the target layer.
        Default 'output' caches the layer's output activations.
        Consider using 'input' to capture the target signal before it's
        processed by the target layer's transformations.
    """

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
    embedding_cache_target: Literal["input", "output"] = "output"
    target_cache_target: Literal["input", "output"] = "output"


@dataclass
class AdversarialTrainingConfig:
    adversarial_configs: Sequence[AdversarialConfig]
