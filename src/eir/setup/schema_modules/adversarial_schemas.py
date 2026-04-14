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
        disentanglement. Default of 1.0 assumes main_loss and adv_loss are on
        similar scales - this is the most important hyperparameter to tune based
        on the relative scales of your losses.

    :param warmup_steps:
        Number of training steps over which to linearly increase lambda_adv from
        0.0 to its final value. This warmup allows the main task to stabilize
        before applying the full adversarial penalty. Default of 5000 is a good
        starting point for most tasks.

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

    :param projection_lcl_residual_blocks:
        When ``projection_type='lcl+mlp_residual'``, use progressive LCL residual
        blocks to reduce dimensionality before the final MLP residual block,
        instead of a single LCL projection. Useful for very high-dimensional
        embeddings where a dense MLP would blow up in parameters.

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
    lambda_adv: float = 1.0
    warmup_steps: int = 5000
    fc_dim: int = 128
    layers: list[int] = field(default_factory=lambda: [2])
    dropout_p: float = 0.1
    stochastic_depth_p: float = 0.0
    projection_type: Literal[
        "linear",
        "lcl",
        "lcl_residual",
        "mlp_residual",
        "lcl+mlp_residual",
        "grouped_linear",
    ] = "linear"
    projection_lcl_residual_blocks: bool = True
    embedding_cache_target: Literal["input", "output"] = "output"
    target_cache_target: Literal["input", "output"] = "output"


@dataclass
class AdversarialTrainingConfig:
    adversarial_configs: Sequence[AdversarialConfig]
