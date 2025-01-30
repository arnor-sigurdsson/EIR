from dataclasses import dataclass


@dataclass
class LatentSamplingConfig:
    """
    :param layers_to_sample: list of layers to sample latents from.
    """

    layers_to_sample: list[str]
