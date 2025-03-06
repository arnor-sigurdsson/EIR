from dataclasses import dataclass


@dataclass
class LatentSamplingConfig:
    """
    :param layers_to_sample: list of layers to sample latents from.

    :param max_samples_for_viz: maximum number of samples to use for visualizations.
                                If None, use all samples.

    :param batch_size_for_saving: number of samples to process before saving to disk.
    """

    layers_to_sample: list[str]
    max_samples_for_viz: int | None = 10000
    batch_size_for_saving: int = 1000
