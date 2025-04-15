from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass
class ArrayOutputTypeConfig:
    """
    :param normalization:
        Which type of normalization to apply to the array data. If ``element``, will
        normalize each element in the array independently. If ``channel``, will
        normalize each channel in the array independently.
        For 'channel', assumes PyTorch format where the channel dimension is the
        first dimension.

    :param adaptive_normalization_max_samples:
        If using adaptive normalization (channel / element),
        how many samples to use to compute the normalization parameters.
        If None, will use all samples.

    :param loss:
        Which loss to use for training the model. Either ``mse`` or ``diffusion``.

    :param diffusion_time_steps:
        Number of time steps to use for diffusion loss. Only used if ``loss`` is
        set to ``diffusion``.

    :param diffusion_beta_schedule:
        Scheduler type to use for the diffusion process. Options are:
        - ``linear``
        - ``scaled_linear``
        - ``squaredcos_cap_v2``
        - ``sigmoid``
    """

    normalization: Literal["element", "channel"] | None = "channel"
    adaptive_normalization_max_samples: int | None = None
    loss: Literal["mse", "diffusion"] = "mse"
    diffusion_time_steps: int | None = 500
    diffusion_beta_schedule: Literal[
        "linear",
        "scaled_linear",
        "squaredcos_cap_v2",
        "sigmoid",
    ] = "linear"


@dataclass
class ArrayOutputSamplingConfig:
    """
    :param manual_inputs:
        Manually specified inputs to use for sequence generation. This is useful
        if you want to generate sequences based on a specific input. Depending
        on the input type, different formats are expected:

        - ``sequence``: A string written directly in the ``.yaml`` file.
        - ``omics``: A file path to NumPy array of shape ``(4, n_SNPs)`` on disk.
        - ``image``: An image file path on disk.
        - ``tabular``: A mapping of :literal:`(column key: value)` written directly
          in the ``.yaml`` file.
        - ``array``: A file path to NumPy array on disk.
        - ``bytes``: A file path to a file on disk.

    :param n_eval_inputs:
        The number of inputs automatically sampled from the validation set for
        sequence generation.

    :param diffusion_inference_steps:
        The number of steps for the diffusion process.

    :param diffusion_sampler:
        The type of scheduler to use for the diffusion process. Options are:
        - ``ddpm``
        - ``ddim``
        - ``dpm_solver``

    :param diffusion_eta:
        Parameter that controls the amount of noise added during the sampling process.
        Only used when diffusion_scheduler_type is "ddim".
    """

    manual_inputs: Sequence[dict[str, str]] = ()
    n_eval_inputs: int = 10

    diffusion_inference_steps: int = 500
    diffusion_sampler: Literal["ddpm", "ddim", "dpm_solver"] = "ddpm"
    diffusion_eta: float = 0.2
