from dataclasses import dataclass
from typing import Literal, Optional, Sequence


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
    """

    normalization: Optional[Literal["element", "channel"]] = "channel"
    adaptive_normalization_max_samples: Optional[int] = None


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
    """

    manual_inputs: Sequence[dict[str, str]] = tuple()
    n_eval_inputs: int = 10
