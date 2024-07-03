from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union


@dataclass
class ImageOutputTypeConfig:
    """
    :param adaptive_normalization_max_samples:
        If using adaptive normalization (channel / element),
        how many samples to use to compute the normalization parameters.
        If None, will use all samples.

    :param resize_approach:
        The method used for resizing the images. Options are:
        - "resize": Directly resize the image to the target size.
        - "randomcrop": Resize the image to a larger size than the target and then
        apply a random crop to the target size.
        - "centercrop": Resize the image to a larger size than the target and then
        apply a center crop to the target size.

    :param mean_normalization_values:
        Average channel values to normalize images with. This can be a sequence matching
        the number of channels, or None. If None and using a pretrained model, the
        values used for the model pretraining will be used. If None and training from
        scratch, will iterate over training data and compute the running average
        per channel.

    :param stds_normalization_values:
        Standard deviation channel values to normalize images with. This can be a
        sequence mathing the number of channels, or None. If None and using a
        pretrained model, the values used for the model pretraining will be used.
        If None and training from scratch, will iterate over training data and compute
        the running average per channel.

    :param mode:
        An explicit mode to convert loaded images to. Useful when working with
        input data with a mixed number of channels, or you want to convert
        images to a specific mode.
        Options are
        - "RGB": Red, Green, Blue (channels=3)
        - "L": Grayscale (channels=1)
        - "RGBA": Red, Green, Blue, Alpha (channels=4)

    :param num_channels:
        Number of channels in the images. If None, will try to infer the number of
        channels from a random image in the training data. Useful when known
        ahead of time how many channels the images have, will raise an error if
        an image with a different number of channels is encountered.

    :param loss:
        Which loss to use for training the model. Either ``mse`` or ``diffusion``.

    :param diffusion_time_steps:
        Number of time steps to use for diffusion loss. Only used if ``loss`` is
        set to ``diffusion``.
    """

    size: Sequence[int] = (64,)
    resize_approach: Union[Literal["resize", "randomcrop", "centercrop"]] = "resize"
    adaptive_normalization_max_samples: Optional[int] = None
    mean_normalization_values: Union[None, Sequence[float]] = None
    stds_normalization_values: Union[None, Sequence[float]] = None
    mode: Optional[Literal["RGB", "L", "RGBA"]] = None
    num_channels: Optional[int] = None
    loss: Literal["mse", "diffusion"] = "mse"
    diffusion_time_steps: Optional[int] = 1000


@dataclass
class ImageOutputSamplingConfig:
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
