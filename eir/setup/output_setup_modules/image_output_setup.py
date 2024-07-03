from dataclasses import dataclass
from typing import Optional

from torchvision.transforms import Compose

from eir.models.output.array.array_output_modules import ArrayOutputModuleConfig
from eir.setup.input_setup_modules.common import DataDimensions
from eir.setup.input_setup_modules.setup_image import (
    ImageNormalizationStats,
    get_image_normalization_values,
    get_image_transforms,
    get_num_channels_wrapper,
)
from eir.setup.schemas import ImageOutputTypeConfig, OutputConfig
from eir.train_utils.step_modules.diffusion import (
    DiffusionConfig,
    initialize_diffusion_config,
)
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class ComputedImageOutputInfo:
    output_config: OutputConfig
    base_transforms: Compose
    all_transforms: Compose
    data_dimensions: DataDimensions
    num_channels: int
    normalization_stats: Optional[ImageNormalizationStats] = None
    diffusion_config: Optional[DiffusionConfig] = None


def set_up_image_output(
    output_config: OutputConfig, *args, **kwargs
) -> ComputedImageOutputInfo:

    output_info = output_config.output_info
    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, ImageOutputTypeConfig)

    image_mode = output_type_info.mode
    num_channels = output_type_info.num_channels

    num_channels = get_num_channels_wrapper(
        image_mode=image_mode,
        num_channels=num_channels,
        source=output_info.output_source,
        deeplake_inner_key=output_info.output_inner_key,
    )

    data_dimensions = DataDimensions(
        channels=num_channels,
        height=output_type_info.size[0],
        width=output_type_info.size[-1],
    )

    output_type_info = output_config.output_type_info
    assert isinstance(output_type_info, ImageOutputTypeConfig)

    oti = output_type_info
    model_config = output_config.model_config
    assert isinstance(model_config, ArrayOutputModuleConfig)
    normalization_stats = get_image_normalization_values(
        source=output_config.output_info.output_source,
        inner_key=output_config.output_info.output_inner_key,
        model_config=model_config,
        mean_normalization_values=oti.mean_normalization_values,
        stds_normalization_values=oti.stds_normalization_values,
        adaptive_normalization_max_samples=oti.adaptive_normalization_max_samples,
        data_dimensions=data_dimensions,
        image_mode=oti.mode,
    )

    base_transforms, all_transforms = get_image_transforms(
        target_size=(data_dimensions.height, data_dimensions.width),
        normalization_stats=normalization_stats,
        auto_augment=False,
        resize_approach=oti.resize_approach,
    )

    diffusion_config = None
    if output_type_info.loss == "diffusion":
        time_steps = output_type_info.diffusion_time_steps
        if time_steps is None:
            raise ValueError(
                "Diffusion loss requires specifying the number of time steps."
                "Please set `diffusion_time_steps` in the output config."
            )
        diffusion_config = initialize_diffusion_config(time_steps=time_steps)

    image_output_object = ComputedImageOutputInfo(
        output_config=output_config,
        data_dimensions=data_dimensions,
        base_transforms=base_transforms,
        all_transforms=all_transforms,
        num_channels=num_channels,
        normalization_stats=normalization_stats,
        diffusion_config=diffusion_config,
    )

    return image_output_object
