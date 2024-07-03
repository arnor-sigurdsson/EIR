from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Generator, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from timm.models._registry import _model_pretrained_cfgs
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor

from eir.data_load.data_source_modules.deeplake_ops import (
    get_deeplake_input_source_iterable,
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.models.input.image.image_models import ImageModelConfig
from eir.models.output.array.array_output_modules import ArrayOutputModuleConfig
from eir.setup import schemas
from eir.setup.input_setup_modules.common import DataDimensions
from eir.setup.setup_utils import ChannelBasedRunningStatistics, collect_stats
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class PretrainedImageModelInfo:
    url: str
    num_classes: int
    input_size: Sequence[int]
    pool_size: Sequence[int]
    mean: Sequence[float]
    std: Sequence[float]
    first_conv: str
    classifier: str


def get_timm_configs() -> Dict[str, PretrainedImageModelInfo]:
    default_configs = {}
    field_names = {i.name for i in fields(PretrainedImageModelInfo)}
    for name, pretrained_config in _model_pretrained_cfgs.items():
        config_dict = asdict(pretrained_config)
        common = {k: v for k, v in config_dict.items() if k in field_names}

        default_configs[name] = PretrainedImageModelInfo(**common)

    return default_configs


@dataclass
class ComputedImageInputInfo:
    input_config: schemas.InputConfig
    base_transforms: Compose
    all_transforms: Compose
    normalization_stats: "ImageNormalizationStats"
    num_channels: int
    data_dimensions: "DataDimensions"


def set_up_image_input_for_training(
    input_config: schemas.InputConfig, *args, **kwargs
) -> ComputedImageInputInfo:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.ImageInputDataConfig)

    image_mode = input_type_info.mode
    num_channels = input_type_info.num_channels
    if not num_channels and not image_mode:
        num_channels = infer_num_image_channels(
            data_source=input_config.input_info.input_source,
            deeplake_inner_key=input_config.input_info.input_inner_key,
        )

    num_channels = get_num_channels_wrapper(
        image_mode=image_mode,
        num_channels=num_channels,
        source=input_config.input_info.input_source,
        deeplake_inner_key=input_config.input_info.input_inner_key,
    )

    data_dimension = DataDimensions(
        channels=num_channels,
        height=input_type_info.size[0],
        width=input_type_info.size[-1],
    )

    iti = input_type_info
    model_config = input_config.model_config
    assert isinstance(model_config, ImageModelConfig)
    normalization_stats = get_image_normalization_values(
        source=input_config.input_info.input_source,
        inner_key=input_config.input_info.input_inner_key,
        model_config=model_config,
        mean_normalization_values=iti.mean_normalization_values,
        stds_normalization_values=iti.stds_normalization_values,
        adaptive_normalization_max_samples=iti.adaptive_normalization_max_samples,
        data_dimensions=data_dimension,
        image_mode=iti.mode,
    )

    base_transforms, all_transforms = get_image_transforms(
        target_size=input_type_info.size,
        normalization_stats=normalization_stats,
        auto_augment=input_type_info.auto_augment,
        resize_approach=input_type_info.resize_approach,
    )

    image_input_info = ComputedImageInputInfo(
        input_config=input_config,
        base_transforms=base_transforms,
        all_transforms=all_transforms,
        normalization_stats=normalization_stats,
        num_channels=num_channels,
        data_dimensions=data_dimension,
    )

    return image_input_info


def get_num_channels_wrapper(
    image_mode: Optional[str],
    num_channels: Optional[int],
    source: str,
    deeplake_inner_key: Optional[str],
) -> int:

    if image_mode and num_channels:
        raise ValueError(
            "Got both image mode and number of channels. Please only specify one."
        )

    if image_mode is not None:
        num_channels = get_num_channels_from_image_mode(image_mode=image_mode)
        return num_channels

    elif num_channels:
        return num_channels

    else:
        num_channels = infer_num_image_channels(
            data_source=source,
            deeplake_inner_key=deeplake_inner_key,
        )
        return num_channels


def get_num_channels_from_image_mode(image_mode: Optional[str]) -> int:

    match image_mode:
        case "RGB":
            channels = 3
        case "L":
            channels = 1
        case "RGBA":
            channels = 4
        case _:
            raise ValueError(f"Unknown image mode: {image_mode}")

    logger.debug(
        "Inferred number of channels from image mode %s as: %d", image_mode, channels
    )

    return channels


def infer_num_image_channels(
    data_source: str, deeplake_inner_key: Optional[str]
) -> int:
    if is_deeplake_dataset(data_source=data_source):
        assert deeplake_inner_key is not None
        deeplake_ds = load_deeplake_dataset(data_source=data_source)
        deeplake_iter = get_deeplake_input_source_iterable(
            deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
        )
        test_image_array = next(deeplake_iter).numpy()
        data_pointer = (
            f"[deeplake dataset {data_source}, input {deeplake_inner_key}, "
            f"image ID: {deeplake_ds['ID'][0].text()}]"
        )
    else:
        test_file = next(Path(data_source).iterdir())
        test_image = default_image_loader(path=str(test_file))
        test_image_array = np.array(test_image)
        data_pointer = test_file.name

    if test_image_array.ndim == 2:
        num_channels = 1
    else:
        num_channels = test_image_array.shape[-1]

    logger.info(
        "Inferring number of channels from source %s (using %s) as: %d",
        data_source,
        data_pointer,
        num_channels,
    )

    return num_channels


def default_image_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        img.load()
        return img


@dataclass
class ImageNormalizationStats:
    means: torch.Tensor
    stds: torch.Tensor


def get_image_normalization_values(
    source: str,
    inner_key: Optional[str],
    model_config: ImageModelConfig | ArrayOutputModuleConfig,
    mean_normalization_values: Optional[Sequence[float] | torch.Tensor],
    stds_normalization_values: Optional[Sequence[float] | torch.Tensor],
    adaptive_normalization_max_samples: Optional[int],
    data_dimensions: DataDimensions,
    image_mode: Optional[str],
) -> ImageNormalizationStats:

    pretrained_model_configs = get_timm_configs()

    means: Optional[torch.Tensor | Sequence[float]]
    stds: Optional[torch.Tensor | Sequence[float]]

    means = mean_normalization_values
    stds = stds_normalization_values

    if hasattr(model_config, "pretrained_model") and model_config.pretrained_model:
        cur_config = pretrained_model_configs[model_config.model_type]

        if not means:
            logger.info(
                "Using inferred image channel means (%s) from base on training "
                "statistics from pretrained '%s' model.",
                cur_config.mean,
                model_config.model_type,
            )
            means = cur_config.mean
        else:
            logger.warning(
                "Got manual values for channel means (%s) when using "
                "pretrained model '%s'. Usually one would use the means "
                "from the training data when '%s' was trained.",
                means,
                model_config.model_type,
                model_config.model_type,
            )
        if not stds:
            logger.info(
                "Using inferred image channel standard deviations (%s) from base on "
                "training statistics from pretrained '%s' model.",
                cur_config.std,
                model_config.model_type,
            )
            stds = cur_config.std
        else:
            logger.warning(
                "Got manual values for channel standard deviations (%s) "
                "when using pretrained model '%s'. Usually one would use "
                "the means from the training data when '%s' was trained.",
                stds,
                model_config.model_type,
                model_config.model_type,
            )
    else:
        if not means or not stds:
            input_source = source
            deeplake_inner_key = inner_key
            logger.info(
                "Not using an external pretrained model "
                "and no mean and standard deviation "
                "statistics passed in. Gathering running image means and standard "
                "deviations from %s.",
                input_source,
            )

            if is_deeplake_dataset(data_source=input_source):
                deeplake_ds = load_deeplake_dataset(data_source=input_source)
                assert deeplake_inner_key is not None
                image_iter = get_deeplake_input_source_iterable(
                    deeplake_dataset=deeplake_ds, inner_key=deeplake_inner_key
                )

                numpy_iter = (i.numpy() for i in image_iter)

                if image_mode:
                    image_iter = (Image.fromarray(i) for i in numpy_iter)
                    image_iter = (i.convert(image_mode) for i in image_iter)
                    numpy_iter = (np.array(i) for i in image_iter)

                tensor_iterator = (to_tensor(i).float() for i in numpy_iter)

            else:
                file_iterator = Path(input_source).rglob("*")
                image_iterator = (default_image_loader(str(f)) for f in file_iterator)
                if image_mode:
                    image_iterator = (i.convert(image_mode) for i in image_iterator)

                tensor_iterator = (to_tensor(i) for i in image_iterator)

            tensor_iterator = _get_maybe_truncated_tensor_iterator(
                tensor_iterator=tensor_iterator,
                max_samples=adaptive_normalization_max_samples,
            )

            gathered_stats = collect_stats(
                tensor_iterable=tensor_iterator,
                collector_class=ChannelBasedRunningStatistics,
                shape=data_dimensions.full_shape(),
            )
            means = gathered_stats.mean
            stds = gathered_stats.std
            logger.info(
                "Gathered the following means: %s and standard deviations: %s "
                "from %s.",
                means,
                stds,
                input_source,
            )

    assert means is not None
    assert stds is not None

    if torch.is_tensor(means):
        assert isinstance(means, torch.Tensor)
        means_tensor = means.clone().detach()
    else:
        means_tensor = torch.tensor(means)
    if torch.is_tensor(stds):
        assert isinstance(stds, torch.Tensor)
        stds_tensor = stds.clone().detach()
    else:
        stds_tensor = torch.tensor(stds)

    stats = ImageNormalizationStats(
        means=means_tensor,
        stds=stds_tensor,
    )

    return stats


def _get_maybe_truncated_tensor_iterator(
    tensor_iterator: Generator[torch.Tensor, None, None], max_samples: Optional[int]
) -> Generator[torch.Tensor, None, None]:

    if max_samples is not None:
        tensor_iterator = (t for _, t in zip(range(max_samples), tensor_iterator))

    return tensor_iterator


def get_image_transforms(
    target_size: Sequence[int],
    normalization_stats: ImageNormalizationStats,
    auto_augment: bool,
    resize_approach: str,
) -> Tuple[transforms.Compose, transforms.Compose]:
    if len(target_size) == 1:
        target_size = (target_size[0], target_size[0])
        logger.info(
            "Got target size as a single value. Assuming square image and "
            "setting target size to %s.",
            target_size,
        )

    random_transforms = transforms.TrivialAugmentWide()

    if resize_approach == "randomcrop":
        resize_transform = transforms.Resize([int(i * 1.5) for i in target_size])
        crop_transform = transforms.RandomCrop(target_size)
    elif resize_approach == "centercrop":
        resize_transform = transforms.Resize([int(i * 1.5) for i in target_size])
        crop_transform = transforms.CenterCrop(target_size)
    else:
        resize_transform = transforms.Resize(target_size)
        crop_transform = transforms.Lambda(lambda x: x)

    base = [
        resize_transform,
        crop_transform,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalization_stats.means,
            std=normalization_stats.stds,
        ),
    ]

    base_transforms = transforms.Compose(transforms=base)
    if auto_augment:
        logger.info("Image will be auto augmented with TrivialAugment during training.")
        all_transforms = transforms.Compose(transforms=[random_transforms] + base)
    else:
        all_transforms = base_transforms

    return base_transforms, all_transforms
