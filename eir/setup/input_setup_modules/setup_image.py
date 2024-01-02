from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from timm.models._registry import _model_pretrained_cfgs
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor

from eir.data_load.data_source_modules.deeplake_ops import (
    get_deeplake_input_source_iterable,
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.models.input.image.image_models import ImageModelConfig
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

    num_channels = input_type_info.num_channels
    if not num_channels:
        num_channels = infer_num_image_channels(
            data_source=input_config.input_info.input_source,
            deeplake_inner_key=input_config.input_info.input_inner_key,
        )

    data_dimension = DataDimensions(
        channels=num_channels,
        height=input_type_info.size[0],
        width=input_type_info.size[-1],
    )

    normalization_stats = get_image_normalization_values(
        input_config=input_config, data_dimensions=data_dimension
    )

    base_transforms, all_transforms = get_image_transforms(
        target_size=input_type_info.size,
        normalization_stats=normalization_stats,
        auto_augment=input_type_info.auto_augment,
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
        test_image = default_loader(path=str(test_file))
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


@dataclass
class ImageNormalizationStats:
    channel_means: torch.Tensor
    channel_stds: torch.Tensor


def get_image_normalization_values(
    input_config: schemas.InputConfig,
    data_dimensions: DataDimensions,
) -> ImageNormalizationStats:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.ImageInputDataConfig)

    pretrained_model_configs = get_timm_configs()

    means: Optional[torch.Tensor | Sequence[float]]
    stds: Optional[torch.Tensor | Sequence[float]]

    means = input_type_info.mean_normalization_values
    stds = input_type_info.stds_normalization_values

    model_config = input_config.model_config
    assert isinstance(model_config, ImageModelConfig)

    if model_config.pretrained_model:
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
            input_source = input_config.input_info.input_source
            deeplake_inner_key = input_config.input_info.input_inner_key
            logger.info(
                "Not using a pretrained model and no mean and standard deviation "
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
                tensor_iterator = (to_tensor(i.numpy()) for i in image_iter)
            else:
                file_iterator = Path(input_source).rglob("*")
                image_iterator = (default_loader(str(f)) for f in file_iterator)
                tensor_iterator = (to_tensor(i) for i in image_iterator)

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
        channel_means=means_tensor,
        channel_stds=stds_tensor,
    )

    return stats


def get_image_transforms(
    target_size: Sequence[int],
    normalization_stats: ImageNormalizationStats,
    auto_augment: bool,
) -> Tuple[Compose, Compose]:
    random_transforms = transforms.TrivialAugmentWide()
    target_resize = [int(i * 1.5) for i in target_size]

    base = [
        transforms.Resize(size=target_resize),
        transforms.CenterCrop(size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=normalization_stats.channel_means,
            std=normalization_stats.channel_stds,
        ),
    ]

    base_transforms = transforms.Compose(transforms=base)
    if auto_augment:
        logger.info("Image will be auto augmented with TrivialAugment during training.")
        all_transforms = transforms.Compose(transforms=[random_transforms] + base)
    else:
        all_transforms = base_transforms

    return base_transforms, all_transforms
