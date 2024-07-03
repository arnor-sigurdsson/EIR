from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Dict, Literal, Optional, Type, Union

import torch
from torch import nn

from eir.models.input.array.models_cnn import CNNModel, CNNModelConfig
from eir.models.input.array.models_locally_connected import (
    FlattenFunc,
    LCLModel,
    LCLModelConfig,
)
from eir.models.input.array.models_transformers import (
    ArrayTransformer,
    ArrayTransformerConfig,
)

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions

al_array_model_types = Literal["cnn", "lcl"]

al_array_model_classes = Type[CNNModel] | Type[LCLModel] | Type[ArrayTransformer]
al_array_models = CNNModel | LCLModel | ArrayTransformer

al_array_model_config_classes = (
    Type[CNNModelConfig] | Type[LCLModelConfig] | Type[ArrayTransformerConfig]
)
al_array_model_configs = CNNModelConfig | LCLModelConfig | ArrayTransformerConfig

al_pre_normalization = Optional[Literal["instancenorm", "layernorm"]]

al_array_model_init_kwargs = dict[
    str,
    Union[
        "DataDimensions",
        CNNModelConfig,
        LCLModelConfig,
        ArrayTransformerConfig,
        FlattenFunc,
    ],
]


def get_array_model_mapping() -> Dict[str, al_array_model_classes]:
    mapping = {
        "cnn": CNNModel,
        "lcl": LCLModel,
        "transformer": ArrayTransformer,
    }

    return mapping


def get_array_model_class(model_type: al_array_model_types) -> al_array_model_classes:
    mapping = get_array_model_mapping()
    return mapping[model_type]


def get_array_config_dataclass_mapping() -> Dict[str, al_array_model_config_classes]:
    mapping = {
        "cnn": CNNModelConfig,
        "lcl": LCLModelConfig,
        "transformer": ArrayTransformerConfig,
    }

    return mapping


def get_model_config_dataclass(model_type: str) -> al_array_model_config_classes:
    mapping = get_array_config_dataclass_mapping()
    return mapping[model_type]


def get_array_model_init_kwargs(
    model_type: al_array_model_types,
    model_config: al_array_model_configs,
    data_dimensions: "DataDimensions",
) -> al_array_model_init_kwargs:
    kwargs: al_array_model_init_kwargs = {}

    model_config_dataclass = get_model_config_dataclass(model_type=model_type)
    model_config_dataclass_instance = model_config_dataclass(**model_config.__dict__)

    kwargs["model_config"] = model_config_dataclass_instance
    kwargs["data_dimensions"] = data_dimensions

    match model_type:
        case "lcl":
            assert isinstance(model_config, LCLModelConfig)

            if model_config.patch_size is not None:
                assert isinstance(model_config.patch_size, (tuple, list))
                assert len(model_config.patch_size) == 3, model_config.patch_size
                kwargs["flatten_fn"] = partial(
                    patchify_and_flatten,
                    size=model_config.patch_size,
                )
            else:
                kwargs["flatten_fn"] = partial(torch.flatten, start_dim=1)

        case "transformer":
            assert isinstance(model_config, ArrayTransformerConfig)
            kwargs["flatten_fn"] = partial(
                patchify,
                size=model_config.patch_size,
                stride=model_config.patch_size,
            )

    return kwargs


def check_patch_and_input_size_compatibility(
    patch_size: Union[tuple[int, int, int], list[int]],
    data_dimensions: "DataDimensions",
) -> None:
    assert isinstance(patch_size, (tuple, list))
    assert len(patch_size) == 3, patch_size

    channels, height, width = patch_size

    if (
        data_dimensions.channels % channels != 0
        or data_dimensions.height % height != 0
        or data_dimensions.width % width != 0
    ):
        mismatch_details = (
            f"Data dimensions {data_dimensions.full_shape()} "
            f"cannot be evenly divided into patches of size {patch_size}. "
            f"Mismatch in channels: {data_dimensions.channels % channels}, "
            f"height: {data_dimensions.height % height}, "
            f"width: {data_dimensions.width % width}."
        )
        raise ValueError(mismatch_details)


def patchify_and_flatten(
    x: torch.Tensor,
    size: tuple[int, int, int],
) -> torch.Tensor:
    stride = size
    patches = patchify(x=x, size=size, stride=stride)
    flattened = flatten_patches(patches=patches)
    return flattened


def patchify(
    x: torch.Tensor, size: tuple[int, int, int], stride: tuple[int, int, int]
) -> torch.Tensor:
    """

    size: (C, H, W)

    Input shape: [256, 3, 64, 64]

    Batch size: batch_size
    Channels: C
    Vertical patches: height / H
    Horizontal patches: width / W
    Patch channels: C
    Patch height: H
    Patch width: W

    After unfolding: [batch_size, C, height / H, width / W, C, H, W]

    After permuting: [batch_size, height / H, width / W, C, C, H, W]
    """
    patches = (
        x.unfold(1, size[0], stride[0])
        .unfold(2, size[1], stride[1])
        .unfold(3, size[2], stride[2])
    )
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6)
    return patches


def flatten_patches(patches: torch.Tensor) -> torch.Tensor:
    reshaped_patches = patches.reshape(patches.size(0), -1)
    return reshaped_patches


@dataclass
class ArrayModelConfig:
    """
    :param model_type:
         Which type of image model to use.

    :param model_init_config:
          Configuration used to initialise model.
    """

    model_type: al_array_model_types
    model_init_config: al_array_model_configs
    pre_normalization: al_pre_normalization = None


class ArrayWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: al_array_models,
        normalization: al_pre_normalization,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.data_dimensions = feature_extractor.data_dimensions
        self.pre_normalization = get_pre_normalization_layer(
            normalization=normalization, data_dimensions=self.data_dimensions
        )
        self.output_shape = self.feature_extractor.output_shape

    @property
    def num_out_features(self):
        return self.feature_extractor.num_out_features

    @property
    def l1_penalized_weights(self) -> torch.Tensor:
        return self.feature_extractor.l1_penalized_weights

    def forward(self, x):
        out = self.pre_normalization(x)
        out = self.feature_extractor(out)
        return out


def get_pre_normalization_layer(
    normalization: al_pre_normalization,
    data_dimensions: "DataDimensions",
) -> Union[nn.InstanceNorm2d, nn.LayerNorm, nn.Identity]:
    channels = data_dimensions.channels
    height = data_dimensions.height
    width = data_dimensions.width

    match normalization:
        case "instancenorm":
            return nn.InstanceNorm2d(
                num_features=channels, affine=True, track_running_stats=True
            )
        case "layernorm":
            return nn.LayerNorm(normalized_shape=[channels, height, width])
        case None:
            return nn.Identity()
