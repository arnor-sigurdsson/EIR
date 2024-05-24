import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from eir.models.input.array.models_locally_connected import FlattenFunc
from eir.models.input.sequence.transformer_models import (
    get_positional_representation_class,
    parse_dim_feedforward,
)

if TYPE_CHECKING:
    from eir.setup.input_setup_modules.common import DataDimensions


@dataclass
class ArrayTransformerConfig:
    """
    :param patch_size:
        Controls the size of the patches used in the first layer. If set to ``None``,
        the input is flattened according to the torch ``flatten`` function. Note that
        when using this parameter, we generally want the kernel width to be set to
        the multiplication of the patch size. Order follows PyTorch convention, i.e.,
        [channels, height, width].

    :param embedding_dim:
        The embedding dimension each patch is projected to. This is also the
        dimension of the transformer encoder layers.

    :param num_heads:
        The number of heads in the multi-head attention layers.

    :param num_layers:
        The number of transformer encoder layers.

    :param dim_feedforward:
        The dimension of the feedforward layers in the transformer model.

    :param dropout:
        The dropout rate to use in the transformer encoder layers.

    :param position:
        Whether to encode the token position or use learnable position embeddings.

    :param position_dropout:
        The dropout rate to use in the position encoding/embedding.
    """

    patch_size: tuple[int, ...]
    embedding_dim: int

    num_heads: int = 8
    num_layers: int = 2
    dim_feedforward: Union[int, Literal["auto"]] = "auto"
    dropout: float = 0.10

    position: Literal["encode", "embed"] = "encode"
    position_dropout: float = 0.10


class ArrayTransformer(nn.Module):
    def __init__(
        self,
        model_config: ArrayTransformerConfig,
        data_dimensions: "DataDimensions",
        flatten_fn: FlattenFunc,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.data_dimensions = data_dimensions
        self.flatten_fn = flatten_fn
        self.embedding_dim = self.model_config.embedding_dim

        n_elements_per_patch = _compute_num_patch_elements(
            patch_size=self.model_config.patch_size
        )

        self.num_patches = _compute_num_patches(
            input_num_elements=self.data_dimensions.num_elements(),
            n_elements_per_patch=n_elements_per_patch,
        )

        self.embedding_projection = nn.Linear(
            in_features=n_elements_per_patch,
            out_features=self.embedding_dim,
        )

        pos_repr_class = get_positional_representation_class(
            position_model_config=self.model_config.position
        )
        self.pos_representation = pos_repr_class(
            embedding_dim=self.embedding_dim,
            dropout=self.model_config.position_dropout,
            max_length=self.num_patches,
        )

        dim_feed_forward = parse_dim_feedforward(
            dim_feedforward=self.model_config.dim_feedforward,
            embedding_dim=self.embedding_dim,
        )

        encoder_layer_base = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.model_config.num_heads,
            dim_feedforward=dim_feed_forward,
            dropout=self.model_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer_base,
            num_layers=self.model_config.num_layers,
            enable_nested_tensor=False,
        )

        self.output_shape = (1, self.num_patches, self.embedding_dim)

    @property
    def num_out_features(self) -> int:
        return self.num_patches * self.embedding_dim

    def forward(self, input: Tensor) -> Tensor:
        out = self.flatten_fn(input)
        out = out.reshape(out.shape[0], self.num_patches, -1)

        out = self.embedding_projection(out)

        out = out * math.sqrt(self.embedding_dim)
        out = self.pos_representation(out)
        out = self.transformer_encoder(out)
        out = out.flatten(1)
        return out


def _compute_num_patches(input_num_elements: int, n_elements_per_patch: int) -> int:
    num_patches = int(input_num_elements / n_elements_per_patch)
    return num_patches


def _compute_num_patch_elements(patch_size: tuple[int, ...]) -> int:
    number_of_patches = 1
    for element in patch_size:
        number_of_patches *= element
    return number_of_patches
