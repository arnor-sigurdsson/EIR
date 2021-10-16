import math
from dataclasses import dataclass
from typing import Union, Literal, Type, Callable, Tuple, Dict
from functools import partial

import torch
from aislib.misc_utils import get_logger
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad

from eir.models.layers import _find_split_padding_needed

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class TransformerWrapperModelConfig:
    """
    :param embedding_dim:
        Which dimension to use for the embeddings. If ``None``, will autoamtically set
        this value based on the number of tokens and attention heads.

    :param position:
        Whether to use positional encodings or embeddings for representing token
        positions.

    :param position_dropout:
        Dropout to use in the positional encoding/embeddings.

    :param window_size:
        If set to more than 0, will apply a sliding window of feature
        extraction over the input, meaning the model (e.g. transformer) will only
        see a part of the input at a time. Can be Useful to avoid the O(n²)
        complexity of transformers, as it becomes n_windows * O(window_size²) instead.
    """

    position: Literal["encode", "embed"] = "encode"
    position_dropout: float = 0.10
    window_size: int = 0


class TransformerWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: "TransformerFeatureExtractor",
        model_config: TransformerWrapperModelConfig,
        embedding_dim: int,
        num_tokens: int,
        max_length: int,
    ) -> None:

        super().__init__()
        self.model_config = model_config
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.max_length = max_length

        pos_repr_class = get_positional_representation_class(
            position_model_config=self.model_config.position
        )
        self.pos_representation = pos_repr_class(
            embedding_dim=self.embedding_dim,
            dropout=model_config.position_dropout,
            max_length=self.max_length,
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.num_tokens, embedding_dim=self.embedding_dim
        )

        self.feature_extractor = feature_extractor

        self.init_weights()

        (
            self.dynamic_extras,
            self.extract_features,
        ) = _get_transformer_wrapper_feature_extractor(
            feature_extractor=self.feature_extractor,
            window_size=self.model_config.window_size,
            max_length=max_length,
        )

    @property
    def num_out_features(self) -> int:
        padding = self.dynamic_extras.get("padding", 0)
        return (self.max_length + padding) * self.embedding_dim

    def embed_tokens(self, input: torch.Tensor) -> torch.Tensor:
        return self.embedding(input)

    def init_weights(self) -> None:
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, input: Tensor) -> Tensor:
        out = input * math.sqrt(self.embedding_dim)
        out = self.pos_representation(out)
        out = self.extract_features(out)

        return out


def get_embedding_dim_for_sequence_model(
    embedding_dim: Union[None, int], num_tokens: int, num_heads: int
) -> int:

    if embedding_dim is None:
        auto_emb_dim = math.ceil((int(num_tokens ** 0.25) / num_heads)) * num_heads
        logger.info(
            "Setting up automatic embedding dimension of %d based on %d "
            "tokens and %d attention heads.",
            auto_emb_dim,
            num_tokens,
            num_heads,
        )
        embedding_dim = auto_emb_dim

    return embedding_dim


def _get_transformer_wrapper_feature_extractor(
    feature_extractor: "TransformerFeatureExtractor",
    window_size: int,
    max_length: int,
) -> Tuple[Dict, Callable[[torch.Tensor], torch.Tensor]]:

    dynamic_extras = {}
    if not window_size:
        extractor = partial(
            _simple_transformer_forward, feature_extractor=feature_extractor
        )
    else:
        num_chunks = int(math.ceil(max_length / window_size))
        logger.debug(
            "Setting num chunks to %d as window size of %d and maximum sequence length "
            "of %d were passed in.",
            num_chunks,
            window_size,
            max_length,
        )

        padding = _find_split_padding_needed(
            input_size=max_length,
            split_size=window_size,
            num_chunks=num_chunks,
        )
        dynamic_extras["padding"] = padding

        extractor = partial(
            _conv_transfomer_forward,
            feature_extractor=feature_extractor,
            max_length=max_length,
            window_size=window_size,
            padding=padding,
        )

    return dynamic_extras, extractor


def _simple_transformer_forward(
    input: torch.Tensor, feature_extractor: "TransformerFeatureExtractor"
) -> torch.Tensor:
    return feature_extractor(input=input)


def _conv_transfomer_forward(
    input: torch.Tensor,
    feature_extractor: "TransformerFeatureExtractor",
    max_length: int,
    window_size: int,
    padding: int,
) -> torch.Tensor:

    out = pad(input=input, pad=[0, 0, padding, 0])
    total_length = max_length + padding

    aggregated_out = None
    for lower_index in range(0, total_length, window_size):
        upper_index = lower_index + window_size

        cur_input = out[:, lower_index:upper_index, :]
        cur_out = feature_extractor(input=cur_input).flatten(1)

        if aggregated_out is None:
            aggregated_out = cur_out
        else:
            aggregated_out = torch.cat((aggregated_out, cur_out), dim=1)

    return aggregated_out


@dataclass
class BasicTransformerFeatureExtractorModelConfig:
    """
    :param num_heads:
        The number of heads in the multi-head attention models

    :param num_layers:
        The number of encoder blocks in the transformer model.

    :param dropout:
         Dropout value to use in the encoder layers.

    """

    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.10


class TransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_config: BasicTransformerFeatureExtractorModelConfig,
        embedding_dim: int,
        num_tokens: int,
        max_length: int,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.max_length = max_length

        encoder_layers = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=model_config.num_heads,
            dim_feedforward=max_length,
            dropout=model_config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=model_config.num_layers
        )

    @property
    def num_out_features(self) -> int:
        return self.max_length * self.embedding_dim

    def forward(self, input: Tensor) -> Tensor:
        out = self.transformer_encoder(input)
        return out.flatten(1)


def next_power_of_2(x: int) -> int:
    if x == 0:
        return 1

    return 2 ** math.ceil(math.log2(x))


def get_positional_representation_class(
    position_model_config: Literal["encode", "embed"]
) -> Union[Type["PositionalEncoding"], Type["PositionalEmbedding"]]:
    if position_model_config == "encode":
        return PositionalEncoding
    elif position_model_config == "embed":
        return PositionalEmbedding
    raise ValueError(
        "Unknown value for positional representation. "
        "Expected 'encode' or 'embed' but got '%s'.",
        position_model_config,
    )


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = max_length

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_length, embedding_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : self.max_length, :]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_length: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = max_length

        self.embedding = torch.nn.Parameter(
            data=torch.randn(1, max_length, embedding_dim), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.embedding[:, : self.max_length, :]
        return self.dropout(x)
