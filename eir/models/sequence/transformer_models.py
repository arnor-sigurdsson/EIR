import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import (
    Union,
    Literal,
    Type,
    Tuple,
    Dict,
    Sequence,
    Callable,
)

import torch
from aislib.misc_utils import get_logger
from perceiver_pytorch import PerceiverIO
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad
from transformers import PreTrainedModel, PretrainedConfig

from eir.setup.setup_utils import get_all_hf_model_names
from eir.models.layers import _find_split_padding_needed


al_sequence_models = tuple(
    Literal[i]
    for i in ["sequence-default", "perceiver"] + list(get_all_hf_model_names())
)

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class SequenceModelConfig:

    """
    :param model_init_config:
          Configuration / arguments used to initialise model.

    :param model_type:
         Which type of image model to use.

    :param embedding_dim:
        Which dimension to use for the embeddings. If ``None``, will automatically set
        this value based on the number of tokens and attention heads.

    :param position:
        Whether to encode the token position or use learnable position embeddings.

    :param position_dropoput:
        Dropout for the positional encoding / embedding.

    :param window_size:
        If set to more than 0, will apply a sliding window of feature
        extraction over the input, meaning the model (e.g. transformer) will only
        see a part of the input at a time. Can be Useful to avoid the O(n²)
        complexity of transformers, as it becomes O(window_size² * n_windows) instead.

    :param pool:
        Whether and how to pool (max / avg) the final feature maps before being
        passed to the final fusion module / predictor. Meaning we pool over the
        sequence (i.e. time) dimension, so the resulting dimensions is embedding_dim
        instead of sequence_length * embedding_dim. If using windowed / conv
        transformers, this becomes embedding_dim * number_of_chunks.

    :param pretrained_model:
          Specify whether the model type is assumed to be pretrained and from the
          Pytorch Image Models repository.

    :param freeze_pretrained_model:
          Whether to freeze the pretrained model weights.

    """

    model_init_config: Union["BasicTransformerFeatureExtractorModelConfig", Dict]

    model_type: al_sequence_models = "sequence-default"

    embedding_dim: int = 64

    position: Literal["encode", "embed"] = "encode"
    position_dropout: float = 0.10
    window_size: int = 0
    pool: Union[Literal["avg"], Literal["max"], None] = None

    pretrained_model: bool = False
    freeze_pretrained_model: bool = False


class TransformerWrapperModel(nn.Module):
    def __init__(
        self,
        feature_extractor: Union["TransformerFeatureExtractor", nn.Module],
        model_config: SequenceModelConfig,
        embedding_dim: int,
        num_tokens: int,
        max_length: int,
        external_feature_extractor: bool,
        device: str,
        embeddings: nn.Embedding = None,
        pre_computed_num_out_features: int = 0,
    ) -> None:

        super().__init__()
        self.model_config = model_config
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.max_length = max_length
        self.external_feature_extractor = external_feature_extractor
        self.pre_computed_num_out_features = pre_computed_num_out_features

        pos_repr_class = get_positional_representation_class(
            position_model_config=self.model_config.position
        )
        self.pos_representation = pos_repr_class(
            embedding_dim=self.embedding_dim,
            dropout=self.model_config.position_dropout,
            max_length=self.max_length,
        )

        if embeddings:
            self.embedding = embeddings
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.num_tokens, embedding_dim=self.embedding_dim
            )

        self.feature_extractor = feature_extractor

        if not embeddings:
            self.init_embedding_weights()

        (
            self.dynamic_extras,
            self.extract_features,
        ) = _get_transformer_wrapper_feature_extractor(
            feature_extractor=self.feature_extractor,
            window_size=self.model_config.window_size,
            max_length=self.max_length,
            embedding_dim=self.embedding_dim,
            pool=self.model_config.pool,
            device=device,
            external_feature_extractor=self.external_feature_extractor,
        )

    @property
    def num_out_features(self) -> int:

        if self.pre_computed_num_out_features:
            return self.pre_computed_num_out_features

        padding = self.dynamic_extras.get("padding", 0)
        length_with_padding = self.max_length + padding

        if self.model_config.pool in ("avg", "max"):

            num_chunks = 1
            if self.model_config.window_size:
                num_chunks = length_with_padding // self.model_config.window_size

            return self.embedding_dim * num_chunks

        return length_with_padding * self.embedding_dim

    def script_submodules_for_tracing(self):
        self.embedding = torch.jit.script(self.embedding)

    def init_embedding_weights(self) -> None:
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
        auto_emb_dim = math.ceil((int(num_tokens**0.25) / num_heads)) * num_heads
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
    feature_extractor: Union["TransformerFeatureExtractor", nn.Module],
    external_feature_extractor: bool,
    window_size: int,
    embedding_dim: int,
    max_length: int,
    device: str,
    pool: Union[Literal["avg"], Literal["max"], None] = None,
) -> Tuple[Dict[str, int], Callable[[torch.Tensor], torch.Tensor]]:

    dynamic_extras = {"padding": 0}

    feature_extractor_forward = _get_feature_extractor_forward(
        is_hf_model=external_feature_extractor,
        feature_extractor=feature_extractor,
        input_length=window_size if window_size else max_length,
        embedding_size=embedding_dim,
        device=device,
        pool=pool,
    )
    if not window_size:
        extractor = partial(
            feature_extractor_forward, feature_extractor=feature_extractor
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
            feature_extractor_forward_callable=feature_extractor_forward,
            max_length=max_length,
            window_size=window_size,
            padding=padding,
        )

    return dynamic_extras, extractor


def _get_feature_extractor_forward(
    is_hf_model: bool,
    feature_extractor: Union[nn.Module, PreTrainedModel],
    input_length: int,
    embedding_size: int,
    device: str,
    pool: Union[Literal["avg"], Literal["max"], None] = None,
) -> Callable[
    [torch.Tensor, Union["TransformerFeatureExtractor", nn.Module]], torch.Tensor
]:
    if is_hf_model:
        return get_hf_transformer_forward(
            feature_extractor_=feature_extractor,
            input_length=input_length,
            embedding_dim=embedding_size,
            device=device,
            pool=pool,
        )
    return _get_simple_transformer_forward(pool=pool)


def _get_simple_transformer_forward(
    pool: Union[Literal["avg"], Literal["max"], None] = None,
):
    pooling_func = _get_sequence_pooling_func(pool=pool)

    def _simple_transformer_forward(
        input: torch.Tensor,
        feature_extractor: "TransformerFeatureExtractor",
    ) -> torch.Tensor:

        tensor_out = feature_extractor(input)
        tensor_pooled = pooling_func(input=tensor_out)
        final_out = tensor_pooled.flatten(1)

        return final_out

    return _simple_transformer_forward


def get_hf_transformer_forward(
    feature_extractor_: PreTrainedModel,
    input_length: int,
    embedding_dim: int,
    device: str,
    pool: Union[Literal["avg"], Literal["max"], None] = None,
):

    forward_argnames = inspect.getfullargspec(feature_extractor_.forward)[0]

    bound_kwargs = _build_transformer_forward_kwargs(
        forward_argnames=forward_argnames,
        config=feature_extractor_.config,
        input_length=input_length,
        embedding_dim=embedding_dim,
        device=device,
    )

    pooling_func = _get_sequence_pooling_func(pool=pool)

    def _hf_transformer_forward(
        input: torch.Tensor,
        feature_extractor: nn.Module,
        key: str = "last_hidden_state",
    ) -> torch.Tensor:
        hf_transformer_out = feature_extractor(inputs_embeds=input, **bound_kwargs)
        tensor_out = getattr(hf_transformer_out, key)
        tensor_pooled = pooling_func(input=tensor_out)
        final_out = tensor_pooled.flatten(1)
        return final_out

    return _hf_transformer_forward


def _get_sequence_pooling_func(
    pool: Union[Literal["avg"], Literal["max"], None]
) -> Callable:
    def _identity(input: torch.Tensor) -> torch.Tensor:
        return input

    def _max(input: torch.Tensor) -> torch.Tensor:
        return input.max(dim=1)[0]

    def _avg(input: torch.Tensor) -> torch.Tensor:
        return input.mean(dim=1)

    if pool is None:
        return _identity

    elif pool == "max":
        return _max

    elif pool == "avg":
        return _avg

    raise ValueError()


def _build_transformer_forward_kwargs(
    forward_argnames: Sequence[str],
    config: PretrainedConfig,
    input_length: int,
    embedding_dim: int,
    device: str,
) -> Dict:
    """
    TODO: Deprecate.
    """
    kwargs = {}

    if "decoder_inputs_embeds" in forward_argnames:
        kwargs["decoder_inputs_embeds"] = torch.randn(1, input_length, embedding_dim)

    return kwargs


def _conv_transfomer_forward(
    input: torch.Tensor,
    feature_extractor: "TransformerFeatureExtractor",
    feature_extractor_forward_callable: Callable[
        [torch.Tensor, Callable], torch.Tensor
    ],
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
        cur_out = feature_extractor_forward_callable(cur_input, feature_extractor)

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

    :param dim_feedforward:
        The dimension of the feedforward network model

    :param dropout:
         Dropout value to use in the encoder layers.
    """

    num_heads: int = 8
    num_layers: int = 2
    dim_feedforward: Union[int, Literal["auto"]] = "auto"
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

        dim_feed_forward = parse_dim_feedforward(
            dim_feedforward=model_config.dim_feedforward,
            embedding_dim=self.embedding_dim,
        )

        encoder_layer_base = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=model_config.num_heads,
            dim_feedforward=dim_feed_forward,
            dropout=model_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer_base, num_layers=model_config.num_layers
        )

    @property
    def num_out_features(self) -> int:
        return self.max_length * self.embedding_dim

    def forward(self, input: Tensor) -> Tensor:
        out = self.transformer_encoder(input)
        return out


def next_power_of_2(x: int) -> int:
    if x == 0:
        return 1

    return 2 ** math.ceil(math.log2(x))


def parse_dim_feedforward(
    dim_feedforward: Union[int, Literal["auto"]], embedding_dim: int
) -> int:
    if dim_feedforward == "auto":
        dim_feedforward = embedding_dim * 4
        logger.info(
            "Setting dim_feedfoward to %d based on %d and 'auto' option.",
            dim_feedforward,
            embedding_dim,
        )

    return dim_feedforward


@dataclass
class PerceiverIOModelConfig:
    """ """

    depth: int = 2
    dim: int = 16
    queries_dim: int = 32
    logits_dim: Union[int, None] = None
    num_latents: int = 32
    latent_dim: int = 128
    cross_heads: int = 1
    latent_heads: int = 8
    cross_dim_head: int = 64
    latent_dim_head: int = 64
    weight_tie_layers: bool = False
    decoder_ff: bool = False


class PerceiverIOFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_config: PerceiverIOModelConfig,
        max_length: int,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.max_length = max_length

        self.perceiver = PerceiverIO(**model_config.__dict__)

    @property
    def num_out_features(self) -> int:
        mc = self.model_config
        if mc.logits_dim:
            return mc.logits_dim * self.max_length

        return mc.latent_dim * mc.num_latents

    def forward(self, input: Tensor) -> Tensor:
        out = self.perceiver(data=input)
        return out


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
