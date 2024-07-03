from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal

import dill
import torch
from torch import nn
from torch.nn import functional as F

from eir.models.fusion.fusion_attention import MetaSequenceProjection
from eir.models.input.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
    TransformerWrapperModel,
    parse_dim_feedforward,
)
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import (
        FeatureExtractorInfo,
        al_meta_model,
    )
    from eir.setup.input_setup import al_input_objects_as_dict
    from eir.setup.output_setup_modules.sequence_output_setup import (
        ComputedSequenceOutputInfo,
    )

al_sequence_output_models = Literal["sequence"]

logger = get_logger(name=__name__)


@dataclass
class TransformerSequenceOutputModuleConfig(
    BasicTransformerFeatureExtractorModelConfig
):
    pass


@dataclass
class SequenceOutputModuleConfig:
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

    :param position_dropout:
        Dropout for the positional encoding / embedding.

    """

    model_init_config: TransformerSequenceOutputModuleConfig
    model_type: Literal["sequence"] = "sequence"

    embedding_dim: int = 64

    position: Literal["encode", "embed"] = "encode"
    position_dropout: float = 0.10

    projection_layer_type: Literal["auto", "lcl", "lcl_residual", "linear"] = "auto"


class SequenceOutputModule(nn.Module):
    def __init__(
        self,
        output_object: "ComputedSequenceOutputInfo",
        output_name: str,
        feature_dimensionalities_and_types: Dict[str, "FeatureExtractorInfo"],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_tokens = len(output_object.vocab)
        self.input_dimensions = feature_dimensionalities_and_types
        self.embedding_dim = output_object.embedding_dim
        self.max_length = output_object.computed_max_length
        self.output_name = output_name
        self.output_model_config = output_object.output_config.model_config
        assert isinstance(self.output_model_config, SequenceOutputModuleConfig)
        self.output_model_init_config = self.output_model_config.model_init_config
        assert isinstance(
            self.output_model_init_config, TransformerSequenceOutputModuleConfig
        )

        dim_feed_forward = parse_dim_feedforward(
            dim_feedforward=self.output_model_init_config.dim_feedforward,
            embedding_dim=self.embedding_dim,
        )

        transformer_output_layer_base = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.output_model_init_config.num_heads,
            dim_feedforward=dim_feed_forward,
            dropout=self.output_model_init_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        mask = torch.triu(
            torch.ones(self.max_length, self.max_length) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("mask", mask)

        self.output_transformer = nn.TransformerEncoder(
            encoder_layer=transformer_output_layer_base,
            num_layers=self.output_model_init_config.num_layers,
            enable_nested_tensor=False,
        )

        self.match_projections = nn.ModuleDict()
        for input_name, feature_extractor_info in self.input_dimensions.items():
            if input_name == self.output_name:
                continue

            match feature_extractor_info.input_type:
                case "sequence":
                    in_embed = feature_extractor_info.input_dimension.width
                case _:
                    in_embed = feature_extractor_info.output_dimension

            in_elements = feature_extractor_info.output_dimension
            cur_projection = MetaSequenceProjection(
                in_total_num_elements=in_elements,
                in_embedding_dim=in_embed,
                target_embedding_dim=self.embedding_dim,
                target_max_length=self.max_length,
                projection_layer_type=self.output_model_config.projection_layer_type,
            )

            self.match_projections[input_name] = cur_projection

        self.head = nn.Linear(self.embedding_dim, self.num_tokens)

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = input[self.output_name]
        out = out.reshape(out.shape[0], self.max_length, self.embedding_dim)

        for input_name, input_tensor in input.items():
            if input_name == self.output_name:
                continue

            cur_projection = self.match_projections[input_name]
            projected = cur_projection(input_tensor=input_tensor, target_tensor=out)
            out = out + projected

        out = self.output_transformer(out, mask=self.mask)

        out = self.head(out)

        return {self.output_name: out}


def overload_embeddings_with_pretrained(
    model: "al_meta_model",
    inputs: "al_input_objects_as_dict",
    pretrained_checkpoint: str,
) -> "al_meta_model":
    """
    Vocab: From serialized input object
    Embeddings: From loaded model

    - If we have a pretrained checkpoint in global config, we have to initialize inputs
    from that experiment in order to get the vocab per input.

    - If we are using selected ones, we can just use the vocab from the input object
    here directly.

    In both cases, we have to load the pretrained model to grab the embeddings.
    Probably it's enough to just use the torch.load() function here, since it's just
    giving us a dictionary.

    First, let's just assume the global case.
    """

    if not pretrained_checkpoint:
        return model

    any_sequence_inputs = any(
        input_object.input_config.input_info.input_type == "sequence"
        for input_object in inputs.values()
    )
    if not any_sequence_inputs:
        return model

    logger.info(
        f"Overloading embeddings with pretrained checkpoint {pretrained_checkpoint}."
    )

    run_folder = Path(pretrained_checkpoint).parent.parent
    serialized_inputs = run_folder / "serializations/sequence_input_serializations"

    input_objects_loaded = {}
    for serialized_input in serialized_inputs.iterdir():
        input_name = serialized_input.stem
        with open(serialized_input, "rb") as f:
            input_object = dill.load(file=f)
        input_objects_loaded[input_name] = input_object

    loaded_state_dict = torch.load(f=pretrained_checkpoint)

    for input_name, input_object in inputs.items():
        input_type = input_object.input_config.input_info.input_type
        if input_type != "sequence":
            continue

        cur_vocab = input_object.vocab.get_stoi()
        prev_input_object_vocab = input_objects_loaded[input_name].vocab.get_stoi()

        prev_emb_key = f"input_modules.{input_name}.embedding.weight"
        prev_embeddings = loaded_state_dict[prev_emb_key]

        cur_input_module = model.input_modules[input_name]
        assert isinstance(cur_input_module, TransformerWrapperModel)

        cur_embedding = cur_input_module.embedding.weight
        cur_embedding_copy = cur_embedding.clone().detach()

        for token, idx in cur_vocab.items():
            if token not in prev_input_object_vocab:
                continue

            prev_idx = prev_input_object_vocab[token]
            prev_emb = prev_embeddings[prev_idx]

            cur_emb = cur_embedding_copy[idx]

            if prev_emb.shape != cur_emb.shape:
                logger.warning(
                    f"Shape mismatch for token {token} in input {input_name}."
                    f"Applying average pooling to match dimensions."
                )
                prev_emb = prev_emb.view(1, 1, -1)
                prev_emb = F.adaptive_avg_pool1d(
                    input=prev_emb,
                    output_size=cur_emb.shape[0],
                )
                prev_emb = prev_emb.view(-1)

            cur_embedding_copy[idx] = prev_emb

        cur_input_module.embedding.weight = nn.Parameter(
            data=cur_embedding_copy,
            requires_grad=True,
        )
        logger.info(f"Overloaded embeddings for {input_name}.")

    return model
