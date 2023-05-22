from dataclasses import dataclass
from pathlib import Path
import dill
from typing import Dict, Type, TYPE_CHECKING

import torch
from torch import nn
from aislib.misc_utils import get_logger
from eir.models.sequence.transformer_models import (
    BasicTransformerFeatureExtractorModelConfig,
    parse_dim_feedforward,
)
from eir.models.fusion.fusion_attention import MetaSequenceProjection

if TYPE_CHECKING:
    from eir.setup.output_setup_modules.sequence_output_setup import (
        ComputedSequenceOutputInfo,
    )
    from eir.setup.input_setup import al_input_objects_as_dict
    from eir.setup.input_setup_modules.common import DataDimensions
    from eir.setup.schemas import GlobalConfig


logger = get_logger(name=__name__)


def sequence_model_registry_output(model_type: str) -> Type[nn.Module]:
    match model_type:
        case "sequence" | "eir-sequence-output-linked-default":
            return SequenceOutputModule
        case _:
            raise ValueError(f"Unknown model type {model_type}")


@dataclass
class TransformerSequenceOutputModuleConfig(
    BasicTransformerFeatureExtractorModelConfig
):
    pass


class SequenceOutputModule(nn.Module):
    def __init__(
        self,
        output_object: "ComputedSequenceOutputInfo",
        output_name: str,
        in_features_per_feature_extractor: Dict[str, "DataDimensions"],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.num_tokens = len(output_object.vocab)
        self.input_dimensions = in_features_per_feature_extractor
        self.embedding_dim = output_object.embedding_dim
        self.max_length = output_object.computed_max_length
        self.output_name = output_name
        self.output_model_config = (
            output_object.output_config.model_config.model_init_config
        )

        dim_feed_forward = parse_dim_feedforward(
            dim_feedforward=self.output_model_config.dim_feedforward,
            embedding_dim=self.embedding_dim,
        )

        transformer_output_layer_base = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.output_model_config.num_heads,
            dim_feedforward=dim_feed_forward,
            dropout=self.output_model_config.dropout,
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
            num_layers=self.output_model_config.num_layers,
        )

        self.match_projections = nn.ModuleDict()
        # TODO: Add input type here
        for input_name, out_features in in_features_per_feature_extractor.items():
            if input_name == self.output_name:
                continue

            cur_projection = MetaSequenceProjection(
                in_total_num_elements=out_features.num_elements(),
                in_embedding_dim=out_features.width,
                target_embedding_dim=self.embedding_dim,
                target_max_length=self.max_length,
            )

            self.match_projections[input_name] = cur_projection

        self.head = nn.Linear(self.embedding_dim, self.num_tokens)

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def _find_nearest_multiple(base: int, target: int) -> int:
    return base * round(target / base)


def overload_embeddings_with_pretrained(
    model: nn.Module,
    inputs: "al_input_objects_as_dict",
    global_config: "GlobalConfig",
) -> nn.Module:
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

    pretrained_checkpoint = global_config.pretrained_checkpoint

    if not pretrained_checkpoint:
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
            input_object = dill.load(f)
        input_objects_loaded[input_name] = input_object

    loaded_state_dict = torch.load(pretrained_checkpoint)

    for input_name, input_object in inputs.items():
        cur_vocab = input_object.vocab.get_stoi()
        prev_input_object_vocab = input_objects_loaded[input_name].vocab.get_stoi()

        prev_emb_key = f"input_modules.{input_name}.embedding.weight"
        prev_embeddings = loaded_state_dict[prev_emb_key]

        cur_embedding = model.input_modules[input_name].embedding.weight

        for token, idx in cur_vocab.items():
            if token not in prev_input_object_vocab:
                continue
            prev_idx = prev_input_object_vocab[token]
            cur_embedding[idx] = prev_embeddings[prev_idx]

        logger.info(f"Overloaded embeddings for {input_name}.")

    return model
