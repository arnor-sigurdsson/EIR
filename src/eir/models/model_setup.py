from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Type, cast

import torch
from torch import nn
from torch.nn import functional as F

from eir.experiment_io.input_object_io_modules.sequence_input_io import (
    load_sequence_input_object,
)
from eir.models.input.sequence.transformer_models import TransformerWrapperModel
from eir.models.model_setup_modules.meta_setup import (
    MetaClassGetterCallable,
    al_meta_model,
    get_default_meta_class,
    get_meta_model_class_and_kwargs_from_configs,
)
from eir.models.model_setup_modules.model_io import load_model, strip_orig_mod_prefix
from eir.models.model_setup_modules.pretrained_setup import (
    overload_meta_model_feature_extractors_with_pretrained,
)
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.train_utils.distributed import maybe_make_model_distributed
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.output_setup import al_output_objects_as_dict

al_model_registry = Dict[str, Callable[[str], Type[nn.Module]]]

logger = get_logger(name=__name__)


def get_model(
    global_config: schemas.GlobalConfig,
    inputs_as_dict: al_input_objects_as_dict,
    fusion_config: schemas.FusionConfig,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: MetaClassGetterCallable = get_default_meta_class,
) -> al_meta_model:
    meta_class, meta_kwargs = get_meta_model_class_and_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        meta_class_getter=meta_class_getter,
    )

    if global_config.m.pretrained_checkpoint:
        logger.info(
            "Loading pretrained checkpoint from '%s'.",
            global_config.m.pretrained_checkpoint,
        )
        loaded_meta_model = load_model(
            model_path=Path(global_config.m.pretrained_checkpoint),
            model_class=meta_class,
            model_init_kwargs=meta_kwargs,
            device=global_config.be.device,
            test_mode=False,
            strict_shapes=global_config.m.strict_pretrained_loading,
        )

        loaded_meta_model = overload_embeddings_with_pretrained(
            model=loaded_meta_model,
            inputs=inputs_as_dict,
            device=global_config.be.device,
            pretrained_checkpoint=global_config.m.pretrained_checkpoint,
        )

        return loaded_meta_model

    input_modules = overload_meta_model_feature_extractors_with_pretrained(
        input_modules=meta_kwargs["input_modules"],
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        meta_class_getter=meta_class_getter,
    )
    meta_kwargs["input_modules"] = input_modules

    meta_model = meta_class(**meta_kwargs)
    device = torch.device(device=global_config.be.device)
    meta_model = meta_model.to(device=device)

    meta_model = maybe_make_model_distributed(
        device=global_config.be.device,
        model=meta_model,
    )

    compiled_model: al_meta_model
    if global_config.m.compile_model:
        compiled_model = cast(al_meta_model, torch.compile(model=meta_model))
    else:
        compiled_model = meta_model

    return compiled_model


def overload_embeddings_with_pretrained(
    model: "al_meta_model",
    inputs: "al_input_objects_as_dict",
    device: str,
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
    serialized_inputs_folder = (
        run_folder / "serializations/sequence_input_serializations"
    )

    any_serialized_sequence_inputs = serialized_inputs_folder.exists()
    if not any_serialized_sequence_inputs:
        return model

    input_objects_loaded = {}
    for serialized_input_folder in serialized_inputs_folder.iterdir():
        input_name = serialized_input_folder.stem
        serialized_input = load_sequence_input_object(
            serialized_input_folder=serialized_input_folder
        )
        input_objects_loaded[input_name] = serialized_input

    loaded_state_dict = torch.load(
        f=pretrained_checkpoint,
        map_location=device,
        weights_only=True,
    )
    loaded_state_dict = strip_orig_mod_prefix(state_dict=loaded_state_dict)

    for input_name, input_object in inputs.items():
        input_type = input_object.input_config.input_info.input_type
        if input_type != "sequence":
            continue

        assert isinstance(input_object, ComputedSequenceInputInfo)
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
