from pathlib import Path
from typing import (
    Callable,
    Dict,
    Type,
    TYPE_CHECKING,
)

import torch
from aislib.misc_utils import get_logger
from torch import nn

from eir.models import al_meta_model
from eir.models.model_setup_modules.meta_setup import (
    get_default_meta_class,
    get_meta_model_class_and_kwargs_from_configs,
)
from eir.models.model_setup_modules.model_io import load_model
from eir.models.model_setup_modules.pretrained_setup import (
    overload_fusion_model_feature_extractors_with_pretrained,
)
from eir.models.output.sequence.sequence_output_modules import (
    overload_embeddings_with_pretrained,
)
from eir.setup import schemas
from eir.setup.input_setup import al_input_objects_as_dict
from eir.train_utils.distributed import maybe_make_model_distributed

if TYPE_CHECKING:
    from eir.setup.output_setup import (
        al_output_objects_as_dict,
    )

al_fusion_class_callable = Callable[[str], Type[nn.Module]]
al_model_registry = Dict[str, Callable[[str], Type[nn.Module]]]

logger = get_logger(name=__name__)


def get_model(
    global_config: schemas.GlobalConfig,
    inputs_as_dict: al_input_objects_as_dict,
    fusion_config: schemas.FusionConfig,
    outputs_as_dict: "al_output_objects_as_dict",
    meta_class_getter: al_fusion_class_callable = get_default_meta_class,
) -> al_meta_model:
    meta_class, meta_kwargs = get_meta_model_class_and_kwargs_from_configs(
        global_config=global_config,
        fusion_config=fusion_config,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        meta_class_getter=meta_class_getter,
    )

    if global_config.pretrained_checkpoint:
        logger.info(
            "Loading pretrained checkpoint from '%s'.",
            global_config.pretrained_checkpoint,
        )
        loaded_meta_model = load_model(
            model_path=Path(global_config.pretrained_checkpoint),
            model_class=meta_class,
            model_init_kwargs=meta_kwargs,
            device=global_config.device,
            test_mode=False,
            strict_shapes=global_config.strict_pretrained_loading,
        )

        loaded_meta_model = overload_embeddings_with_pretrained(
            model=loaded_meta_model,
            inputs=inputs_as_dict,
            pretrained_checkpoint=global_config.pretrained_checkpoint,
        )

        return loaded_meta_model

    input_modules = overload_fusion_model_feature_extractors_with_pretrained(
        input_modules=meta_kwargs["input_modules"],
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        meta_class_getter=meta_class_getter,
    )
    meta_kwargs["input_modules"] = input_modules

    meta_model = meta_class(**meta_kwargs)
    meta_model = meta_model.to(device=global_config.device)

    meta_model = maybe_make_model_distributed(
        device=global_config.device, model=meta_model
    )

    if global_config.compile_model:
        meta_model = torch.compile(model=meta_model)

    return meta_model
