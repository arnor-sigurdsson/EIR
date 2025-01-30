from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_preparation_modules.imputation import (
    impute_missing_modalities_wrapper,
)
from eir.data_load.data_utils import Batch
from eir.data_load.label_setup import Labels
from eir.models.model_setup_modules.meta_setup import al_meta_model
from eir.setup.config import Configs
from eir.setup.input_setup import set_up_inputs_for_training
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train_utils.step_logic import prepare_base_batch_default


def check_eir_model(
    meta_model: "al_meta_model", example_inputs: dict[str, Any]
) -> None:
    with torch.inference_mode():
        meta_model(inputs=example_inputs)


def prepare_example_test_batch(
    configs: Configs, labels: Labels, model: torch.nn.Module, batch_size: int = 2
) -> Batch:
    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=tuple(labels.train_labels["ID"]),
        valid_ids=tuple(labels.valid_labels["ID"]),
        hooks=None,
    )

    imputed_inputs = impute_missing_modalities_wrapper(
        inputs_values={}, inputs_objects=inputs_as_dict
    )

    loader_batch = (imputed_inputs, {}, [])
    batch_as_list = [loader_batch] * batch_size
    loader_batch_collated = default_collate(batch=batch_as_list)

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=configs.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=labels.label_transformers,
    )

    batch = prepare_base_batch_default(
        loader_batch=loader_batch_collated,
        input_objects=inputs_as_dict,
        output_objects=outputs_as_dict,
        model=model,
        device=configs.gc.be.device,
    )

    return batch
