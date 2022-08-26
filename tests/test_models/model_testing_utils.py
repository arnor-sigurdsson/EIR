import torch
from torch.utils.data._utils.collate import default_collate

from eir.data_load.data_utils import Batch
from eir.data_load.datasets import impute_missing_modalities_wrapper
from eir.data_load.label_setup import Labels
from eir.setup.config import Configs
from eir.setup.input_setup import set_up_inputs_for_training
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train import prepare_base_batch_default


def prepare_example_batch(
    configs: Configs, labels: Labels, model: torch.nn.Module
) -> Batch:

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=tuple(labels.train_labels.keys()),
        valid_ids=tuple(labels.valid_labels.keys()),
        hooks=None,
    )

    imputed_inputs = impute_missing_modalities_wrapper(
        inputs_values={}, inputs_objects=inputs_as_dict
    )

    loader_batch = (imputed_inputs, {}, list())
    loader_batch_collated = default_collate([loader_batch])

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=configs.output_configs,
        target_transformers=labels.label_transformers,
    )

    batch = prepare_base_batch_default(
        loader_batch=loader_batch_collated,
        input_objects=inputs_as_dict,
        output_objects=outputs_as_dict,
        model=model,
        device=configs.global_config.device,
    )

    return batch
