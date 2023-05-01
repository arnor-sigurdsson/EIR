import pytest
from torch import nn
import torch

import eir.setup
from eir.models.model_setup import get_default_model_registry_per_input_type, get_model
from eir.setup import config
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train_utils.optim import maybe_wrap_model_with_swa


@pytest.fixture()
def create_test_model(
    create_test_config: config.Configs, create_test_labels
) -> nn.Module:
    gc = create_test_config.global_config
    target_labels = create_test_labels

    inputs_as_dict = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=tuple(create_test_labels.train_labels.keys()),
        valid_ids=tuple(create_test_labels.valid_labels.keys()),
        hooks=None,
    )

    input_model_registry = get_default_model_registry_per_input_type()

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    model = get_model(
        inputs_as_dict=inputs_as_dict,
        model_registry_per_input_type=input_model_registry,
        model_registry_per_output_type={},
        fusion_config=create_test_config.fusion_config,
        outputs_as_dict=outputs_as_dict,
        global_config=gc,
    )

    model = maybe_wrap_model_with_swa(
        n_iter_before_swa=gc.n_iter_before_swa,
        model=model,
        device=torch.device(gc.device),
    )

    return model
