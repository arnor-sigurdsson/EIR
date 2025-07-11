import pytest
import torch

from eir.models.model_setup import get_model
from eir.models.model_setup_modules.meta_setup import al_meta_model
from eir.setup import config, input_setup
from eir.setup.output_setup import set_up_outputs_for_training
from eir.train_utils.optim import maybe_wrap_model_with_swa
from eir.utils.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture()
def create_test_model(
    create_test_config: config.Configs,
    create_test_labels,
) -> al_meta_model:
    gc = create_test_config.global_config
    target_labels = create_test_labels

    train_ids = tuple(create_test_labels.train_labels["ID"])
    valid_ids = tuple(create_test_labels.valid_labels["ID"])

    inputs_as_dict = input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=target_labels.label_transformers,
    )

    input_model_names = [
        input_config.model_config.model_type
        for input_config in create_test_config.input_configs
    ]
    try:
        logger.info(f"=====Setting up input models: {input_model_names}=====")
        model = get_model(
            inputs_as_dict=inputs_as_dict,
            fusion_config=create_test_config.fusion_config,
            outputs_as_dict=outputs_as_dict,
            global_config=gc,
        )
    except Exception as e:
        logger.error(
            f"====Failed to setup model. Input model names: {input_model_names}===="
            f"due to {e}"
        )
        raise

    model = maybe_wrap_model_with_swa(
        n_iter_before_swa=gc.m.n_iter_before_swa,
        model=model,
        device=torch.device(gc.be.device),
    )

    return model
