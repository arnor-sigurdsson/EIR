from typing import Dict, Sequence
import warnings


import pytest
import torch

from eir.models.model_training_utils import check_eir_model
from eir.setup.setup_utils import get_all_hf_model_names
from tests.conftest import should_skip_in_gha
from tests.test_modelling.test_tabular_modelling.test_sequence_modelling import (
    _get_common_model_config_overload,
    _parse_model_specific_config_values,
)
from tests.test_models.model_testing_utils import prepare_example_batch


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("sequence",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Positional Encoding
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "encode"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
        # Case 2: Classification - Positional Embedding and Windowed
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification",
                    "n_epochs": 12,
                    "memory_dataset": True,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"window_size": 16, "position": "embed"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_internal_sequence_models(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
    create_test_model,
    create_test_labels,
):
    model = create_test_model

    example_batch = prepare_example_batch(
        configs=create_test_config, labels=create_test_labels, model=model
    )

    model.eval()
    check_eir_model(meta_model=model, example_inputs=example_batch.inputs)


def get_test_external_sequence_models_parametrization() -> Sequence[Dict]:
    """
    Some models seem to fail tracing or a bit troublesome to configure, skip for now.
    """

    all_models = get_all_hf_model_names()
    all_models_filtered = []

    for model in all_models:
        if any(i for i in {"big_bird", "bigbird", "gpt_neo", "reformer"} if i in model):
            continue
        else:
            all_models_filtered.append(model)

    all_parametrizations = []

    for model_type in all_models_filtered:
        model_init_config = _get_common_model_config_overload()
        model_init_config_parsed = _parse_model_specific_config_values(
            model_config=model_init_config, model_name=model_type
        )

        cur_params = {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {
                            "input_name": "test_sequence",
                        },
                        "input_type_info": {"max_length": 64},
                        "model_config": {
                            "model_type": model_type,
                            "window_size": 64,
                            "pretrained_model": False,
                            "model_init_config": model_init_config_parsed,
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            }
        }
        all_parametrizations.append(cur_params)

    return all_parametrizations


@pytest.mark.skipif(condition=should_skip_in_gha(), reason="In GHA.")
@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("sequence",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    get_test_external_sequence_models_parametrization(),
    indirect=True,
)
def test_external_sequence_models(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
    create_test_model,
    create_test_labels,
):
    model = create_test_model

    example_batch = prepare_example_batch(
        configs=create_test_config, labels=create_test_labels, model=model
    )

    model.eval()
    model_name = create_test_config.input_configs[0].model_config.model_type
    with torch.no_grad():
        model(inputs=example_batch.inputs)
        if model_name in get_models_to_skip_test_trace():
            return
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            check_eir_model(meta_model=model, example_inputs=example_batch.inputs)


def get_models_to_skip_test_trace() -> Sequence[str]:
    """Skip randomly failing models."""
    return ["git"]
