import copy
from collections.abc import Sequence

import pytest

from eir.models.model_setup_modules.input_model_setup.input_model_setup_sequence import (  # noqa
    _get_hf_sequence_feature_extractor_objects,
    _get_manual_out_features_for_external_feature_extractor,
)
from eir.setup.setup_utils import get_all_hf_model_names
from eir.utils.logging import get_logger
from tests.conftest import should_skip_in_gha
from tests.test_models.model_testing_utils import (
    check_eir_model,
    prepare_example_test_batch,
)

logger = get_logger(name=__name__)


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
                    "basic_experiment": {
                        "output_folder": "test_classification",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"position": "encode"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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
                    "basic_experiment": {
                        "output_folder": "test_classification",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_sequence"},
                        "model_config": {"window_size": 16, "position": "embed"},
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
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

    example_batch = prepare_example_test_batch(
        configs=create_test_config, labels=create_test_labels, model=model
    )

    model.eval()
    check_eir_model(meta_model=model, example_inputs=example_batch.inputs)


def _get_common_model_config_overload() -> dict:
    n_heads = 4
    n_layers = 2
    embedding_dim = 16

    config = {
        "num_hidden_layers": n_layers,
        "attention_types": [[["global"], 1]],
        "encoder_layers": n_layers,
        "num_encoder_layers": n_layers,
        "decoder_layers": n_layers,
        "num_decoder_layers": n_layers,
        "block_sizes": [2],
        "attention_head_size": 4,
        "embedding_size": embedding_dim,
        "rotary_dim": n_heads,
        "d_embed": embedding_dim,
        "hidden_size": n_heads * 4,
        "num_attention_heads": n_heads,
        "encoder_attention_heads": n_heads,
        "decoder_attention_heads": n_heads,
        "num_heads": n_heads,
        "num_key_value_heads": n_heads,
        "intermediate_size": 32,
        "d_model": 16,
        "pad_token_id": 0,
        "attention_window": 16,
        "axial_pos_embds_dim": (4, 12),
        "axial_pos_shape": (8, 8),
        "num_channels": 8,
        "num_groups": 4,
    }

    assert config["hidden_size"] % n_heads == 0

    return config


def _parse_model_specific_config_values(model_config: dict, model_name: str) -> dict:
    mc = copy.copy(model_config)

    if model_name == "prophetnet" or model_name == "xlm-prophetnet":
        mc.pop("num_hidden_layers")

    return mc


def get_test_external_sequence_models_parametrization() -> Sequence[dict]:
    """
    Some models seem to fail tracing or a bit troublesome to configure, skip for now.
    """

    all_models = get_all_hf_model_names()
    all_models_filtered = []

    for model in all_models:
        if any(i for i in {"big_bird", "bigbird", "gpt_neo", "reformer"} if i in model):
            continue

        # skip these large models for now
        if any(i for i in {"llama"} if i in model):
            continue

        all_models_filtered.append(model)

    all_parameterizations = []

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
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            }
        }
        all_parameterizations.append(cur_params)

    return all_parameterizations


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
def test_external_sequence_models_forward(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
    create_test_model,
    create_test_labels,
):
    """
    Note that some models, e.g. T5, T5-long and Switch Transformer
    seem to dislike batch size >1 when num_decoder_layers are >0.
    Possibly a configuration issue or bug in the model.
    """
    model = create_test_model

    model_name = create_test_config.input_configs[0].model_config.model_type
    logger.info(f"=====Testing model: {model_name}=====")

    example_batch = prepare_example_test_batch(
        configs=create_test_config,
        labels=create_test_labels,
        model=model,
        batch_size=1,
    )

    model.eval()
    check_eir_model(meta_model=model, example_inputs=example_batch.inputs)


@pytest.mark.skipif(condition=should_skip_in_gha(), reason="In GHA.")
@pytest.mark.parametrize(
    "model_name",
    get_all_hf_model_names(),
)
def test_external_nlp_feature_extractor_setup(model_name: str):
    model_config = _get_common_model_config_overload()
    model_config_parsed = _parse_model_specific_config_values(
        model_config=model_config, model_name=model_name
    )
    sequence_length = 64

    feature_extractor_objects = _get_hf_sequence_feature_extractor_objects(
        model_name=model_name,
        model_config=model_config_parsed,
        feature_extractor_max_length=sequence_length,
        num_chunks=1,
        pool=None,
    )
    _get_manual_out_features_for_external_feature_extractor(
        input_length=sequence_length,
        embedding_dim=feature_extractor_objects.embedding_dim,
        num_chunks=1,
        feature_extractor=feature_extractor_objects.feature_extractor,
        pool=None,
    )
