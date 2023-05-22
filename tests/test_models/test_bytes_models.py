import pytest

from eir.models.model_training_utils import trace_eir_model
from tests.test_models.model_testing_utils import prepare_example_batch


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Basic Transformer
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_classification_vanilla_transformer_bytes",
                    "n_epochs": 12,
                    "memory_dataset": True,
                    "mixing_alpha": 1.0,
                    "attribution_background_samples": 8,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_bytes"},
                        "model_config": {
                            "position": "embed",
                            "window_size": 64,
                            "model_type": "sequence-default",
                            "model_init_config": {
                                "num_heads": 2,
                                "num_layers": 2,
                                "dropout": 0.10,
                            },
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
                        "model_config": {"model_init_config": {"fc_task_dim": 32}},
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_bytes_models(
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
    _ = trace_eir_model(meta_model=model, example_inputs=example_batch.inputs)
