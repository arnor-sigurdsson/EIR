import pytest
import torch

from tests.test_models.model_testing_utils import (
    check_eir_model,
    prepare_example_test_batch,
)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
            ),
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "random_samples_dropped_from_modalities": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "multi_task_multi_modal",
                        "n_epochs": 6,
                    },
                    "attribution_analysis": {"attribution_background_samples": 8},
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-04},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
                    },
                    {
                        "input_info": {"input_name": "test_image"},
                        "model_config": {
                            "model_init_config": {
                                "layers": [2],
                                "kernel_width": 2,
                                "kernel_height": 2,
                                "down_stride_width": 2,
                                "down_stride_height": 2,
                            },
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {
                            "model_type": "tabular",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "fusion_configs": {
                    "model_type": "mlp-residual",
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        }
    ],
    indirect=True,
)
def test_multi_modal_multi_task(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
    create_test_model,
    create_test_labels,
):
    model = create_test_model

    example_batch = prepare_example_test_batch(
        configs=create_test_config,
        labels=create_test_labels,
        model=model,
    )

    model.eval()
    with torch.no_grad():
        check_eir_model(meta_model=model, example_inputs=example_batch.inputs)
