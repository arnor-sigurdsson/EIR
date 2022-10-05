import pytest

from eir.models import layers
from eir.models.model_training_utils import trace_eir_model
from eir.models.omics import models_cnn
from eir.setup.input_setup import DataDimensions
from tests.test_models.model_testing_utils import prepare_example_batch


def test_make_conv_layers():
    """
    Check that:
        - We have correct number of layers (+2 for first conv layer and
          self attn).
        - We start with correct block (first conv layer).
        - Self attention is the second last layer.

    """
    conv_layer_list = [1, 1, 1, 1]
    test_model_config = models_cnn.CNNModelConfig(
        kernel_width=5,
        layers=None,
        fc_repr_dim=512,
        down_stride=4,
        rb_do=0.1,
        first_kernel_expansion=1,
        first_stride_expansion=5,
        first_channel_expansion=1,
        dilation_factor=1,
        channel_exp_base=5,
        sa=True,
    )
    test_data_dimensions = DataDimensions(channels=1, height=4, width=int(8e5))
    conv_layers = models_cnn._make_conv_layers(
        residual_blocks=conv_layer_list,
        cnn_model_configuration=test_model_config,
        data_dimensions=test_data_dimensions,
    )

    # account for first block, add +2 instead if using SA
    assert len(conv_layers) == len(conv_layer_list) + 2
    assert isinstance(conv_layers[0], layers.FirstCNNBlock)
    assert isinstance(conv_layers[-2], layers.SelfAttention)


def get_test_module_dict_data():
    test_classes_dict = {"Origin": 10, "Height": 1, "BMI": 1}
    test_fc_in = 128

    return test_classes_dict, test_fc_in


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"dilation_factor": 2, "block_number": 1, "width": 1000}, 2),
        ({"dilation_factor": 4, "block_number": 2, "width": 1000}, 16),
        ({"dilation_factor": 5, "block_number": 5, "width": 100}, 25),
    ],
)
def test_get_cur_dilation(test_input, expected):
    test_dilation = models_cnn._get_cur_dilation(**test_input)

    assert test_dilation == expected


@pytest.mark.parametrize(
    "create_test_data", [{"task_type": "multi_task"}], indirect=True
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: MLP
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-03},
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
            },
        },
        # Case 2: CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-03,
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
                    },
                ],
            },
        },
        # Case 3: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "identity"},
                    },
                ],
                "fusion_configs": {
                    "model_type": "identity",
                },
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
        # Case 4: Check that we add and use extra inputs.
        {
            "injections": {
                "global_configs": {
                    "output_folder": "extra_inputs",
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
        # Case 5: Normal multi task with CNN
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "channel_exp_base": 5,
                                "rb_do": 0.15,
                                "fc_repr_dim": 64,
                                "l1": 1e-03,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {"fc_task_dim": 64, "rb_do": 0.10, "fc_do": 0.10},
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 6:  Normal multi task with MLP, note we have to reduce the LR for
        # stability and add L1 for regularization
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "linear",
                            "model_init_config": {"l1": 1e-03},
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {"fc_task_dim": 64, "rb_do": 0.10, "fc_do": 0.10},
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 7: Using the Simple LCL model
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "mlp-split",
                            "model_init_config": {
                                "fc_repr_dim": 8,
                                "split_mlp_num_splits": 64,
                                "l1": 1e-03,
                            },
                        },
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 8: Using the GLN
        {
            "injections": {
                "global_configs": {
                    "lr": 1e-03,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 1e-03,
                                "rb_do": 0.20,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.20,
                        "rb_do": 0.20,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 9: Using the MGMoE fusion
        {
            "injections": {
                "global_configs": {
                    "output_folder": "mgmoe",
                    "lr": 1e-03,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 1e-03,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_type": "mgmoe",
                    "model_config": {"mg_num_experts": 3},
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 10: Using the GLN with mixing
        {
            "injections": {
                "global_configs": {
                    "output_folder": "mixing_multi",
                    "lr": 1e-03,
                    "mixing_alpha": 0.5,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {
                            "mixing_subtype": "cutmix-uniform",
                        },
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 1e-03,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.10,
                        "rb_do": 0.10,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
        # Case 11: Using the GLN with limited activations
        {
            "injections": {
                "global_configs": {
                    "output_folder": "limited_activations",
                    "lr": 1e-03,
                    "max_acts_per_class": 100,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {
                            "model_type": "genome-local-net",
                            "model_init_config": {
                                "kernel_width": 8,
                                "channel_exp_base": 2,
                                "l1": 1e-03,
                                "rb_do": 0.20,
                            },
                        },
                    },
                ],
                "fusion_configs": {
                    "model_config": {
                        "fc_task_dim": 64,
                        "fc_do": 0.20,
                        "rb_do": 0.20,
                    },
                },
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height", "ExtraTarget"],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_omics_models(
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
