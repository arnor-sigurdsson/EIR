from argparse import Namespace

import pytest
import torch

from human_origins_supervised.models import models, layers


def test_make_conv_layers():
    """
    Check that:
        - We have correct number of layers (+2 for first conv layer and
          self attn).
        - We start with correct block (first conv layer).
        - Self attention is the second last layer.

    """
    conv_layer_list = [1, 1, 1, 1]
    test_cl_args = Namespace(
        kernel_width=5,
        down_stride=4,
        target_width=int(8e5),
        rb_do=0.1,
        first_kernel_expansion=1,
        first_stride_expansion=5,
        first_channel_expansion=1,
        dilation_factor=1,
        channel_exp_base=5,
        sa=True,
    )
    conv_layers = models._make_conv_layers(conv_layer_list, test_cl_args)

    # account for first block, add +2 instead if using SA
    assert len(conv_layers) == len(conv_layer_list) + 2
    assert isinstance(conv_layers[0], layers.FirstBlock)
    assert isinstance(conv_layers[-2], layers.SelfAttention)


def get_test_module_dict_data():
    test_classes_dict = {"Origin": 10, "Height": 1, "BMI": 1}
    test_fc_in = 128

    return test_classes_dict, test_fc_in


def test_get_module_dict_from_target_columns():
    test_classes_dict, test_fc_in = get_test_module_dict_data()

    output_layer_model_dict = models._get_module_dict_from_target_columns(
        num_classes=test_classes_dict, fc_in=test_fc_in
    )

    for target_column, num_classes in test_classes_dict.items():
        cur_module = output_layer_model_dict[target_column]
        assert cur_module.in_features == test_fc_in
        assert cur_module.out_features == test_classes_dict[target_column]


def test_calculate_final_multi_output():
    test_classes_dict, test_fc_in = get_test_module_dict_data()

    output_layer_model_dict = models._get_module_dict_from_target_columns(
        num_classes=test_classes_dict, fc_in=test_fc_in
    )

    test_input = torch.zeros(128)

    test_multi_output = models._calculate_task_branch_outputs(
        input_=test_input, last_module=output_layer_model_dict
    )

    # since the input is zero, we only get the bias
    for target_column, tensor in test_multi_output.items():
        module_bias = output_layer_model_dict[target_column].bias
        assert (module_bias == tensor).all()


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"dilation_factor": 2, "block_number": 1, "width": 1000}, 2),
        ({"dilation_factor": 4, "block_number": 2, "width": 1000}, 16),
        ({"dilation_factor": 5, "block_number": 5, "width": 100}, 25),
    ],
)
def test_get_cur_dilation(test_input, expected):
    test_dilation = models._get_cur_dilation(**test_input)

    assert test_dilation == expected
