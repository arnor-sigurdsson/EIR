from copy import deepcopy

import pytest
import torch
from torch import nn

from eir.models import model_training_utils
from eir.models.input.array import models_cnn
from eir.models.input.array.models_cnn import ConvParamSuggestion


@pytest.fixture
def create_test_util_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc_1 = nn.Linear(10, 10, bias=True)
            self.act_1 = nn.PReLU()
            self.bn_1 = nn.BatchNorm1d(10)

            self.fc_2 = nn.Linear(10, 10, bias=True)
            self.act_2 = nn.PReLU()
            self.bn_2 = nn.BatchNorm1d(10)

        def forward(self, x):
            return x

    model = TestModel()

    return model


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {
                "input_size_w": 1000,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 4,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 1000,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 4,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [2],
        ),
        (
            {
                "input_size_w": 10000,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 4,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 10000,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 4,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [2, 1],
        ),
        (
            {
                "input_size_w": 1e6,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 3,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 1e6,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 3,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [2, 2, 3, 2],
        ),
        (
            {
                "input_size_w": 64,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 64,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [1],
        ),
        (
            {
                "input_size_w": 128,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 128,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [2],
        ),
        (
            {
                "input_size_w": 32,
                "kernel_w": 3,
                "first_kernel_expansion_w": 1,
                "stride_w": 2,
                "first_stride_expansion_w": 1,
                "dilation_w": 1,
                "down_sample_every_n_blocks": None,
                "input_size_h": 32,
                "kernel_h": 3,
                "first_kernel_expansion_h": 1,
                "stride_h": 2,
                "first_stride_expansion_h": 1,
                "dilation_h": 1,
                "cutoff": 128,
            },
            [],
        ),
    ],
)
def test_find_no_residual_blocks_needed(test_input, expected):
    assert models_cnn.auto_find_no_cnn_residual_blocks_needed(**test_input) == expected


def test_get_model_params(create_test_util_model):
    test_model = create_test_util_model

    weight_decay = 0.05
    model_params = model_training_utils.add_wd_to_model_params(
        model=test_model,
        wd=weight_decay,
    )

    model_params_with_decay = model_params[0]
    model_params_no_decay = model_params[1]

    # the 2 linear layers
    assert len(model_params_with_decay["params"]) == 2
    assert model_params_with_decay["weight_decay"] == weight_decay

    # 2 PReLU + 2 BN Gamma + 2 BN Bias + 2 Linear Bias
    assert len(model_params_no_decay["params"]) == 8
    assert model_params_no_decay["weight_decay"] == 0.0


def set_up_stack_list_of_tensors_dicts_data():
    test_batch_base = {
        "Target_Column_1": torch.ones((5, 5)),
        "Target_Column_2": torch.ones((5, 5)) * 2,
    }

    test_list_of_batches = [deepcopy(test_batch_base) for _ in range(3)]

    for i in range(3):
        test_list_of_batches[i]["Target_Column_1"] *= i
        test_list_of_batches[i]["Target_Column_2"] *= i

    return test_list_of_batches


def test_stack_list_of_tensor_dicts():
    test_input = set_up_stack_list_of_tensors_dicts_data()

    test_output = model_training_utils._stack_list_of_batch_dicts(test_input)

    assert (test_output["Target_Column_1"][0] == 0.0).all()
    assert (test_output["Target_Column_1"][5] == 1.0).all()
    assert (test_output["Target_Column_1"][10] == 2.0).all()

    assert (test_output["Target_Column_2"][0] == 0.0).all()
    assert (test_output["Target_Column_2"][5] == 2.0).all()
    assert (test_output["Target_Column_2"][10] == 4.0).all()


def set_up_stack_list_of_output_tensor_dicts_data():
    test_batch_base = {
        "test_output_tabular": {
            "Target_Column_1": torch.ones((5, 5)),
            "Target_Column_2": torch.ones((5, 5)) * 2,
        }
    }

    test_list_of_batches = [deepcopy(test_batch_base) for _ in range(3)]

    for i in range(3):
        test_list_of_batches[i]["test_output_tabular"]["Target_Column_1"] *= i
        test_list_of_batches[i]["test_output_tabular"]["Target_Column_2"] *= i

    return test_list_of_batches


def test_stack_list_of_output_target_dicts():
    test_input = set_up_stack_list_of_output_tensor_dicts_data()

    test_output = model_training_utils.stack_list_of_output_target_dicts(test_input)

    assert (test_output["test_output_tabular"]["Target_Column_1"][0] == 0.0).all()
    assert (test_output["test_output_tabular"]["Target_Column_1"][5] == 1.0).all()
    assert (test_output["test_output_tabular"]["Target_Column_1"][10] == 2.0).all()

    assert (test_output["test_output_tabular"]["Target_Column_2"][0] == 0.0).all()
    assert (test_output["test_output_tabular"]["Target_Column_2"][5] == 2.0).all()
    assert (test_output["test_output_tabular"]["Target_Column_2"][10] == 4.0).all()


def test_calc_size_after_conv_sequence():
    class SimpleBlock(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.bn = nn.BatchNorm2d(16)
            self.act = nn.ReLU(True)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            return out

    conv_seq = nn.Sequential(*[SimpleBlock()] * 3)
    width, height = models_cnn.calc_size_after_conv_sequence(
        input_width=224, input_height=8, conv_sequence=conv_seq
    )

    assert width == 28
    assert height == 1

    input_tensor = torch.rand(1, 16, 224, 8)
    output_tensor = conv_seq(input_tensor)
    assert output_tensor.shape == (1, 16, 28, 1)

    conv_seq_bad = nn.Sequential(*[SimpleBlock()] * 10)
    with pytest.raises(ValueError):
        models_cnn.calc_size_after_conv_sequence(
            input_width=224, input_height=8, conv_sequence=conv_seq_bad
        )


@pytest.mark.parametrize(
    "test_input,expected",
    [  # Even input and kernel
        ((1000, 10, 4, 1), ConvParamSuggestion(10, 0, 4, 1, 1)),
        ((1000, 10, 4, 3), ConvParamSuggestion(10, 0, 4, 3, 0)),
        ((250, 4, 4, 1), ConvParamSuggestion(4, 0, 4, 1, 1)),
        # # Odd input, odd kernel
        ((1001, 11, 2, 1), ConvParamSuggestion(11, 0, 2, 1, 0)),
        ((1001, 11, 1, 1), ConvParamSuggestion(11, 0, 1, 1, 5)),
        ((1001, 11, 4, 2), ConvParamSuggestion(11, 0, 4, 2, 0)),
        # Odd input, mixed kernels
        ((1001, 11, 11, 1), ConvParamSuggestion(11, 0, 11, 1, 0)),
        ((1001, 10, 10, 1), ConvParamSuggestion(11, 0, 10, 1, 0)),
        ((1001, 11, 3, 2), ConvParamSuggestion(12, 0, 3, 2, 0)),
    ],
)
def test_calc_conv_padding_needed_pass(test_input, expected):
    """
    input_width, kernel_size, stride, dilation
    """
    result = models_cnn.calc_conv_params_needed(*test_input)

    assert result.kernel_size == expected.kernel_size
    assert result.padding == expected.padding
    assert result.stride == expected.stride
    assert result.dilation == expected.dilation


def test_calc_padding_needed_fail():
    with pytest.raises(ValueError):
        models_cnn.calc_conv_params_needed(-1000, 10, 4, 1)
