from copy import deepcopy

import pytest
import torch
from torch import nn

from snp_pred.models import model_training_utils
from snp_pred.models.omics import models_cnn


@pytest.fixture
def create_test_util_model():
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc_1 = nn.Linear(10, 10, bias=False)
            self.act_1 = nn.PReLU()
            self.bn_1 = nn.BatchNorm1d(10)

            self.fc_2 = nn.Linear(10, 10, bias=False)
            self.act_2 = nn.PReLU()
            self.bn_2 = nn.BatchNorm1d(10)

        def forward(self, x):
            return x

    model = TestModel()

    return model


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"width": 1000, "stride": 4, "first_stride_expansion": 1}, [2]),
        ({"width": 10000, "stride": 4, "first_stride_expansion": 1}, [2, 2]),
        ({"width": 1e6, "stride": 4, "first_stride_expansion": 1}, [2, 2, 2, 1]),
    ],
)
def test_find_no_resblocks_needed(test_input, expected):
    assert models_cnn.find_no_cnn_resblocks_needed(**test_input) == expected


def test_get_model_params(create_test_util_model):
    test_model = create_test_util_model

    weight_decay = 0.05
    model_params = model_training_utils.add_wd_to_model_params(
        model=test_model, wd=weight_decay
    )

    # BN has weight and bias, hence 6 + 2 = 8 parameter groups
    assert len(model_params) == 8

    for param_group in model_params:
        if param_group["params"].shape[0] == 1:
            assert param_group["weight_decay"] == 0.00
        else:
            assert param_group["weight_decay"] == 0.05


def set_up_stack_list_of_tensors_dicts_data():
    test_batch_base = {
        "Target_Column_1": torch.ones((5, 5)),
        "Target_Column_2": torch.ones((5, 5)) * 2,
    }

    test_list_of_batches = [deepcopy(test_batch_base) for i in range(3)]

    for i in range(3):
        test_list_of_batches[i]["Target_Column_1"] *= i
        test_list_of_batches[i]["Target_Column_2"] *= i

    return test_list_of_batches


def test_stack_list_of_tensor_dicts():
    test_input = set_up_stack_list_of_tensors_dicts_data()

    test_output = model_training_utils._stack_list_of_tensor_dicts(test_input)

    assert (test_output["Target_Column_1"][0] == 0.0).all()
    assert (test_output["Target_Column_1"][5] == 1.0).all()
    assert (test_output["Target_Column_1"][10] == 2.0).all()

    assert (test_output["Target_Column_2"][0] == 0.0).all()
    assert (test_output["Target_Column_2"][5] == 2.0).all()
    assert (test_output["Target_Column_2"][10] == 4.0).all()
