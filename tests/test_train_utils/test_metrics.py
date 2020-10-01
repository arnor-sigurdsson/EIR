from copy import deepcopy
from math import isclose

import pytest
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from snp_pred import train
from snp_pred.train_utils import metrics


def test_calculate_batch_metrics():
    test_kwargs = get_calculate_batch_metrics_data_test_kwargs()
    test_batch_metrics = metrics.calculate_batch_metrics(**test_kwargs)

    assert test_batch_metrics["Origin"]["Origin_mcc"] == 1.0
    assert test_batch_metrics["Origin"]["Origin_loss"] == 0.0

    assert test_batch_metrics["BMI"]["BMI_r2"] == 1.0
    assert test_batch_metrics["BMI"]["BMI_rmse"] == 0.0
    # sometimes slight numerical instability with scipy pearsonr
    assert isclose(test_batch_metrics["BMI"]["BMI_pcc"], 1.0)
    assert test_batch_metrics["BMI"]["BMI_loss"] == 0.0

    assert test_batch_metrics["Height"]["Height_r2"] < 0
    assert test_batch_metrics["Height"]["Height_rmse"] > 0.0
    assert isclose(test_batch_metrics["Height"]["Height_pcc"], -1.0)
    assert test_batch_metrics["Height"]["Height_loss"] == 1.0


def get_calculate_batch_metrics_data_test_kwargs():
    target_columns = {"cat": ["Origin"], "con": ["BMI", "Height"]}

    standard_scaler_fit_arr = [[0.0], [1.0], [2.0]]

    losses = {
        "Origin": torch.tensor(0.0),
        "BMI": torch.tensor(0.0),
        "Height": torch.tensor(1.0),
    }

    outputs = {
        "Origin": torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        "BMI": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
        "Height": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
    }

    labels = {
        "Origin": torch.tensor([0, 1, 2]).unsqueeze(1),
        "BMI": torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1),
        "Height": torch.tensor([-1.0, -2.0, -3.0]).unsqueeze(1),
    }

    target_transformers = {
        "Origin": LabelEncoder().fit([1, 2, 3]),
        "BMI": StandardScaler().fit(standard_scaler_fit_arr),
        "Height": StandardScaler().fit(standard_scaler_fit_arr),
    }
    metrics_ = metrics.get_default_metrics(target_transformers=target_transformers)

    batch_metrics_function_kwargs = {
        "target_columns": target_columns,
        "losses": losses,
        "outputs": outputs,
        "labels": labels,
        "mode": "val",
        "metric_record_dict": metrics_,
    }

    return batch_metrics_function_kwargs


def test_calculate_losses_good():
    """
    Note that CrossEntropy applies LogSoftmax() before calculating the NLLLoss().

    We expect the the CrossEntropyLosses to be around 0.9048

        >>> loss = torch.nn.CrossEntropyLoss()
        >>> input_ = torch.zeros(1, 5)
        >>> input_[0, 0] = 1
        >>> target = torch.zeros(1, dtype=torch.long)
        >>> loss(input_, target)
        tensor(0.9048)
    """

    common_values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    test_criterions, test_labels, test_outputs = set_up_calculate_losses_data(
        label_values=common_values, output_values=common_values
    )

    perfect_pred_loss = metrics.calculate_prediction_losses(
        criterions=test_criterions, targets=test_labels, inputs=test_outputs
    )

    assert perfect_pred_loss["Height"].item() == 0.0
    assert perfect_pred_loss["BMI"].item() == 0.0

    assert 0.904 < perfect_pred_loss["Origin"].item() < 0.905
    assert 0.904 < perfect_pred_loss["HairColor"].item() < 0.905


def test_calculate_losses_bad():

    # diff of 2 between each pair, RMSE expected to be 4.0
    label_values = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    output_values = torch.tensor([2, 3, 4, 5, 6], dtype=torch.int64)
    test_criterions, test_labels, test_outputs = set_up_calculate_losses_data(
        label_values=label_values, output_values=output_values
    )

    bad_pred_loss = metrics.calculate_prediction_losses(
        criterions=test_criterions, targets=test_labels, inputs=test_outputs
    )

    expected_rmse = 4.0
    assert bad_pred_loss["Height"].item() == expected_rmse
    assert bad_pred_loss["BMI"].item() == expected_rmse

    # check that the loss is more than upper bound (0.905) in perfect case
    perfect_upper_bound = 0.905
    assert bad_pred_loss["Origin"].item() > perfect_upper_bound
    assert bad_pred_loss["HairColor"].item() > perfect_upper_bound


def set_up_calculate_losses_data(
    label_values: torch.Tensor, output_values: torch.Tensor
):
    test_target_columns_dict = {
        "con": ["Height", "BMI"],
        "cat": ["Origin", "HairColor"],
    }

    def generate_base_dict(values: torch.Tensor):

        base_dict = {
            "Height": deepcopy(values).to(dtype=torch.float32),
            "BMI": deepcopy(values).to(dtype=torch.float32),
            "Origin": deepcopy(values),
            "HairColor": deepcopy(values),
        }

        return base_dict

    test_criterions = train._get_criterions(
        target_columns=test_target_columns_dict, model_type="cnn"
    )
    test_labels = generate_base_dict(values=label_values)

    test_outputs = generate_base_dict(output_values)

    one_hot = torch.nn.functional.one_hot
    test_outputs["Origin"] = one_hot(test_outputs["Origin"])
    test_outputs["Origin"] = test_outputs["Origin"].to(dtype=torch.float32)

    test_outputs["HairColor"] = one_hot(test_outputs["HairColor"])
    test_outputs["HairColor"] = test_outputs["HairColor"].to(dtype=torch.float32)

    return test_criterions, test_labels, test_outputs


def test_aggregate_losses():
    # expected average of [0,1,2,3,4] = 2.0
    losses_dict = {str(i): torch.tensor(i, dtype=torch.float32) for i in range(5)}

    test_aggregated_losses = metrics.aggregate_losses(losses_dict)
    assert test_aggregated_losses.item() == 2.0


@pytest.fixture()
def get_l1_test_model():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.fc_1 = torch.nn.Linear(10, 1)
            self.l1_penalized_weights = self.fc_1.weight

        def forward(self, x):
            return x

    return TestModel()


def test_get_extra_loss_term_functions_pass(get_l1_test_model):

    test_model = get_l1_test_model

    extra_loss_functions_with_l1 = metrics.get_extra_loss_term_functions(
        model=test_model, l1_weight=1.0
    )
    assert len(extra_loss_functions_with_l1) == 1


def test_get_extra_loss_term_functions_fail(get_l1_test_model):

    test_model = get_l1_test_model
    delattr(test_model, "l1_penalized_weights")

    with pytest.raises(AttributeError):
        metrics.get_extra_loss_term_functions(model=test_model, l1_weight=1.0)


def test_l1_extra_loss(get_l1_test_model):
    test_model = get_l1_test_model

    torch.nn.init.ones_(test_model.fc_1.weight)
    extra_loss_functions_with_l1 = metrics.get_extra_loss_term_functions(
        model=test_model, l1_weight=1.0
    )

    l1_loss_func = extra_loss_functions_with_l1[0]
    l1_loss = l1_loss_func()
    assert l1_loss == 10.0

    torch.nn.init.zeros_(test_model.fc_1.weight)
    extra_loss_functions_with_l1 = metrics.get_extra_loss_term_functions(
        model=test_model, l1_weight=1.0
    )

    l1_loss_func = extra_loss_functions_with_l1[0]
    l1_loss = l1_loss_func()
    assert l1_loss == 0.0


def test_add_extra_losses(get_l1_test_model):

    test_model = get_l1_test_model

    torch.nn.init.ones_(test_model.fc_1.weight)
    extra_loss_functions_with_l1 = metrics.get_extra_loss_term_functions(
        model=test_model, l1_weight=1.0
    )
    total_loss = metrics.add_extra_losses(
        total_loss=torch.tensor(0.0), extra_loss_functions=extra_loss_functions_with_l1
    )
    assert total_loss == 10.0

    # test that multiple losses are aggregated correctly
    extra_loss_functions_with_l1_multiple = extra_loss_functions_with_l1 * 3
    total_loss = metrics.add_extra_losses(
        total_loss=torch.tensor(0.0),
        extra_loss_functions=extra_loss_functions_with_l1_multiple,
    )
    assert total_loss == 30.0
