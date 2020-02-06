from copy import deepcopy

import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from human_origins_supervised import train
from human_origins_supervised.train_utils import metric_funcs


def test_calculate_batch_metrics():
    test_kwargs = get_calculate_batch_metrics_data_test_kwargs()
    test_batch_metrics = metric_funcs.calculate_batch_metrics(**test_kwargs)

    assert test_batch_metrics["Origin"]["v_Origin_mcc"] == 1.0
    assert test_batch_metrics["Origin"]["v_Origin_loss"] == 0.0

    assert test_batch_metrics["BMI"]["v_BMI_r2"] == 1.0
    assert test_batch_metrics["BMI"]["v_BMI_rmse"] == 0.0
    assert test_batch_metrics["BMI"]["v_BMI_pcc"] == 1.0
    assert test_batch_metrics["BMI"]["v_BMI_loss"] == 0.0

    assert test_batch_metrics["Height"]["v_Height_r2"] < 0
    assert test_batch_metrics["Height"]["v_Height_rmse"] > 0.0
    assert test_batch_metrics["Height"]["v_Height_pcc"] == -1.0
    assert test_batch_metrics["Height"]["v_Height_loss"] == 1.0


def get_calculate_batch_metrics_data_test_kwargs():
    target_columns = {"cat": ["Origin"], "con": ["BMI", "Height"]}

    standard_scaler_fit_arr = [[0.0], [1.0], [2.0]]
    target_transformers = {
        "Origin": LabelEncoder().fit([1, 2, 3]),
        "BMI": StandardScaler().fit(standard_scaler_fit_arr),
        "Height": StandardScaler().fit(standard_scaler_fit_arr),
    }

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

    batch_metrics_function_kwargs = {
        "target_columns": target_columns,
        "target_transformers": target_transformers,
        "losses": losses,
        "outputs": outputs,
        "labels": labels,
        "prefix": "v_",
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

    perfect_pred_loss = metric_funcs.calculate_losses(
        criterions=test_criterions, labels=test_labels, outputs=test_outputs
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

    bad_pred_loss = metric_funcs.calculate_losses(
        criterions=test_criterions, labels=test_labels, outputs=test_outputs
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

    test_criterions = train._get_criterions(test_target_columns_dict)
    test_labels = generate_base_dict(label_values)

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

    test_aggregated_losses = metric_funcs.aggregate_losses(losses_dict)
    assert test_aggregated_losses.item() == 2.0
