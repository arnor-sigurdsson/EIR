from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

import train_utils.metric_funcs
import train_utils.utils
from human_origins_supervised import train
from human_origins_supervised.models.models import CNNModel, MLPModel


@patch("human_origins_supervised.train.get_run_folder", autospec=True)
def test_prepare_run_folder_pass(patched_get_run_folder, tmp_path):

    # patch since we don't want to create run folders while testing
    patched_get_run_folder.return_value = tmp_path / "test_folder"
    train._prepare_run_folder("test_folder")

    assert (tmp_path / "test_folder").exists()


@patch("human_origins_supervised.train.get_run_folder", autospec=True)
def test_prepare_run_folder_fail(patched_get_run_folder, tmp_path):

    patched_path = tmp_path / "test_folder"
    patched_get_run_folder.return_value = patched_path
    patched_path.mkdir()

    fake_file = patched_path / "training_history.log"
    fake_file.write_text("Disco Elysium")

    with pytest.raises(FileExistsError):
        train._prepare_run_folder("test_folder")


@pytest.mark.parametrize("create_test_data", [{"class_type": "multi"}], indirect=True)
def test_get_train_sampler(args_config, create_test_data, create_test_datasets):
    cl_args = args_config
    train_dataset, *_ = create_test_datasets
    cl_args.weighted_sampling_column = "Origin"

    test_sampler = train.get_train_sampler(
        column_to_sample=cl_args.weighted_sampling_column, train_dataset=train_dataset
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    cl_args.weighted_sampling_column = None
    test_sampler = train.get_train_sampler(
        column_to_sample=cl_args.weighted_sampling_column, train_dataset=train_dataset
    )
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"class_type": "multi"}], indirect=True)
def test_get_dataloaders(create_test_cl_args, create_test_data, create_test_datasets):
    cl_args = create_test_cl_args
    cl_args.weighted_sampling_column = "Origin"

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = train.get_train_sampler(
        cl_args.weighted_sampling_column, train_dataset
    )

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, train_sampler, valid_dataset, cl_args.batch_size
    )

    assert train_dataloader.batch_size == cl_args.batch_size
    assert valid_dataloader.batch_size == cl_args.batch_size
    assert isinstance(train_dataloader.sampler, WeightedRandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, None, valid_dataset, cl_args.batch_size
    )

    assert isinstance(train_dataloader.sampler, RandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)


def _modify_bs_for_multi_gpu():
    patch_target = "human_origins_supervised.train.torch.cuda.device_count"
    with patch(patch_target, autospec=True) as m:
        m.return_value = 2

        assert train._modify_bs_for_multi_gpu(True, 32) == 64
        assert train._modify_bs_for_multi_gpu(False, 32) == 64


def test_get_optimizer(args_config):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(10, 10)

        def forward(self, x):
            pass

    model = FakeModel()

    args_config.optimizer = "adamw"
    adamw_optimizer = train.get_optimizer(model, args_config)
    assert isinstance(adamw_optimizer, AdamW)

    args_config.optimizer = "sgdm"
    sgdm_optimizer = train.get_optimizer(model, args_config)
    assert isinstance(sgdm_optimizer, SGD)
    assert sgdm_optimizer.param_groups[0]["momentum"] == 0.9


def test_get_model(args_config):

    # TODO: Refactor checking of fc_3 into separate test.

    args_config.model_type = "cnn"
    num_classes_dict = {"Origin": 10, "Height": 1}
    cnn_model = train.get_model(args_config, num_classes_dict, None)

    assert isinstance(cnn_model, CNNModel)
    assert cnn_model.fc_3_last_module["Origin"].out_features == 10
    assert cnn_model.fc_3_last_module["Height"].out_features == 1

    args_config.model_type = "mlp"
    mlp_model = train.get_model(args_config, num_classes_dict, None)
    assert isinstance(mlp_model, MLPModel)
    assert mlp_model.fc_3_last_module["Origin"].out_features == 10
    assert mlp_model.fc_3_last_module["Height"].out_features == 1


def test_get_criterions():

    test_target_columns_dict = {
        "con": ["Height", "BMI"],
        "cat": ["Origin", "HairColor"],
    }

    test_criterions = train._get_criterions(test_target_columns_dict)
    for column_name in test_target_columns_dict["con"]:
        assert isinstance(test_criterions[column_name], nn.MSELoss)

    for column_name in test_target_columns_dict["cat"]:
        assert isinstance(test_criterions[column_name], nn.CrossEntropyLoss)


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

    perfect_pred_loss = train_utils.metric_funcs.calculate_losses(
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

    bad_pred_loss = train_utils.metric_funcs.calculate_losses(
        criterions=test_criterions, labels=test_labels, outputs=test_outputs
    )

    expected_rmse = 4.0
    assert bad_pred_loss["Height"].item() == expected_rmse
    assert bad_pred_loss["BMI"].item() == expected_rmse

    # check that the loss is more than upper bound (0.905) in perfect case
    perfect_upper_bound = 0.905
    assert bad_pred_loss["Origin"].item() > perfect_upper_bound
    assert bad_pred_loss["HairColor"].item() > perfect_upper_bound


def test_aggregate_losses():
    # expected average of [0,1,2,3,4] = 2.0
    losses_dict = {str(i): torch.tensor(i, dtype=torch.float32) for i in range(5)}

    test_aggregated_losses = train_utils.metric_funcs.aggregate_losses(losses_dict)
    assert test_aggregated_losses.item() == 2.0
