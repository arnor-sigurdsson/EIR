from argparse import Namespace
from unittest.mock import patch

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

from snp_pred import train
from snp_pred.models.models import CNNModel, MLPModel
from snp_pred.train_utils import optimizers


@patch("snp_pred.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_pass(patched_get_run_folder, tmp_path):

    # patch since we don't want to create run folders while testing
    patched_get_run_folder.return_value = tmp_path / "test_folder"
    train._prepare_run_folder("test_folder")

    assert (tmp_path / "test_folder").exists()


@patch("snp_pred.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_fail(patched_get_run_folder, tmp_path):

    patched_path = tmp_path / "test_folder"
    patched_get_run_folder.return_value = patched_path
    patched_path.mkdir()

    fake_file = patched_path / "train_average_history.log"
    fake_file.write_text("Disco Elysium")

    with pytest.raises(FileExistsError):
        train._prepare_run_folder("test_folder")


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
def test_get_train_sampler(args_config, create_test_data, create_test_datasets):
    cl_args = args_config
    train_dataset, *_ = create_test_datasets
    cl_args.weighted_sampling_columns = ["Origin"]

    test_sampler = train.get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    cl_args.weighted_sampling_columns = None
    test_sampler = train.get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
def test_get_dataloaders(create_test_cl_args, create_test_data, create_test_datasets):
    cl_args = create_test_cl_args
    cl_args.weighted_sampling_columns = ["Origin"]

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = train.get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_columns, train_dataset=train_dataset
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
    patch_target = "snp_pred.train.torch.cuda.device_count"
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
    adamw_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, cl_args=args_config
    )
    assert isinstance(adamw_optimizer, AdamW)

    args_config.optimizer = "sgdm"
    sgdm_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, cl_args=args_config
    )
    assert isinstance(sgdm_optimizer, SGD)
    assert sgdm_optimizer.param_groups[0]["momentum"] == 0.9


def test_get_model(args_config):

    # TODO: Refactor checking of fc_3 into separate test.

    args_config.model_type = "cnn"
    num_classes_dict = {"Origin": 10, "Height": 1}
    cnn_model = train.get_model(args_config, num_classes_dict, None)

    assert isinstance(cnn_model, CNNModel)
    assert cnn_model.multi_task_branches["Origin"].fc_3_final.out_features == 10
    assert cnn_model.multi_task_branches["Height"].fc_3_final.out_features == 1

    args_config.model_type = "mlp"
    mlp_model = train.get_model(args_config, num_classes_dict, None)
    assert isinstance(mlp_model, MLPModel)
    # assert mlp_model.multi_task_branches["Origin"].fc_3_final.out_features == 10
    # assert mlp_model.multi_task_branches["Height"].fc_3_final.out_features == 1


def test_get_criterions_nonlinear():

    test_target_columns_dict = {
        "con": ["Height", "BMI"],
        "cat": ["Origin", "HairColor"],
    }

    test_criterions = train._get_criterions(test_target_columns_dict, model_type="cnn")
    for column_name in test_target_columns_dict["con"]:
        assert isinstance(test_criterions[column_name], nn.MSELoss)

    for column_name in test_target_columns_dict["cat"]:
        assert isinstance(test_criterions[column_name], nn.CrossEntropyLoss)


def test_get_criterions_linear_pass():

    test_target_columns_dict_con = {"con": ["Height"], "cat": []}

    test_criterions_con = train._get_criterions(
        test_target_columns_dict_con, model_type="linear"
    )

    assert len(test_criterions_con) == 1
    for column_name in test_target_columns_dict_con["con"]:
        assert isinstance(test_criterions_con[column_name], nn.MSELoss)

    test_target_columns_dict_cat = {"con": [], "cat": ["Origin"]}

    test_criterions_cat = train._get_criterions(
        test_target_columns_dict_cat, model_type="linear"
    )
    assert len(test_criterions_cat) == 1
    for column_name in test_target_columns_dict_cat["cat"]:
        # TODO: Do this better, a bit hacky currently as calc_bce is private
        assert test_criterions_cat[column_name].__name__ == "calc_bce"


def test_check_linear_model_columns_pass():
    extra = {"extra_cat_columns": [], "extra_con_columns": []}
    test_input_cat = Namespace(
        target_cat_columns=["Origin"], target_con_columns=[], **extra
    )
    train._check_linear_model_columns(cl_args=test_input_cat)

    test_input_con = Namespace(
        target_con_columns=["Height"], target_cat_columns=[], **extra
    )
    train._check_linear_model_columns(cl_args=test_input_con)


def test_check_linear_model_columns_fail():
    extra = {"extra_cat_columns": [], "extra_con_columns": []}
    test_input_cat = Namespace(
        target_cat_columns=["Origin", "Height"], target_con_columns=[], **extra
    )
    with pytest.raises(NotImplementedError):
        train._check_linear_model_columns(cl_args=test_input_cat)

    test_input_con = Namespace(
        target_con_columns=["Height", "BMI"], target_cat_columns=[], **extra
    )
    with pytest.raises(NotImplementedError):
        train._check_linear_model_columns(cl_args=test_input_con)

    test_input_mixed = Namespace(
        target_con_columns=["Height", "BMI"], target_cat_columns=["Height"], **extra
    )
    with pytest.raises(NotImplementedError):
        train._check_linear_model_columns(cl_args=test_input_mixed)
