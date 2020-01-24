from unittest.mock import patch

import pytest
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

from human_origins_supervised import train
from human_origins_supervised.models.models import CNNModel, MLPModel


def test_prepare_run_folder_pass(tmp_path):
    # patch since we don't want to create run folders while testing
    with patch("human_origins_supervised.train.get_run_folder") as m:
        m.return_value = tmp_path / "test_folder"
        train._prepare_run_folder("test_folder")

        assert (tmp_path / "test_folder").exists()


def test_prepare_run_folder_fail(tmp_path):
    # patch since we don't want to create run folders while testing
    with patch("human_origins_supervised.train.get_run_folder") as m:
        patched_path = tmp_path / "test_folder"
        m.return_value = patched_path
        patched_path.mkdir()

        fake_file = patched_path / "training_history.log"
        fake_file.write_text("Disco Elysium")

        with pytest.raises(FileExistsError):
            train._prepare_run_folder("test_folder")


@pytest.mark.parametrize("create_test_data", [{"class_type": "multi"}], indirect=True)
def test_get_train_sampler(args_config, create_test_data, create_test_datasets):
    cl_args = args_config
    train_dataset, *_ = create_test_datasets
    cl_args.weighted_sampling = True

    test_sampler = train.get_train_sampler(cl_args, train_dataset)
    assert isinstance(test_sampler, WeightedRandomSampler)

    cl_args.weighted_sampling = False
    test_sampler = train.get_train_sampler(cl_args, train_dataset)
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"class_type": "multi"}], indirect=True)
def test_get_dataloaders(create_test_cl_args, create_test_data, create_test_datasets):
    cl_args = create_test_cl_args
    cl_args.weighted_sampling = True

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = train.get_train_sampler(cl_args, train_dataset)

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

    args_config.model_type = "cnn"
    cnn_model = train.get_model(args_config, 10, None)
    assert isinstance(cnn_model, CNNModel)
    assert cnn_model.fc_3[-1].out_features == 10

    args_config.model_type = "mlp"
    mlp_model = train.get_model(args_config, 10, None)
    assert isinstance(mlp_model, MLPModel)
    assert mlp_model.fc_3[-1].out_features == 10


def test_get_criterion(args_config):

    args_config.model_task = "cls"
    cls_criterion = train.get_criterion(args_config)
    assert isinstance(cls_criterion, nn.CrossEntropyLoss)

    args_config.model_task = "mse"
    mse_criterion = train.get_criterion(args_config)
    assert isinstance(mse_criterion, nn.MSELoss)
