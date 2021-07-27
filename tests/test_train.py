from argparse import Namespace
from unittest.mock import patch

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

import eir.data_load.data_utils
import eir.setup.config
import eir.models.omics.omics_models
import eir.setup.input_setup
import eir.train
from eir import train
from eir.data_load import label_setup
from eir.models.fusion import FusionModel
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.models_mlp import MLPModel
from eir.train_utils import optimizers


@patch("eir.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_pass(patched_get_run_folder, tmp_path):

    # patch since we don't want to create run folders while testing
    patched_get_run_folder.return_value = tmp_path / "test_folder"
    train._prepare_run_folder("test_folder")

    assert (tmp_path / "test_folder").exists()


@patch("eir.train.utils.get_run_folder", autospec=True)
def test_prepare_run_folder_fail(patched_get_run_folder, tmp_path):

    patched_path = tmp_path / "test_folder"
    patched_get_run_folder.return_value = patched_path
    patched_path.mkdir()

    fake_file = patched_path / "train_average_history.log"
    fake_file.write_text("Disco Elysium")

    with pytest.raises(FileExistsError):
        train._prepare_run_folder("test_folder")


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
def test_get_train_sampler(test_config_base, create_test_data, create_test_datasets):
    cl_args = test_config_base
    train_dataset, *_ = create_test_datasets
    cl_args.weighted_sampling_columns = ["Origin"]

    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    cl_args.weighted_sampling_columns = None
    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
def test_get_dataloaders(create_test_config, create_test_data, create_test_datasets):
    cl_args = create_test_config
    cl_args.weighted_sampling_columns = ["Origin"]

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = eir.data_load.data_utils.get_train_sampler(
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
    patch_target = "eir.train.torch.cuda.device_count"
    with patch(patch_target, autospec=True) as m:
        m.return_value = 2

        assert train._modify_bs_for_multi_gpu(True, 32) == 64
        assert train._modify_bs_for_multi_gpu(False, 32) == 64


def test_get_optimizer(test_config_base):
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(10, 10)

        def forward(self, x):
            pass

    model = FakeModel()

    test_config_base.optimizer = "adamw"
    adamw_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=test_config_base
    )
    assert isinstance(adamw_optimizer, AdamW)

    test_config_base.optimizer = "sgdm"
    sgdm_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=test_config_base
    )
    assert isinstance(sgdm_optimizer, SGD)
    assert sgdm_optimizer.param_groups[0]["momentum"] == 0.9


def test_get_model(test_config_base):

    # TODO: Refactor checking of fc_3 into separate test.

    test_config_base.model_type = "cnn"
    num_outputs_per_target_dict = {"Origin": 10, "Height": 1}

    data_dimensions = {
        "omics_test": eir.setup.input_setup.DataDimensions(
            channels=1, height=4, width=1000
        )
    }

    cnn_fusion_model = train.get_model(
        global_config=test_config_base,
        omics_data_dimensions=data_dimensions,
        num_outputs_per_target=num_outputs_per_target_dict,
        tabular_label_transformers=None,
    )

    assert isinstance(cnn_fusion_model, FusionModel)
    assert cnn_fusion_model.multi_task_branches["Origin"][-1][-1].out_features == 10
    assert cnn_fusion_model.multi_task_branches["Height"][-1][-1].out_features == 1
    assert isinstance(cnn_fusion_model.modules_to_fuse["omics_test"], CNNModel)

    test_config_base.model_type = "mlp"
    mlp_fusion_model = train.get_model(
        global_config=test_config_base,
        omics_data_dimensions=data_dimensions,
        num_outputs_per_target=num_outputs_per_target_dict,
        tabular_label_transformers=None,
    )
    assert isinstance(mlp_fusion_model.modules_to_fuse["omics_test"], MLPModel)


def test_get_criterions_nonlinear():

    test_target_columns_dict = {
        "con": ["Height", "BMI"],
        "cat": ["Origin", "HairColor"],
    }

    test_criterions = train._get_criterions(test_target_columns_dict, model_type="cnn")
    for column_name in test_target_columns_dict["con"]:
        assert test_criterions[column_name].func is train._calc_mse

    for column_name in test_target_columns_dict["cat"]:
        assert isinstance(test_criterions[column_name], nn.CrossEntropyLoss)


def test_get_criterions_linear_pass():

    test_target_columns_dict_con = {"con": ["Height"], "cat": []}

    test_criterions_con = train._get_criterions(
        test_target_columns_dict_con, model_type="linear"
    )

    assert len(test_criterions_con) == 1
    for column_name in test_target_columns_dict_con["con"]:
        assert test_criterions_con[column_name].func is train._calc_mse

    test_target_columns_dict_cat = {"con": [], "cat": ["Origin"]}

    test_criterions_cat = train._get_criterions(
        test_target_columns_dict_cat, model_type="linear"
    )
    assert len(test_criterions_cat) == 1
    for column_name in test_target_columns_dict_cat["cat"]:
        assert test_criterions_cat[column_name] is train._calc_bce


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


def test_set_up_num_classes(get_transformer_test_data):
    df_test, test_target_columns_dict = get_transformer_test_data

    test_transformers = label_setup._get_fit_label_transformers(
        df_labels=df_test, label_columns=test_target_columns_dict
    )

    num_classes = train.set_up_num_outputs_per_target(
        target_transformers=test_transformers
    )

    assert num_classes["Height"] == 1
    assert num_classes["Origin"] == 3
