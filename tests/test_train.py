from unittest.mock import patch

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import WeightedRandomSampler, SequentialSampler, RandomSampler

import eir.data_load.data_utils
import eir.models.omics.omics_models
import eir.setup.config
import eir.setup.input_setup
import eir.train
from eir import train
from eir.data_load import label_setup
from eir.models.fusion import FusionModel
from eir.models.omics.models_cnn import CNNModel
from eir.models.omics.models_linear import LinearModel
from eir.setup.config import Configs
from eir.setup.schemas import GlobalConfig
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
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_default_experiment(create_test_config):
    test_configs = create_test_config

    default_experiment = train.get_default_experiment(configs=test_configs, hooks=None)
    assert default_experiment.configs == test_configs

    assert default_experiment.hooks is None

    assert len(default_experiment.criterions) == 1
    assert isinstance(default_experiment.criterions["Origin"], nn.CrossEntropyLoss)

    assert len(default_experiment.inputs) == 1
    assert set(default_experiment.inputs.keys()) == {"omics_test_genotype"}

    assert default_experiment.target_columns == {"cat": ["Origin"], "con": []}


def test_modify_bs_for_multi_gpu():
    no_gpu = train._modify_bs_for_multi_gpu(multi_gpu=False, batch_size=64)
    assert no_gpu == 64

    with patch("eir.train.torch.cuda.device_count", autospec=True) as mocked:
        mocked.return_value = 4
        gpu = train._modify_bs_for_multi_gpu(multi_gpu=True, batch_size=64)
        assert gpu == 64 * 4


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_train_sampler(create_test_data, create_test_datasets, create_test_config):

    test_config = create_test_config
    gc = test_config.global_config

    train_dataset, *_ = create_test_datasets
    gc.weighted_sampling_columns = ["Origin"]

    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert isinstance(test_sampler, WeightedRandomSampler)

    gc.weighted_sampling_columns = None
    test_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )
    assert test_sampler is None


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {"lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_dataloaders(create_test_config, create_test_data, create_test_datasets):
    test_config = create_test_config
    gc = test_config.global_config
    gc.weighted_sampling_columns = ["Origin"]

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = eir.data_load.data_utils.get_train_sampler(
        columns_to_sample=gc.weighted_sampling_columns, train_dataset=train_dataset
    )

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, train_sampler, valid_dataset, gc.batch_size
    )

    assert train_dataloader.batch_size == gc.batch_size
    assert valid_dataloader.batch_size == gc.batch_size
    assert isinstance(train_dataloader.sampler, WeightedRandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset, None, valid_dataset, gc.batch_size
    )

    assert isinstance(train_dataloader.sampler, RandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)


def _modify_bs_for_multi_gpu():
    patch_target = "eir.train.torch.cuda.device_count"
    with patch(patch_target, autospec=True) as m:
        m.return_value = 2

        assert train._modify_bs_for_multi_gpu(True, 32) == 64
        assert train._modify_bs_for_multi_gpu(False, 32) == 64


def test_get_optimizer():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(10, 10)

        def forward(self, x):
            pass

    model = FakeModel()

    gc_adamw = GlobalConfig(run_name="test", optimizer="adamw")

    adamw_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=gc_adamw
    )
    assert isinstance(adamw_optimizer, AdamW)

    gc_sgdm = GlobalConfig(run_name="test", optimizer="sgdm")
    sgdm_optimizer = optimizers.get_optimizer(
        model=model, loss_callable=lambda x: x, global_config=gc_sgdm
    )
    assert isinstance(sgdm_optimizer, SGD)
    assert sgdm_optimizer.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "cnn"},
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            }
        },
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "linear"},
                    },
                ],
                "target_configs": {
                    "target_cat_columns": ["Origin"],
                    "target_con_columns": ["Height"],
                },
            }
        },
    ],
    indirect=True,
)
def test_get_model(create_test_config: Configs, create_test_labels):
    test_config = create_test_config
    gc = create_test_config.global_config
    target_labels = create_test_labels

    num_outputs_per_class = train.set_up_num_outputs_per_target(
        target_transformers=target_labels.label_transformers
    )

    inputs_as_dict = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=tuple(create_test_labels.train_labels.keys()),
        valid_ids=tuple(create_test_labels.valid_labels.keys()),
        hooks=None,
    )

    model = train.get_model(
        inputs_as_dict=inputs_as_dict,
        global_config=gc,
        predictor_config=create_test_config.predictor_config,
        num_outputs_per_target=num_outputs_per_class,
    )

    assert len(test_config.input_configs) == 1
    omics_model_type = test_config.input_configs[0].input_type_info.model_type
    _check_model(model_type=omics_model_type, model=model)


def _check_model(model_type: str, model: nn.Module):

    if model_type == "cnn":
        assert isinstance(model, FusionModel)
        assert model.multi_task_branches["Origin"][-1][-1].out_features == 3
        assert model.multi_task_branches["Height"][-1][-1].out_features == 1
        assert isinstance(model.modules_to_fuse["omics_test_genotype"], CNNModel)

    elif model_type == "linear":
        assert isinstance(model.modules_to_fuse["omics_test_genotype"], LinearModel)


def test_get_criterions():

    test_target_columns_dict = {
        "con": ["Height", "BMI"],
        "cat": ["Origin", "HairColor"],
    }

    test_criterions = train._get_criterions(
        test_target_columns_dict,
    )
    for column_name in test_target_columns_dict["con"]:
        assert test_criterions[column_name].func is train._calc_mse

    for column_name in test_target_columns_dict["cat"]:
        assert isinstance(test_criterions[column_name], nn.CrossEntropyLoss)


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
