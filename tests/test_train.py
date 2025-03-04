from unittest.mock import create_autospec, patch

import pytest
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from eir import train
from eir.data_load.data_utils import get_finite_train_sampler
from eir.models.input.omics.omics_models import CNNModel, LinearModel
from eir.models.meta.meta import MetaModel
from eir.models.model_setup import get_model
from eir.setup.config import Configs
from eir.setup.input_setup import set_up_inputs_for_training
from eir.setup.output_setup import set_up_outputs_for_training
from eir.setup.schemas import GlobalConfig, OptimizationConfig
from eir.train_utils import optim


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
                "global_configs": {
                    "optimization": {
                        "lr": 1e-03,
                    },
                    "basic_experiment": {
                        "output_folder": "test_get_default_experiment",
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_default_experiment(
    create_test_config: Configs,
):
    test_configs = create_test_config

    hooks = train.get_default_hooks(configs=test_configs)
    default_experiment = train.get_default_experiment(configs=test_configs, hooks=hooks)
    assert default_experiment.configs == test_configs

    assert len(default_experiment.criteria) == 1

    assert len(default_experiment.inputs) == 1
    assert set(default_experiment.inputs.keys()) == {"test_genotype"}

    assert len(default_experiment.outputs) == 1
    assert default_experiment.outputs["test_output_tabular"].target_columns == {
        "cat": ["Origin"],
        "con": [],
    }


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {
                    "optimization": {"lr": 1e-03},
                    "training_control": {
                        "weighted_sampling_columns": ["all"],
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_get_dataloaders(
    create_test_config: Configs, create_test_data, create_test_datasets
):
    test_config = create_test_config
    gc = test_config.global_config
    gc.training_control.weighted_sampling_columns = ["test_output_tabular__Origin"]

    train_dataset, valid_dataset = create_test_datasets
    train_sampler = get_finite_train_sampler(
        columns_to_sample=gc.training_control.weighted_sampling_columns,
        train_dataset=train_dataset,
    )

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=gc.be.batch_size,
    )

    assert train_dataloader.batch_size == gc.be.batch_size
    assert valid_dataloader.batch_size == gc.be.batch_size
    assert isinstance(train_dataloader.sampler, WeightedRandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)

    train_dataloader, valid_dataloader = train.get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=None,
        valid_dataset=valid_dataset,
        batch_size=gc.be.batch_size,
    )

    assert isinstance(train_dataloader.sampler, RandomSampler)
    assert isinstance(valid_dataloader.sampler, SequentialSampler)


def test_get_optimizer():
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1 = nn.Linear(10, 10)

        def forward(self, x):
            pass

    model = FakeModel()

    gc_adamw = create_autospec(GlobalConfig, instance=True)
    gc_adamw.optimization = create_autospec(OptimizationConfig, instance=True)
    gc_adamw.optimization.optimizer = "adamw"
    gc_adamw.optimization.lr = 1e-03
    gc_adamw.optimization.b1 = 0.9
    gc_adamw.optimization.b2 = 0.999
    gc_adamw.wd = 1e-04
    gc_adamw.opt = create_autospec(OptimizationConfig, instance=True)
    gc_adamw.opt.optimizer = "adamw"
    gc_adamw.opt.lr = 1e-03
    gc_adamw.opt.b1 = 0.9
    gc_adamw.opt.b2 = 0.999
    gc_adamw.opt.wd = 1e-04

    adamw_optimizer = optim.get_optimizer(
        model=model,
        loss_callable=lambda x: x,
        global_config=gc_adamw,
    )
    assert isinstance(adamw_optimizer, AdamW)

    gc_sgdm = create_autospec(GlobalConfig, instance=True)
    gc_sgdm.optimization = create_autospec(OptimizationConfig, instance=True)
    gc_sgdm.optimization.optimizer = "sgdm"
    gc_sgdm.optimization.lr = 1e-03
    gc_sgdm.optimization.wd = 1e-04
    gc_sgdm.opt = create_autospec(OptimizationConfig, instance=True)
    gc_sgdm.opt.optimizer = "sgdm"
    gc_sgdm.opt.lr = 1e-03
    gc_sgdm.opt.wd = 1e-04

    sgdm_optimizer = optim.get_optimizer(
        model=model,
        loss_callable=lambda x: x,
        global_config=gc_sgdm,
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
                        "model_config": {"model_type": "cnn"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            }
        },
        {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    }
                ],
            }
        },
    ],
    indirect=True,
)
def test_get_model(create_test_config: Configs, create_test_labels):
    test_config = create_test_config
    gc = create_test_config.global_config
    target_labels = create_test_labels

    train_ids = tuple(create_test_labels.train_labels["ID"])
    valid_ids = tuple(create_test_labels.valid_labels["ID"])

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=inputs_as_dict,
        target_transformers=target_labels.label_transformers,
    )

    model = get_model(
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
        fusion_config=test_config.fusion_config,
        global_config=gc,
    )

    assert len(test_config.input_configs) == 1
    omics_model_type = test_config.input_configs[0].model_config.model_type
    _check_model(model_type=omics_model_type, model=model)


def _check_model(model_type: str, model: nn.Module):
    if model_type == "cnn":
        assert isinstance(model, MetaModel)
        output_module = model.output_modules.test_output_tabular
        origin_module = output_module.multi_task_branches["Origin"]
        assert origin_module[-1][-1].out_features == 3

        height_module = output_module.multi_task_branches["Height"]
        assert height_module[-1][-1].out_features == 1

        assert isinstance(model.input_modules["test_genotype"], CNNModel)

    elif model_type == "linear":
        assert isinstance(model.input_modules["test_genotype"], LinearModel)
