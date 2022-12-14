import inspect
from unittest.mock import patch

import pytest
from ignite.engine import Events

from eir.setup.config import Configs
from eir.train import train
from eir.train_utils import train_handlers
from tests.conftest import al_prep_modelling_test_configs


def test_unflatten_engine_metrics_dict():
    test_step_base = {
        "test_output": {
            "Origin": {"test_output_Origin_mcc": 0.9, "test_output_Origin_loss": 0.1},
            "Height": {"test_output_Height_pcc": 0.9, "test_output_Height_rmse": 0.1},
        }
    }
    test_flat_metrics_dict = {
        "test_output_Origin_mcc": 0.99,
        "test_output_Origin_loss": 0.11,
        "test_output_Height_pcc": 0.99,
        "test_output_Height_rmse": 0.11,
    }

    test_output = train_handlers._unflatten_engine_metrics_dict(
        step_base=test_step_base, engine_metrics_dict=test_flat_metrics_dict
    )

    # we want to make sure the original values are present
    assert test_output["test_output"]["Origin"]["test_output_Origin_mcc"] == 0.99
    assert test_output["test_output"]["Height"]["test_output_Height_pcc"] == 0.99

    assert test_output["test_output"]["Origin"]["test_output_Origin_loss"] == 0.11
    assert test_output["test_output"]["Height"]["test_output_Height_rmse"] == 0.11


def test_get_activation_handler_and_event_no_act_sample_factor():
    f = train_handlers._get_activation_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 10,
        "act_every_sample_factor": 0,
        "early_stopping_patience": 5,
    }

    # Check act_every_sample_factor = 0 causing acts computed at end only
    activation_handler_callable, activation_event = f(**base_kwargs)
    assert activation_event == Events.COMPLETED


@patch("eir.train_utils.train_handlers.Events", autospec=True)
def test_get_activation_handler_and_event_only_interval(patched_events):
    f = train_handlers._get_activation_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 10,
        "act_every_sample_factor": 0,
        "early_stopping_patience": 5,
    }

    _ = f(**{**base_kwargs, **{"act_every_sample_factor": 2}})
    assert patched_events.ITERATION_COMPLETED.call_count == 1
    assert patched_events.ITERATION_COMPLETED.call_args.kwargs["every"] == 20


def test_get_activation_handler_and_event_interval_and_end():
    f = train_handlers._get_activation_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 7,
        "act_every_sample_factor": 0,
        "early_stopping_patience": None,
    }

    activation_handler_callable, activation_event = f(
        **{**base_kwargs, **{"act_every_sample_factor": 2}}
    )
    assert len(activation_event) == 2
    assert activation_event[0] == Events.ITERATION_COMPLETED
    assert activation_event[1] == Events.COMPLETED


def test_get_early_stopping_event_kwargs():
    event_kwargs_no_buffer = train_handlers._get_early_stopping_event_filter_kwargs(
        early_stopping_iteration_buffer=None, sample_interval=100
    )
    assert event_kwargs_no_buffer["every"] == 100

    event_kwargs_with_buffer = train_handlers._get_early_stopping_event_filter_kwargs(
        early_stopping_iteration_buffer=1000, sample_interval=100
    )
    early_buffer_func = event_kwargs_with_buffer["event_filter"]

    assert early_buffer_func(None, 200) is False
    assert inspect.getclosurevars(early_buffer_func).nonlocals["has_checked"] is False
    assert early_buffer_func(None, 2000) is True
    assert inspect.getclosurevars(early_buffer_func).nonlocals["has_checked"] is True
    assert early_buffer_func(None, 2500) is True


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Linear
        {
            "injections": {
                "global_configs": {
                    "output_folder": "test_hparam_summary_writer",
                    "lr": 1e-03,
                    "n_epochs": 4,
                    "sample_interval": 100,
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@patch("eir.train_utils.train_handlers.SummaryWriter", autospec=True)
def test_add_hparams_to_tensorboard(
    patched_writer, prep_modelling_test_configs: al_prep_modelling_test_configs
):
    test_experiment, *_ = prep_modelling_test_configs
    global_config = test_experiment.configs.global_config

    train(experiment=test_experiment)

    expected = ["lr", "batch_size"]
    random = ["random_1", "random_2"]
    test_hparam_keys = expected + random

    train_handlers.add_hparams_to_tensorboard(
        h_params=test_hparam_keys, experiment=test_experiment, writer=patched_writer
    )
    assert patched_writer.add_hparams.call_count == 1

    hparam_kwarg: dict = patched_writer.add_hparams.call_args.kwargs["hparam_dict"]
    assert set(hparam_kwarg.keys()) == set(expected)
    assert hparam_kwarg["batch_size"] == global_config.batch_size
    assert hparam_kwarg["lr"] == global_config.lr

    metric_dict_kwarg = patched_writer.add_hparams.call_args.kwargs["metric_dict"]
    assert metric_dict_kwarg["best_overall_performance"] > 0.8


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
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_generate_hparam_dict(create_test_config: Configs):
    test_configs = create_test_config

    global_config = test_configs.global_config

    expected = ["lr", "batch_size"]
    random = ["random_1", "random_2"]
    test_hparam_keys = expected + random

    hparam_dict = train_handlers._generate_h_param_dict(
        global_config=global_config, h_params=test_hparam_keys
    )
    assert set(hparam_dict.keys()) == set(expected)
    assert hparam_dict.get("batch_size") == global_config.batch_size
    assert hparam_dict.get("lr") == global_config.lr
