import inspect
from unittest.mock import patch

from eir.train import Experiment
from eir.train_utils import train_handlers
from eir.train_utils.ignite_port.events import Events
from tests.setup_tests.fixtures_create_experiment import ModelTestConfig

al_prep_modelling_test_configs = tuple[Experiment, ModelTestConfig]


def test_unflatten_engine_metrics_dict():
    test_step_base = {
        "test_output_tabular": {
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
    assert (
        test_output["test_output_tabular"]["Origin"]["test_output_Origin_mcc"] == 0.99
    )
    assert (
        test_output["test_output_tabular"]["Height"]["test_output_Height_pcc"] == 0.99
    )

    assert (
        test_output["test_output_tabular"]["Origin"]["test_output_Origin_loss"] == 0.11
    )
    assert (
        test_output["test_output_tabular"]["Height"]["test_output_Height_rmse"] == 0.11
    )


def test_get_attribution_handler_and_event_no_act_sample_factor():
    f = train_handlers._get_attribution_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 10,
        "attributions_every_sample_factor": 0,
        "early_stopping_patience": 5,
    }

    # Check attributions_every_sample_factor = 0 causing acts computed at end only
    attribution_handler_callable, attribution_event = f(**base_kwargs)
    assert attribution_event == Events.COMPLETED


@patch("eir.train_utils.train_handlers.Events", autospec=True)
def test_get_attribution_handler_and_event_only_interval(patched_events):
    f = train_handlers._get_attribution_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 10,
        "attributions_every_sample_factor": 0,
        "early_stopping_patience": 5,
    }

    _ = f(**{**base_kwargs, **{"attributions_every_sample_factor": 2}})
    assert patched_events.ITERATION_COMPLETED.call_count == 1
    assert patched_events.ITERATION_COMPLETED.call_args.kwargs["every"] == 20


def test_get_attribution_handler_and_event_interval_and_end():
    f = train_handlers._get_attribution_handler_and_event
    base_kwargs = {
        "iter_per_epoch": 10,
        "n_epochs": 5,
        "sample_interval_base": 7,
        "attributions_every_sample_factor": 0,
        "early_stopping_patience": None,
    }

    attribution_handler_callable, attribution_event = f(
        **{**base_kwargs, **{"attributions_every_sample_factor": 2}}
    )
    assert len(attribution_event) == 2
    assert attribution_event[0] == Events.ITERATION_COMPLETED
    assert attribution_event[1] == Events.COMPLETED


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
