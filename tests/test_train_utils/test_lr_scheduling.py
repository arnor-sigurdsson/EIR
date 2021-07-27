from math import isclose
from unittest.mock import patch

import pytest
import torch
from ignite.contrib.handlers import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    ParamGroupScheduler,
)
from ignite.engine import Engine, State
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from eir.train_utils import lr_scheduling
from eir.train_utils.lr_scheduling import get_optimizer_lr
from eir.train_utils.train_handlers import HandlerConfig


@pytest.fixture()
def create_dummy_test_optimizer(request):
    """
    The parametrization here feels a bit roundabout, would probably be easier to just
    have this as a function instead of fixture?
    """

    dummy_params = [{"params": torch.randn(10, requires_grad=True)}]

    # default case, if not using indirect parametrization
    if not hasattr(request, "param"):
        return Adam(dummy_params, lr=0.001, betas=(0.9, 0.999))

    optimizer_type = request.param["optimizer_type"]

    if optimizer_type == "Adam":

        b2 = 0.999
        if "b2" in request.param:
            b2 = request.param["b2"]

        adam = Adam(dummy_params, lr=0.001, betas=(0.9, b2))
        return adam, request.param

    elif optimizer_type == "SGD":
        sgd = SGD(dummy_params, lr=0.001)
        return sgd, request.param

    raise ValueError()


def test_get_reduce_lr_on_plateu_step_params(
    test_config_base, create_dummy_test_optimizer
):
    cl_args = test_config_base
    optimizer = create_dummy_test_optimizer

    params = lr_scheduling._get_reduce_lr_on_plateau_step_params(
        cl_args=cl_args, optimizer=optimizer
    )

    assert params["lr_upper_bound"] == cl_args.lr
    assert params["lr_lower_bound"] == cl_args.lr_lb
    assert params["sample_interval"] == cl_args.sample_interval
    assert params["warmup_steps"] == cl_args.warmup_steps
    assert params["optimizer"] == optimizer
    assert params["validation_history_fpath"].name == "validation_average_history.log"


@pytest.fixture()
def get_dummy_handler_config(prep_modelling_test_configs) -> HandlerConfig:
    config, test_config = prep_modelling_test_configs
    cl_args = config.cl_args

    handler_config = HandlerConfig(
        config=config,
        run_folder=test_config.run_path,
        run_name=cl_args.run_name,
        monitoring_metrics=[("tmp_var", "tmp_var")],
    )

    return handler_config


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config",
    [{"custom_cl_args": {"lr_schedule": "plateau", "lr": 1e-3}}],
    indirect=True,
)
def test_set_up_lr_scheduler_plateau(get_dummy_handler_config):
    handler_config = get_dummy_handler_config
    c = handler_config.experiment
    cl_args = c.cl_args

    lr_scheduler = lr_scheduling.set_up_lr_scheduler(handler_config=handler_config)
    assert isinstance(lr_scheduler, ReduceLROnPlateau)

    # Note: Check comment in lr_scheduling.set_up_lr_scheduler for why -1
    expected_patience = cl_args.lr_plateau_patience - 1
    assert lr_scheduler.patience == expected_patience
    assert lr_scheduler.optimizer == c.optimizer


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config",
    [
        {"custom_cl_args": {"lr_schedule": "cycle", "lr": 1e-3}},
        {"custom_cl_args": {"lr_schedule": "cosine", "lr": 1e-3}},
    ],
    indirect=True,
)
def test_set_up_lr_scheduler_cycle(get_dummy_handler_config):
    handler_config = get_dummy_handler_config
    c = handler_config.experiment
    cl_args = c.cl_args

    lr_scheduler = lr_scheduling.set_up_lr_scheduler(handler_config=handler_config)

    _check_warmup_concat_scheduler(
        lr_scheduler=lr_scheduler, warmup_steps=cl_args.warmup_steps
    )

    _check_cosine_scheduler(
        cosine_scheduler=lr_scheduler.schedulers[1],
        lr_schedule=cl_args.lr_schedule,
        cycle_iter_size=len(c.train_loader),
        n_epochs=cl_args.n_epochs,
        warmup_steps=cl_args.warmup_steps,
    )


def _check_warmup_concat_scheduler(
    lr_scheduler: ConcatScheduler, warmup_steps: int
) -> None:
    assert isinstance(lr_scheduler, ConcatScheduler)

    scheduler_list = lr_scheduler.schedulers

    assert len(scheduler_list) == 2
    warmup_scheduler = scheduler_list[0]
    cosine_scheduler = scheduler_list[1]
    assert isinstance(warmup_scheduler, ParamGroupScheduler)
    assert isinstance(cosine_scheduler, CosineAnnealingScheduler)

    assert lr_scheduler.durations[0] == warmup_steps


def _check_cosine_scheduler(
    cosine_scheduler: CosineAnnealingScheduler,
    lr_schedule: str,
    cycle_iter_size,
    n_epochs: int,
    warmup_steps: int,
):
    if lr_schedule == "cycle":
        assert cosine_scheduler.cycle_mult == 2
        assert cosine_scheduler.cycle_size == cycle_iter_size
    elif lr_schedule == "cosine":
        assert cosine_scheduler.cycle_mult == 1
        assert cosine_scheduler.cycle_size == cycle_iter_size * n_epochs - warmup_steps
    else:
        raise ValueError()


@pytest.mark.parametrize("test_schedule", ["cycle", "cosine"])
def test_get_cosine_lr_scheduler_cosine(test_schedule, create_dummy_test_optimizer):
    optimizer = create_dummy_test_optimizer
    lr_ub = lr_scheduling.get_optimizer_lr(optimizer=optimizer)
    lr_lb = 1e-5

    # with warmup 0 and n_epochs = 0 we expect both schedulers to have the same cycle
    # size
    warmup_steps = 0
    n_epochs = 1
    cycle_iter_size = 100

    test_cosine_scheduler, _ = lr_scheduling._get_cosine_lr_scheduler(
        optimizer=optimizer,
        lr_upper_bound=lr_ub,
        lr_lower_bound=lr_lb,
        cycle_iter_size=cycle_iter_size,
        schedule=test_schedule,
    )

    _check_cosine_scheduler(
        cosine_scheduler=test_cosine_scheduler,
        lr_schedule=test_schedule,
        cycle_iter_size=cycle_iter_size,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
    )


@pytest.mark.parametrize(
    "create_dummy_test_optimizer",
    [{"optimizer_type": "Adam"}, {"optimizer_type": "SGD"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "test_input,expected", [(None, 0), ("auto", 2000), ("100", 100)]
)
def test_get_warmup_steps_from_cla_pass(
    test_input, expected, create_dummy_test_optimizer
):
    optimizer, _ = create_dummy_test_optimizer

    test_output = lr_scheduling._get_warmup_steps_from_cla(
        warmup_steps_arg=test_input, optimizer=optimizer
    )
    assert test_output == expected


@pytest.mark.parametrize(
    "create_dummy_test_optimizer",
    [{"optimizer_type": "Adam"}, {"optimizer_type": "SGD"}],
    indirect=True,
)
def test_get_warmup_steps_from_cla_fail(create_dummy_test_optimizer):
    optimizer, _ = create_dummy_test_optimizer

    with pytest.raises(ValueError):
        lr_scheduling._get_warmup_steps_from_cla(
            warmup_steps_arg="fail", optimizer=optimizer
        )


@pytest.mark.parametrize(
    "create_dummy_test_optimizer",
    [
        {"optimizer_type": "Adam", "b2": 0.9},
        {"optimizer_type": "Adam", "b2": 0.99},
        {"optimizer_type": "Adam", "b2": 0.999},
    ],
    indirect=True,
)
def test_calculate_auto_warmup_steps_adaptive(create_dummy_test_optimizer):
    steps_mapping = {0.9: 20, 0.99: 200, 0.999: 2000}

    optimizer, fixture_params = create_dummy_test_optimizer
    warmup_steps = lr_scheduling._calculate_auto_warmup_steps(optimizer=optimizer)

    assert steps_mapping[fixture_params["b2"]] == warmup_steps


@pytest.mark.parametrize("test_schedule", ["cycle", "cosine"])
def ttest_attach_warmup_to_scheduler(test_schedule, create_dummy_test_optimizer):
    optimizer = create_dummy_test_optimizer
    lr_ub = lr_scheduling.get_optimizer_lr(optimizer=optimizer)
    lr_lb = 1e-5

    cycle_iter_size = 100
    warmup_steps = 25

    test_cosine_scheduler, _ = lr_scheduling._get_cosine_lr_scheduler(
        optimizer=optimizer,
        lr_upper_bound=lr_ub,
        lr_lower_bound=lr_lb,
        cycle_iter_size=cycle_iter_size,
        schedule=test_schedule,
    )

    test_cosine_scheduler_w_warmup, _ = lr_scheduling._attach_warmup_to_scheduler(
        lr_scheduler=test_cosine_scheduler,
        lr_lower_bound=lr_lb,
        lr_upper_bound=lr_ub,
        duration=warmup_steps,
    )

    _check_warmup_concat_scheduler(
        lr_scheduler=test_cosine_scheduler_w_warmup, warmup_steps=warmup_steps
    )


@pytest.fixture()
def create_test_ignite_engine():
    def dummy_process_func(engine, x):
        return x

    test_engine = Engine(process_function=dummy_process_func)
    test_engine.state = State()

    # note the state is 1-based (starting at 1)
    test_engine.state.iteration = 1
    test_engine.state.epoch = 1

    return test_engine


@pytest.fixture()
def create_test_plateau_scheduler(create_dummy_test_optimizer):
    optimizer = create_dummy_test_optimizer
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        patience=2,
        min_lr=1e-6,
    )
    return scheduler


def test_step_reduce_on_plateau_scheduler(
    create_test_ignite_engine,
    create_dummy_test_optimizer,
    create_test_plateau_scheduler,
    create_dummy_val_history_file,
):
    def step_func(mock_return_value):

        # NOTE: A bit hacky, but lr_scheduling is getting it's dataframe by calling
        # a function in metrics.py, that's why we need to mock that
        patch_target = "eir.train_utils.metrics.pd.Series.iloc"
        with patch(target=patch_target, autospec=True) as m:

            # What we really are mocking is the __getitem__ called last in
            # validation_df["perf-average"].iloc[-1], but we cannot mock it directly
            m.__getitem__.return_value = mock_return_value

            lr_scheduling._step_reduce_on_plateau_scheduler(
                engine=test_engine,
                optimizer=test_optimizer,
                lr_upper_bound=lr_ub,
                lr_lower_bound=lr_lb,
                sample_interval=sample_interval,
                reduce_on_plateau_scheduler=test_scheduler,
                validation_history_fpath=hist_file,
                warmup_steps=warmup_steps,
            )

        test_engine.state.iteration += 1

    test_engine = create_test_ignite_engine
    test_optimizer = create_dummy_test_optimizer
    test_scheduler = create_test_plateau_scheduler

    lr_ub = get_optimizer_lr(optimizer=test_optimizer)
    lr_lb = test_scheduler.min_lrs[0]

    sample_interval = 1
    hist_file = create_dummy_val_history_file
    warmup_steps = 20

    # note that optimizer starts with base lr, then lr is modified according to
    # upper and lower bounds,
    assert get_optimizer_lr(test_optimizer) == lr_ub

    # step 1
    step_func(0)
    assert get_optimizer_lr(test_optimizer) == lr_lb + ((lr_ub - lr_lb) / warmup_steps)

    # step 2 - 19
    for i in range(warmup_steps - 1):
        step_func(-i)

    # step 20 - 100
    for i in range(100 - warmup_steps):
        step_func(-i)

    assert isclose(get_optimizer_lr(optimizer=test_optimizer), lr_lb)


@pytest.fixture()
def create_dummy_val_history_file(tmp_path):
    tmp_file = tmp_path / "dummy_val_history.csv"
    with open(str(tmp_file), "w") as outfile:
        outfile.write("iteration,loss-average,perf-average\n")

    return tmp_file
