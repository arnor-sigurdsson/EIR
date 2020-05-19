from math import isclose

import pytest
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, State

from human_origins_supervised.train_utils import lr_scheduling


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


def test_get_reduce_lr_on_plateu_step_params():
    pass


def test_set_up_lr_scheduler():
    pass


def test_get_cosine_lr_scheduler():
    pass


@pytest.mark.parametrize(
    "create_dummy_test_optimizer",
    [{"optimizer_type": "Adam"}, {"optimizer_type": "SGD"}],
    indirect=True,
)
@pytest.mark.parametrize(
    "test_input,expected", [(None, None), ("auto", 2000), ("100", 100)]
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


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"steps_per_epoch": 15000, "sample_interval": 1000}, 30),  # 15 per epoch
        ({"steps_per_epoch": 100, "sample_interval": 50}, 4),  # 2 per epoch
        ({"steps_per_epoch": 100, "sample_interval": 200}, 2),  # 0.5 per epoch, min 1
        ({"steps_per_epoch": 14591, "sample_interval": 3000}, 8),  # 4 per epoch
    ],
)
def test_calc_plateau_patience(test_input, expected):
    patience = lr_scheduling._calc_plateu_patience(**test_input)
    assert patience == expected


def test_attach_warmup_to_scheduler():
    pass


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
        optimizer=optimizer, mode="min", patience=2, min_lr=1e-6
    )
    return scheduler


def test_step_reduce_on_plateau_scheduler(
    create_test_ignite_engine,
    create_dummy_test_optimizer,
    create_test_plateau_scheduler,
    create_dummy_val_history_file,
):
    def _get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    def step_func():
        lr_scheduling._step_reduce_on_plateau_scheduler(
            engine=test_engine,
            optimizer=test_optimizer,
            lr_upper_bound=lr_up,
            lr_lower_bound=lr_lb,
            sample_interval=sample_interval,
            reduce_on_plateau_scheduler=test_scheduler,
            eval_history_fpath=hist_file,
            warmup_steps=warmup_steps,
        )
        test_engine.state.iteration += 1

    test_engine = create_test_ignite_engine
    test_optimizer = create_dummy_test_optimizer
    test_scheduler = create_test_plateau_scheduler

    lr_up = _get_lr(optimizer=test_optimizer)
    lr_lb = test_scheduler.min_lrs[0]

    sample_interval = 1
    hist_file = create_dummy_val_history_file
    warmup_steps = 20

    # note that optimizer starts with base lr, then lr is modified according to
    # upper and lower bounds,
    # TODO: Make more explicit where the base_lr is coming from instead of hardcoding.
    assert _get_lr(test_optimizer) == lr_up

    # step 1
    step_func()
    assert _get_lr(test_optimizer) == lr_lb + ((lr_up - lr_lb) / warmup_steps)

    # step 2 - 19
    for i in range(warmup_steps - 1):
        step_func()

    # step 20 - 100
    for i in range(100 - warmup_steps):
        step_func()

    assert isclose(_get_lr(optimizer=test_optimizer), lr_lb)


@pytest.fixture()
def create_dummy_val_history_file(tmp_path):
    # Note that this has a growing loss, so e.g. plateau scheduler with mode='min'
    # should get called
    tmp_file = tmp_path / "dummy_val_history.csv"
    with open(str(tmp_file), "w") as outfile:
        outfile.write("iteration,v_loss-average\n")

        for iteration in range(100):
            outfile.write(f"{iteration},{float(iteration)*2}\n")

    return tmp_file
