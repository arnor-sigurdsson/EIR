from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
from ignite.contrib.handlers import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from eir.setup.schemas import GlobalConfig
from eir.train_utils.evaluation import validation_handler
from eir.train_utils.metrics import (
    get_average_history_filepath,
    read_metrics_history_file,
)
from eir.train_utils.utils import get_run_folder, validate_handler_dependencies
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.train_utils.train_handlers import HandlerConfig

logger = get_logger(name=__name__, tqdm_compatible=True)


@validate_handler_dependencies([validation_handler])
def attach_lr_scheduler(
    engine: Engine,
    lr_scheduler: Callable,
    experiment: "Experiment",
) -> None:
    """
    We use Events.ITERATION_COMPLETED for the plateau lr_scheduler to be in sync with
    the evaluation handler (which runs on completed iteration as well). To make sure
    we start at the lower bound, we manually set the LR below (in the case of warmup
    being used).

    We use iteration started for the cycle lr_scheduler, to make sure we start at the
    lower bound of the learning rate when using warmup for the first iteration.
    """
    gc = experiment.configs.global_config

    if gc.lr_schedule.lr_schedule in ["cycle", "cosine"]:
        engine.add_event_handler(
            event_name=Events.ITERATION_STARTED, handler=lr_scheduler
        )

    elif gc.lr_schedule.lr_schedule == "plateau":
        if gc.lr.warmup_steps:
            logger.debug("Setting first iteration optimizer LR to %.0e.", gc.opt.lr_lb)
            update_optimizer_lr(
                lr=gc.opt.lr_lb,
                optimizer=experiment.optimizer,
            )

        step_scheduler_params = _get_reduce_lr_on_plateau_step_params(
            global_config=gc,
            optimizer=experiment.optimizer,
        )
        engine.add_event_handler(
            event_name=Events.ITERATION_COMPLETED,
            handler=_step_reduce_on_plateau_scheduler,
            reduce_on_plateau_scheduler=lr_scheduler,
            **step_scheduler_params
        )
    else:
        raise ValueError()


def _get_reduce_lr_on_plateau_step_params(
    global_config: GlobalConfig, optimizer: Optimizer
) -> Dict:
    run_folder = get_run_folder(output_folder=global_config.be.output_folder)
    validation_history_fpath = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="validation_"
    )

    warmup_steps = _get_warmup_steps_from_cla(
        warmup_steps_arg=global_config.lr.warmup_steps,
        optimizer=optimizer,
    )

    params = {
        "validation_history_fpath": validation_history_fpath,
        "optimizer": optimizer,
        "sample_interval": global_config.ec.sample_interval,
        "lr_upper_bound": global_config.opt.lr,
        "lr_lower_bound": global_config.opt.lr_lb,
        "warmup_steps": warmup_steps,
    }

    return params


def set_up_lr_scheduler(
    handler_config: "HandlerConfig",
) -> Callable:
    exp = handler_config.experiment
    gc = exp.configs.global_config

    lr_lower_bound = gc.opt.lr_lb
    lr_upper_bound = gc.opt.lr

    def _get_cycle_iter_size(warmup_steps_: int) -> int:
        """
        Why this weird max(2, ...)? This is because the `ignite`
        `CosineAnnealingScheduler` expects at least 2 steps, so if we have fewer steps
        than the warmup period, we have max(2, negative number) = 2 for
        compatibility.
        """
        steps = len(exp.train_loader)
        schedule: str = gc.lr.lr_schedule
        assert isinstance(schedule, str)
        if schedule == "cosine":
            steps = max(2, steps * gc.be.n_epochs - warmup_steps_)

        return steps

    def _get_total_num_events(n_epochs: int, iter_per_epoch: int) -> int:
        return n_epochs * iter_per_epoch

    if gc.lr.lr_schedule in ["cycle", "cosine"]:
        warmup_steps = _get_warmup_steps_from_cla(
            warmup_steps_arg=gc.lr.warmup_steps,
            optimizer=exp.optimizer,
        )

        cycle_iter_size = _get_cycle_iter_size(warmup_steps_=warmup_steps)

        lr_scheduler: CosineAnnealingScheduler | ConcatScheduler | ReduceLROnPlateau
        lr_scheduler, lr_scheduler_args = _get_cosine_lr_scheduler(
            optimizer=exp.optimizer,
            lr_upper_bound=lr_upper_bound,
            lr_lower_bound=lr_lower_bound,
            cycle_iter_size=cycle_iter_size,
            schedule=gc.lr.lr_schedule,
        )

        if warmup_steps:
            lr_scheduler, lr_scheduler_args = _attach_warmup_to_scheduler(
                lr_scheduler=lr_scheduler,
                lr_lower_bound=lr_lower_bound,
                lr_upper_bound=lr_upper_bound,
                duration=warmup_steps,
            )

        num_events = _get_total_num_events(
            n_epochs=gc.be.n_epochs, iter_per_epoch=len(exp.train_loader)
        )

        if gc.lr.plot_lr_schedule:
            _plot_lr_schedule(
                lr_scheduler=lr_scheduler,
                num_events=num_events,
                output_folder=handler_config.run_folder,
                lr_scheduler_args=lr_scheduler_args,
            )

    elif gc.lr.lr_schedule == "plateau":
        logger.debug("Plateau patience set to %d.", gc.lr.lr_plateau_patience)

        """
        For compatibility with ignite EarlyStopping handler, we reduce the plateau
        patience steps passed into the ReduceLROnPlateau constructor. This is because
        EarlyStopping uses bad_steps >= patience, while ReduceLROnPlateau uses bad_steps
        > patience. In order to have a common interface when passing in any type of
        patience steps, we are going to reduce the passed in steps here.
        """
        patience_steps = gc.lr.lr_plateau_patience - 1
        lr_scheduler = ReduceLROnPlateau(
            optimizer=exp.optimizer,
            mode="max",
            factor=gc.lr.lr_plateau_factor,
            patience=patience_steps,
            min_lr=lr_lower_bound,
        )

    else:
        raise ValueError()

    return cast(Callable, lr_scheduler)


def _get_cosine_lr_scheduler(
    optimizer: Optimizer,
    lr_upper_bound: float,
    lr_lower_bound: float,
    cycle_iter_size: int,
    schedule: str,
) -> Tuple[CosineAnnealingScheduler, Dict]:
    """
    We return the arguments because the simulate_values are classmethods, which
    we need to pass the arguments too.
    """

    cycle_mult = 1.0
    start_value_mult = 1.0

    if schedule == "cycle":
        cycle_mult = 2.0
        start_value_mult = 1.0

    lr_scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        param_name="lr",
        start_value=lr_upper_bound,
        end_value=lr_lower_bound,
        cycle_size=cycle_iter_size,
        cycle_mult=cycle_mult,
        start_value_mult=start_value_mult,
    )

    cosine_scheduler_kwargs = {
        "optimizer": optimizer,
        "param_name": "lr",
        "start_value": lr_upper_bound,
        "end_value": lr_lower_bound,
        "cycle_size": cycle_iter_size,
        "cycle_mult": cycle_mult,
        "start_value_mult": start_value_mult,
    }

    return lr_scheduler, cosine_scheduler_kwargs


def _get_warmup_steps_from_cla(
    warmup_steps_arg: Optional[Literal["auto"] | int], optimizer: Optimizer
) -> int:
    if warmup_steps_arg is None:
        return 0
    elif warmup_steps_arg == "auto":
        auto_steps = _calculate_auto_warmup_steps(optimizer=optimizer)
        logger.debug(
            "Using calculated %d steps for learning rate warmup due to 'auto' "
            "option for warmup.",
            auto_steps,
        )
        return auto_steps
    else:
        return int(warmup_steps_arg)


def _calculate_auto_warmup_steps(optimizer: Optimizer) -> int:
    def _calc_steps(b2):
        return round(2 / (1 - b2))

    first_param_group = optimizer.param_groups[0]
    if "betas" in first_param_group:
        b2_value = first_param_group["betas"][1]
        return _calc_steps(b2_value)

    return 2000


def _plot_lr_schedule(
    lr_scheduler: Union[ConcatScheduler, CosineAnnealingScheduler],
    num_events: int,
    lr_scheduler_args: dict,
    output_folder: Path,
):
    simulated_vals = np.array(
        lr_scheduler.simulate_values(num_events=num_events, **lr_scheduler_args)
    )

    plt.plot(simulated_vals[:, 0], simulated_vals[:, 1])
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")

    plt.savefig(output_folder / "lr_schedule.pdf")
    plt.close("all")


def _attach_warmup_to_scheduler(
    lr_scheduler: CosineAnnealingScheduler,
    lr_lower_bound: float,
    lr_upper_bound: float,
    duration: int,
) -> Tuple[ConcatScheduler, Dict]:
    """
    We have `patched_duration_for_cosine_end` because attaching a warmup to a cosine
    scheduler seems to 'offset' it's length by one step. This means that if we have
    100 warmup steps, and 125 cosine steps, iteration 225 will be in the next cycle
    of a cosine annealing schedule, meaning the LR upper bound.

    This can be seen by monitoring the iteration and current optimizer LR, which shows
    e.g. (LR=0.01 iteration=225) when printing the values.

    This happens because internally, the `create_lr_scheduler_with_warmup` `ignite`
    function reduces the duration of the warmup by one step. That is if we pass in 100
    warmup steps, steps 1-98 will be in the warmup phase (99 steps in total). This
    happens because of the following line in `ignite` 0.3:

    milestones_values[-1] = (warmup_duration - 2, warmup_end_value - d)

    So since the warmup steps are 1 less, it means that the cosine annealing will be
    one step more, meaning that the last step is a new cycle.

    This can be tested quite easily with the `output_simulated_values` argument,
    which shows the jump in LR in the last cycle if we pass in `durations` unpatched.
    """

    patched_duration_for_cosine_end = duration + 1
    scheduler_w_warmup = create_lr_scheduler_with_warmup(
        lr_scheduler=lr_scheduler,
        warmup_start_value=lr_lower_bound,
        warmup_end_value=lr_upper_bound,
        warmup_duration=patched_duration_for_cosine_end,
    )

    concat_scheduler_args = {
        "schedulers": scheduler_w_warmup.schedulers,
        "durations": [patched_duration_for_cosine_end],
    }

    return scheduler_w_warmup, concat_scheduler_args


def _step_reduce_on_plateau_scheduler(
    engine: Engine,
    optimizer: Optimizer,
    lr_upper_bound: float,
    lr_lower_bound: float,
    sample_interval: int,
    reduce_on_plateau_scheduler: ReduceLROnPlateau,
    validation_history_fpath: Path,
    warmup_steps: Union[None, int],
) -> None:
    """
    We do the warmup manually here because currently ignite does not support warmup
    with ReduceLROnPlateau through create_lr_scheduler_with_warmup because
    ReduceLROnPlateau does not inherit from _LRScheduler.
    """
    iteration = engine.state.iteration

    if warmup_steps is not None and iteration <= warmup_steps:
        cur_lr = calculate_lr_after_linear_step(
            lr_start=lr_lower_bound,
            lr_end=lr_upper_bound,
            warmup_steps=warmup_steps,
            iteration=iteration,
        )
        update_optimizer_lr(lr=cur_lr, optimizer=optimizer)

    else:
        prev_lr = get_optimizer_lr(optimizer=optimizer)
        cur_bad_steps = reduce_on_plateau_scheduler.num_bad_epochs

        if iteration % sample_interval == 0 and not isclose(prev_lr, lr_lower_bound):
            validation_df = read_metrics_history_file(
                file_path=validation_history_fpath
            )
            latest_val_performance = validation_df["perf-average"].iloc[-1]

            reduce_on_plateau_scheduler.step(metrics=latest_val_performance)

            # See comment in set_up_lr_scheduler for +1 explanation
            assert cur_bad_steps is not None
            assert reduce_on_plateau_scheduler.num_bad_epochs is not None
            streamlined_patience = reduce_on_plateau_scheduler.patience + 1
            _log_plateu_bad_step(
                iteration=iteration,
                prev_bad_steps=cur_bad_steps,
                cur_bad_steps=reduce_on_plateau_scheduler.num_bad_epochs,
                patience=streamlined_patience,
            )

            new_lr = get_optimizer_lr(optimizer=optimizer)
            _log_reduce_on_plateu_step(
                reduce_on_plateau_scheduler=reduce_on_plateau_scheduler,
                iteration=iteration,
                prev_lr=prev_lr,
                cur_lr=new_lr,
                patience=streamlined_patience,
            )


def update_optimizer_lr(lr: float, optimizer: Optimizer) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def calculate_lr_after_linear_step(
    lr_start: float,
    lr_end: float,
    warmup_steps: int,
    iteration: int,
) -> float:
    step_size = (lr_end - lr_start) / warmup_steps
    cur_lr = lr_start + (step_size * iteration)

    return cur_lr


def get_optimizer_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _log_plateu_bad_step(
    iteration: int, prev_bad_steps: int, cur_bad_steps: int, patience: int
) -> None:
    if cur_bad_steps > prev_bad_steps:
        logger.debug(
            "Iteration %d: Reduce LR On Plateau: %d / %d",
            iteration,
            cur_bad_steps,
            patience,
        )


def _log_reduce_on_plateu_step(
    reduce_on_plateau_scheduler: ReduceLROnPlateau,
    patience: int,
    iteration: int,
    prev_lr: float,
    cur_lr: float,
) -> None:
    """
    NOTE: The ReduceLROnPlateau works differently from ignite's EarlyStopping in the
    sense that EarlyStopping will trigger when bad steps are >= patience, while
    ReduceLROnPlateau will trigger when bad steps are *>* patience.
    """
    sched = reduce_on_plateau_scheduler

    if not isclose(prev_lr, cur_lr) and cur_lr > sched.min_lrs[0]:
        logger.info(
            "Iteration %d: Reduce LR On Plateau %d / %d. "
            "Reduced learning rate from %.0e to %.0e.",
            iteration,
            patience,
            patience,
            prev_lr,
            cur_lr,
        )
