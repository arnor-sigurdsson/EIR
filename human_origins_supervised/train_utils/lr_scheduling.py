from argparse import Namespace
from math import isclose
from pathlib import Path
from typing import Tuple, Union, Dict, TYPE_CHECKING, overload

import numpy as np
from aislib.misc_utils import get_logger
from ignite.contrib.handlers import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from human_origins_supervised.train_utils.utils import get_run_folder
from human_origins_supervised.train_utils.metrics import read_metrics_history_file

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.train import Config

logger = get_logger(name=__name__, tqdm_compatible=True)


def attach_lr_scheduler(
    engine: Engine,
    lr_scheduler: Union[ConcatScheduler, CosineAnnealingScheduler, ReduceLROnPlateau],
    config: "Config",
) -> None:
    """
    We use Events.ITERATION_COMPLETED for the plateau lr_scheduler to be in sync with
    the evaluation handler (which runs on completed iteration as well).

    We use iteration started for the cycle lr_scheduler, to make sure we start at the
    lower bound of the learning rate when using warmup for the first iteration.
    """
    cl_args = config.cl_args

    if cl_args.lr_schedule in ["cycle", "cosine"]:
        engine.add_event_handler(
            event_name=Events.ITERATION_STARTED, handler=lr_scheduler
        )

    elif cl_args.lr_schedule == "plateau":

        step_scheduler_params = _get_reduce_lr_on_plateu_step_params(
            cl_args=cl_args, optimizer=config.optimizer
        )
        engine.add_event_handler(
            event_name=Events.ITERATION_COMPLETED,
            handler=_step_reduce_on_plateau_scheduler,
            reduce_on_plateau_scheduler=lr_scheduler,
            **step_scheduler_params
        )
    else:
        raise ValueError()


def _get_reduce_lr_on_plateu_step_params(
    cl_args: Namespace, optimizer: Optimizer
) -> Dict:
    eval_history_fpath = get_run_folder(cl_args.run_name) / "v_average_history.log"

    warmup_steps = _get_warmup_steps_from_cla(
        warmup_steps_arg=cl_args.warmup_steps, optimizer=optimizer
    )

    params = {
        "eval_history_fpath": eval_history_fpath,
        "optimizer": optimizer,
        "sample_interval": cl_args.sample_interval,
        "lr_upper_bound": cl_args.lr,
        "lr_lower_bound": cl_args.lr_lb,
        "warmup_steps": warmup_steps,
    }

    return params


def set_up_lr_scheduler(
    handler_config: "HandlerConfig",
) -> Union[ConcatScheduler, CosineAnnealingScheduler, ReduceLROnPlateau]:

    c = handler_config.config
    cl_args = c.cl_args

    lr_lower_bound = c.cl_args.lr_lb
    lr_upper_bound = c.cl_args.lr

    def _get_cycle_iter_size(warmup_steps_: int) -> int:
        steps = len(c.train_loader)
        if cl_args.lr_schedule == "cosine":
            steps = steps * c.cl_args.n_epochs - warmup_steps_

        return steps

    def _get_total_num_events(n_epochs: int, iter_per_epoch: int) -> int:
        return n_epochs * iter_per_epoch

    if cl_args.lr_schedule in ["cycle", "cosine"]:
        warmup_steps = _get_warmup_steps_from_cla(
            warmup_steps_arg=cl_args.warmup_steps, optimizer=c.optimizer
        )

        cycle_iter_size = _get_cycle_iter_size(warmup_steps_=warmup_steps)

        lr_scheduler, lr_scheduler_args = _get_cosine_lr_scheduler(
            optimizer=c.optimizer,
            lr_upper_bound=lr_upper_bound,
            lr_lower_bound=lr_lower_bound,
            cycle_iter_size=cycle_iter_size,
            schedule=cl_args.lr_schedule,
        )

        if warmup_steps:
            lr_scheduler, lr_scheduler_args = _attach_warmup_to_scheduler(
                lr_scheduler=lr_scheduler,
                lr_lower_bound=lr_lower_bound,
                lr_upper_bound=lr_upper_bound,
                duration=warmup_steps,
            )

        num_events = _get_total_num_events(
            n_epochs=cl_args.n_epochs, iter_per_epoch=len(c.train_loader)
        )
        _plot_lr_schedule(
            lr_scheduler=lr_scheduler,
            num_events=num_events,
            output_folder=handler_config.run_folder,
            lr_scheduler_args=lr_scheduler_args,
        )

    elif cl_args.lr_schedule == "plateau":
        patience_steps = _calc_plateu_patience(
            steps_per_epoch=len(c.train_loader), sample_interval=cl_args.sample_interval
        )
        logger.info("Plateau patience set to %d.", patience_steps)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=c.optimizer,
            mode="min",
            patience=patience_steps,
            min_lr=lr_lower_bound,
        )

    else:
        raise ValueError()

    return lr_scheduler


def _get_cosine_lr_scheduler(
    optimizer: Optimizer,
    lr_upper_bound: float,
    lr_lower_bound: float,
    cycle_iter_size: int,
    schedule: str,
) -> Tuple[CosineAnnealingScheduler, Dict]:

    """
    Note that the full cycle of the linear lr_scheduler increases and decreases, hence
    we use durations as cycle_iter_size // 2 to make the first lr_scheduler only go for
    the increasing phase.

    We return the arguments because the simulate_values are classmethods, which
    we need to pass the arguments too.
    """

    cosine_scheduler_kwargs = {
        "optimizer": optimizer,
        "param_name": "lr",
        "start_value": lr_upper_bound,
        "end_value": lr_lower_bound,
        "cycle_size": cycle_iter_size,
    }

    if schedule == "cycle":
        cosine_scheduler_kwargs["cycle_mult"] = 2
        cosine_scheduler_kwargs["start_value_mult"] = 1

    lr_scheduler = CosineAnnealingScheduler(**cosine_scheduler_kwargs)

    return lr_scheduler, cosine_scheduler_kwargs


@overload
def _get_warmup_steps_from_cla(warmup_steps_arg: None, optimizer: Optimizer) -> None:
    ...


@overload
def _get_warmup_steps_from_cla(warmup_steps_arg: str, optimizer: Optimizer) -> int:
    ...


def _get_warmup_steps_from_cla(warmup_steps_arg, optimizer):
    if warmup_steps_arg is None:
        return warmup_steps_arg
    elif warmup_steps_arg == "auto":
        auto_steps = _calculate_auto_warmup_steps(optimizer=optimizer)
        logger.info(
            "Using %d steps for learning rate due to 'auto' option warmup.", auto_steps
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
    lr_scheduler_args: Dict,
    output_folder: Path,
):

    simulated_vals = np.array(
        lr_scheduler.simulate_values(num_events=num_events, **lr_scheduler_args)
    )

    plt.plot(simulated_vals[:, 0], simulated_vals[:, 1])
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")

    plt.savefig(output_folder / "lr_schedule.png")
    plt.close("all")


def _calc_plateu_patience(steps_per_epoch: int, sample_interval: int):
    """
    For example if we have 15000 iterations and sample every 1000 we have
    (15000 / 1000) * 2 = 30 patience steps.

    In the case we have a sample interval higher than steps per epoch, we
    make sure that the patience is at least 2 epochs as well.
    """
    num_epochs_patience = 2
    samples_per_epoch = max((1, int(steps_per_epoch / sample_interval)))
    return samples_per_epoch * num_epochs_patience


def _attach_warmup_to_scheduler(
    lr_scheduler: CosineAnnealingScheduler,
    lr_lower_bound: float,
    lr_upper_bound: float,
    duration: int,
) -> Tuple[ConcatScheduler, Dict]:

    scheduler_w_warmup = create_lr_scheduler_with_warmup(
        lr_scheduler=lr_scheduler,
        warmup_start_value=lr_lower_bound,
        warmup_end_value=lr_upper_bound,
        warmup_duration=duration,
    )

    concat_scheduler_args = {
        "schedulers": scheduler_w_warmup.schedulers,
        "durations": [duration],
    }

    return scheduler_w_warmup, concat_scheduler_args


def _step_reduce_on_plateau_scheduler(
    engine: Engine,
    optimizer: Optimizer,
    lr_upper_bound: float,
    lr_lower_bound: float,
    sample_interval: int,
    reduce_on_plateau_scheduler: ReduceLROnPlateau,
    eval_history_fpath: Path,
    warmup_steps: Union[None, int],
) -> None:
    """
    We do the warmup manually here because currently ignite does not support warmup
    with ReduceLROnPlateau through create_lr_scheduler_with_warmup because
    ReduceLROnPlateau does not inherit from _LRScheduler.

    TODO:   Possibly use average performance here as measure of whether to step (instead
            of loss)?
    """
    iteration = engine.state.iteration

    # manual warmup
    if warmup_steps is not None and iteration <= warmup_steps:
        step_size = (lr_upper_bound - lr_lower_bound) / warmup_steps
        cur_lr = lr_lower_bound + step_size * iteration
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

    else:
        cur_lr = get_optimizer_lr(optimizer=optimizer)

        if iteration % sample_interval == 0 and not isclose(cur_lr, lr_lower_bound):
            _log_reduce_on_plateu_step(reduce_on_plateau_scheduler, iteration)

            eval_df = read_metrics_history_file(eval_history_fpath)
            latest_val_loss = eval_df["v_loss-average"].iloc[-1]
            reduce_on_plateau_scheduler.step(latest_val_loss)


def get_optimizer_lr(optimizer: Optimizer):
    return optimizer.param_groups[0]["lr"]


def _log_reduce_on_plateu_step(
    reduce_on_plateau_scheduler: ReduceLROnPlateau, iteration: int
) -> None:
    sched = reduce_on_plateau_scheduler

    prev_lr = get_optimizer_lr(optimizer=sched.optimizer)
    new_lr = prev_lr * sched.factor
    if sched.num_bad_epochs >= sched.patience and prev_lr > sched.min_lrs[0]:
        logger.info(
            "Iter %d: Reducing learning rate from %.0e to %.0e.",
            iteration,
            prev_lr,
            new_lr,
        )
