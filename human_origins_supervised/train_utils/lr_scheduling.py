from pathlib import Path
from typing import Tuple, Union, Dict, TYPE_CHECKING

import numpy as np
from aislib.misc_utils import get_logger
from ignite.contrib.handlers import (
    ConcatScheduler,
    CosineAnnealingScheduler,
    LinearCyclicalScheduler,
)
from ignite.engine import Engine, Events
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from human_origins_supervised.train_utils.utils import check_if_iteration_sample

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.train import Config

logger = get_logger(name=__name__, tqdm_compatible=True)


def _get_cyclic_lr_scheduler(
    optimizer, start_lr, end_lr, cycle_iter_size, do_warmup: bool = True
) -> Tuple[Union[ConcatScheduler, CosineAnnealingScheduler], Dict]:

    """
    Note that the full cycle of the linear lr_scheduler increases and decreases, hence
    we use durations as cycle_iter_size // 2 to make the first lr_scheduler only go for
    the increasing phase.

    We return the arguments because the simulate_values are classmethods, which
    we need to pass the arguments too.
    """

    warmup_scheduler = LinearCyclicalScheduler(
        optimizer,
        "lr",
        start_value=end_lr,
        end_value=start_lr,
        cycle_size=cycle_iter_size,
    )

    cosine_scheduler_kwargs = {
        "optimizer": optimizer,
        "param_name": "lr",
        "start_value": start_lr,
        "end_value": end_lr,
        "cycle_size": cycle_iter_size,
        "cycle_mult": 2,
        "start_value_mult": 1,
    }
    cosine_anneal_scheduler = CosineAnnealingScheduler(**cosine_scheduler_kwargs)

    if do_warmup:
        concat_scheduler_kwargs = {
            "schedulers": [warmup_scheduler, cosine_anneal_scheduler],
            "durations": [cycle_iter_size // 2],
        }
        concat_scheduler = ConcatScheduler(**concat_scheduler_kwargs)
        return concat_scheduler, concat_scheduler_kwargs

    return cosine_anneal_scheduler, cosine_scheduler_kwargs


def _log_reduce_on_plateu_step(
    reduce_on_plateau_scheduler: ReduceLROnPlateau, iteration: int
) -> None:
    sched = reduce_on_plateau_scheduler

    prev_lr = sched.optimizer.param_groups[0]["lr"]
    new_lr = prev_lr * sched.factor
    if sched.num_bad_epochs >= sched.patience:
        logger.info(
            "Iter %d: Reducing learning rate from %.0e to %.0e.",
            iteration,
            prev_lr,
            new_lr,
        )


def _step_reduce_on_plateau_scheduler(
    engine: Engine, config: "Config", reduce_on_plateau_scheduler: ReduceLROnPlateau
) -> None:
    cl_args = config.cl_args
    iteration = engine.state.iteration

    n_iters_per_epoch = len(config.train_loader)
    do_step = check_if_iteration_sample(
        iteration, cl_args.sample_interval, n_iters_per_epoch, cl_args.n_epochs
    )

    if do_step:
        _log_reduce_on_plateu_step(reduce_on_plateau_scheduler, iteration)
        reduce_on_plateau_scheduler.step(engine.state.metrics["v_loss"])


def _plot_lr_schedule(
    lr_scheduler: Union[ConcatScheduler, CosineAnnealingScheduler],
    n_epochs: int,
    cycle_iter_size: int,
    lr_scheduler_args: Dict,
    output_folder: Path,
):

    simulated_vals = np.array(
        lr_scheduler.simulate_values(
            num_events=n_epochs * cycle_iter_size, **lr_scheduler_args
        )
    )

    plt.plot(simulated_vals[:, 0], simulated_vals[:, 1])
    plt.savefig(output_folder / "lr_schedule.png")
    plt.close()


def set_up_scheduler(
    handler_config: "HandlerConfig"
) -> Union[ConcatScheduler, CosineAnnealingScheduler, ReduceLROnPlateau]:

    c = handler_config.config
    cl_args = c.cl_args

    if cl_args.lr_schedule == "cycle":
        lr_scheduler, lr_scheduler_args = _get_cyclic_lr_scheduler(
            c.optimizer, c.cl_args.lr, c.cl_args.lr_lb, len(c.train_loader)
        )

        _plot_lr_schedule(
            lr_scheduler=lr_scheduler,
            n_epochs=cl_args.n_epochs,
            cycle_iter_size=len(c.train_loader),
            output_folder=handler_config.run_folder,
            lr_scheduler_args=lr_scheduler_args,
        )

    else:
        lr_scheduler = ReduceLROnPlateau(
            c.optimizer, "min", patience=10, min_lr=c.cl_args.lr_lb
        )

    return lr_scheduler


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

    if config.cl_args.lr_schedule == "cycle":
        engine.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    else:
        engine.add_event_handler(
            Events.ITERATION_COMPLETED,
            _step_reduce_on_plateau_scheduler,
            config=config,
            reduce_on_plateau_scheduler=lr_scheduler,
        )
