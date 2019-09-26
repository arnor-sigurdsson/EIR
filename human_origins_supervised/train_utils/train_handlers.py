import csv
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np
from aislib.misc_utils import get_logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage

from human_origins_supervised.train_utils.activation_analysis import (
    activation_analysis_handler,
)
from human_origins_supervised.train_utils.benchmark import benchmark
from human_origins_supervised.train_utils.evaluation import evaluation_handler
from human_origins_supervised.train_utils.metric_funcs import get_train_metrics
from human_origins_supervised.train_utils.utils import check_if_iteration_sample
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

logger = get_logger(__name__)


class MyRunningAverage(RunningAverage):
    def __init__(self, epoch_bound=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.epoch_bound = epoch_bound

    def attach(self, engine, name):
        if self.epoch_bound:
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def compute(self):
        if self._value is None:
            self._value = self._get_src_value()
        else:
            self._value = (
                self._get_src_value() * self.alpha + (1.0 - self.alpha) * self._value
            )
        return self._value


@dataclass
class HandlerConfig:
    config: "Config"
    run_folder: Path
    run_name: str
    pbar: ProgressBar
    monitoring_metrics: List[str]


def attach_metrics(engine: Engine, run: HandlerConfig) -> None:
    """
    For each metric, we crate an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has it's own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for metric in run.monitoring_metrics:
        partial_func = partial(lambda x, metric_: x[metric_], metric_=metric)
        MyRunningAverage(output_transform=partial_func, alpha=0.95).attach(
            engine, metric
        )


def log_stats(engine: Engine, run: HandlerConfig) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    for name, value in engine.state.metrics.items():
        if name.startswith("t_"):
            log_string += f" | {name}: {value:.4f}"

    run.pbar.log_message(log_string)


def write_metrics(engine: Engine, run: HandlerConfig):
    with open(str(run.run_folder) + "/training_history.log", "a") as logfile:
        fieldnames = sorted(engine.state.metrics.keys())
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)

        if engine.state.iteration == 1:
            writer.writeheader()
        writer.writerow(engine.state.metrics)


def plot_progress(engine: Engine, run: HandlerConfig) -> None:
    args = run.config.cl_args
    hook_funcs = []

    def plot_benchmark_hook(ax):
        benchmark_file = Path(run.run_folder, "benchmark/benchmark_metrics.txt")
        with open(str(benchmark_file), "r") as bfile:
            lines = [i.strip() for i in bfile if i.startswith("VAL MCC")]
            value = float(lines[0].split(": ")[-1])

        benchm_line = ax.axhline(
            y=value, linewidth=0.5, color="gray", linestyle="dashed"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(benchm_line)
        labels.append("LR Benchmark")
        ax.legend(handles, labels)

    if args.benchmark:
        hook_funcs.append(plot_benchmark_hook)

    n_iter_per_epoch = len(run.config.train_loader)

    if check_if_iteration_sample(
        engine.state.iteration, args.sample_interval, n_iter_per_epoch, args.n_epochs
    ):
        metrics_file = Path(run.run_folder, "training_history.log")
        vf.generate_all_plots(metrics_file, hook_funcs=hook_funcs)

        with open(Path(run.run_folder, "model_info.txt"), "w") as mfile:
            mfile.write(str(run.config.model))


def _add_event_handlers(trainer: Engine, run: HandlerConfig):
    """
    This makes sure to add the appropriate event handlers
    if the user wants to keep the output

    TODO: Better docstring.
    """
    args = run.config.cl_args
    checkpoint_handler = ModelCheckpoint(
        Path(run.run_folder, "saved_models"),
        args.run_name,
        create_dir=True,
        n_saved=100,
        save_interval=args.checkpoint_interval,
        save_as_state_dict=True,
    )

    if args.model_task == "cls":
        np.save(
            Path(run.run_folder, "saved_models", "classes.npy"),
            run.config.label_encoder.classes_,
        )

    with open(run.run_folder + "/run_config.json", "w") as config_file:
        config_dict = vars(args)
        json.dump(config_dict, config_file, sort_keys=True, indent=4)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint_handler, to_save={"model": run.config.model}
    )

    # this needs to be attached before plot progress so we have the last row
    # when plotting
    trainer.add_event_handler(Events.ITERATION_COMPLETED, write_metrics, run=run)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, plot_progress, run=run)

    if args.benchmark:
        trainer.add_event_handler(
            Events.STARTED, benchmark, config=run.config, run_folder=run.run_folder
        )
    return trainer


def configure_trainer(trainer: Engine, config: "Config") -> Engine:
    """
    NOTE:
        **Important** the evaluate handler must be attached before the
        ``save_progress`` function, as it manually adds validation metrics
        to the engine state. I.e. we need to make sure they have been
        calculated before calling ``save_progress`` during training.

    TODO:
        Check if there is a better way to address the above, e.g. reordering
        the handlers in this func in the end?
    """
    args = config.cl_args
    run_folder = "runs/" + args.run_name
    pbar = ProgressBar()
    run_name = args.run_name
    monitoring_metrics = ["t_loss"] + get_train_metrics(model_task=args.model_task)

    run = HandlerConfig(config, run_folder, run_name, pbar, monitoring_metrics)

    for handler in evaluation_handler, activation_analysis_handler:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, handler, run=run)

    attach_metrics(trainer, run=run)
    pbar.attach(trainer, metric_names=run.monitoring_metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_stats, run=run)

    if run.run_name:
        trainer = _add_event_handlers(trainer, run)

    return trainer
