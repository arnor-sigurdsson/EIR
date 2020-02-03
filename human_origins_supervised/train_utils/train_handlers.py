import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Tuple, TYPE_CHECKING, Dict

from aislib.misc_utils import get_logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage

from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.train_utils.activation_analysis import (
    activation_analysis_handler,
)
from human_origins_supervised.train_utils.benchmark import benchmark
from human_origins_supervised.train_utils.evaluation import evaluation_handler
from human_origins_supervised.train_utils.lr_scheduling import (
    set_up_scheduler,
    attach_lr_scheduler,
)
from human_origins_supervised.train_utils.metric_funcs import get_train_metrics
from human_origins_supervised.train_utils.utils import (
    get_custom_module_submodule,
    append_metrics_to_file,
    read_metrics_history_file,
    get_run_folder,
    get_metrics_files,
    ensure_metrics_paths_exists,
    filter_items_from_engine_metrics_dict,
)
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

# Aliases
al_get_custom_handles_return_value = Union[Tuple[Callable, ...], Tuple[None]]
al_get_custom_handlers = Callable[["HandlerConfig"], al_get_custom_handles_return_value]


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class HandlerConfig:
    config: "Config"
    run_folder: Path
    run_name: str
    pbar: ProgressBar
    monitoring_metrics: List[Tuple[str, str]]


def attach_metrics(engine: Engine, handler_config: HandlerConfig) -> None:
    """
    For each metric, we crate an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has it's own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for column_name, metric_name in handler_config.monitoring_metrics:

        def output_transform(
            metric_dict_from_step: Dict[str, Dict[str, float]],
            main_key: str,
            secondary_key: str,
        ) -> float:
            return metric_dict_from_step[main_key][secondary_key]

        partial_func = partial(
            output_transform, main_key=column_name, secondary_key=metric_name
        )

        RunningAverage(
            output_transform=partial_func, alpha=0.98, epoch_bound=False
        ).attach(engine, name=metric_name)


def log_stats(engine: Engine, handler_config: HandlerConfig) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    for name, value in engine.state.metrics.items():
        if name.startswith("t_"):
            log_string += f" | {name}: {value:.4g}"

    handler_config.pbar.log_message(log_string)


def _write_training_metrics_handler(engine: Engine, handler_config: HandlerConfig):
    """
    Note that engine.state.metrics contains the running averages we are interested in.

    The main "problem" here is that we lose the structure of the metrics dict that
    we get from `train_utils.metric_funcs.calculate_batch_metrics`, so we have to
    filter all metrics for a given target column specifically, from the 1d array
    engine.state.metrics gives us.
    """
    args = handler_config.config.cl_args
    iteration = engine.state.iteration
    target_columns = handler_config.config.target_columns

    engine_metrics_dict = engine.state.metrics

    run_folder = get_run_folder(run_name=args.run_name)

    is_first_iteration = True if iteration == 1 else False

    metrics_files = get_metrics_files(
        target_columns=target_columns, run_folder=run_folder, target_prefix="t"
    )

    if is_first_iteration:
        ensure_metrics_paths_exists(metrics_files)

    for metrics_name, metrics_history_file in metrics_files.items():
        cur_metric_dict = filter_items_from_engine_metrics_dict(
            engine_metrics_dict=engine_metrics_dict, target=metrics_name
        )

        append_metrics_to_file(
            filepath=metrics_history_file,
            metrics=cur_metric_dict,
            iteration=iteration,
            write_header=is_first_iteration,
        )


def _plot_benchmark_hook(ax, run_folder, target: str):

    benchmark_file = Path(run_folder, "benchmark/benchmark_metrics.txt")
    with open(str(benchmark_file), "r") as bfile:
        lines = [i.strip() for i in bfile if i.startswith(target)]

        # If we did not run benchmark for this metric, don't plot anything
        if not lines:
            return

        value = float(lines[0].split(": ")[-1])

    benchm_line = ax.axhline(y=value, linewidth=0.5, color="gray", linestyle="dashed")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(benchm_line)
    labels.append("LR Benchmark")
    ax.legend(handles, labels)


def _plot_progress_handler(engine: Engine, handler_config: HandlerConfig) -> None:
    args = handler_config.config.cl_args
    hook_funcs = []

    if args.benchmark:
        hook_funcs.append(
            partial(_plot_benchmark_hook, run_folder=handler_config.run_folder)
        )

    all_results_folder = get_run_folder(args.run_name) / "results"

    for results_dir in all_results_folder.iterdir():
        target_column = results_dir.name

        train_history_path = read_metrics_history_file(
            results_dir / f"t_{target_column}_history.log"
        )
        valid_history_path = read_metrics_history_file(
            results_dir / f"v_{target_column}_history.log"
        )

        vf.generate_all_plots(
            training_history=train_history_path,
            valid_history=valid_history_path,
            output_folder=results_dir,
            hook_funcs=hook_funcs,
        )

    # TODO: Refactor
    run_folder = get_run_folder(args.run_name)
    average_training_history = read_metrics_history_file(
        run_folder / "t_average-loss_history.log"
    )
    average_eval_history = read_metrics_history_file(
        run_folder / "v_average-loss_history.log"
    )
    vf.generate_all_plots(
        training_history=average_training_history,
        valid_history=average_eval_history,
        output_folder=run_folder,
        hook_funcs=hook_funcs,
    )

    with open(Path(handler_config.run_folder, "model_info.txt"), "w") as mfile:
        mfile.write(str(handler_config.config.model))


def _get_custom_handlers(handler_config: "HandlerConfig"):
    custom_lib = handler_config.config.cl_args.custom_lib

    custom_handlers_module = get_custom_module_submodule(custom_lib, "custom_handlers")

    if not custom_handlers_module:
        return None

    if not hasattr(custom_handlers_module, "get_custom_handlers"):
        raise ImportError(
            f"'get_custom_handlers' function must be defined in "
            f"{custom_handlers_module} for custom handler attachment."
        )

    custom_handlers_getter = custom_handlers_module.get_custom_handlers
    custom_handlers = custom_handlers_getter(handler_config)

    return custom_handlers


def _attach_custom_handlers(trainer: Engine, handler_config, custom_handlers):
    if not custom_handlers:
        return trainer

    for custom_handler_attacher in custom_handlers:
        trainer = custom_handler_attacher(trainer, handler_config)

    return trainer


def _attach_run_event_handlers(trainer: Engine, handler_config: HandlerConfig):
    """
    This makes sure to add the appropriate event handlers
    if the user wants to keep the output

    TODO: Better docstring.
    TODO: Better name for `run`
    """
    cl_args = handler_config.config.cl_args
    checkpoint_handler = ModelCheckpoint(
        Path(handler_config.run_folder, "saved_models"),
        cl_args.run_name,
        create_dir=True,
        n_saved=100,
        save_as_state_dict=True,
    )

    with open(str(handler_config.run_folder / "cl_args.json"), "w") as config_file:
        config_dict = vars(cl_args)
        json.dump(config_dict, config_file, sort_keys=True, indent=4)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=cl_args.checkpoint_interval),
        checkpoint_handler,
        to_save={"model": handler_config.config.model},
    )

    # *gotcha*: write_metrics needs to be attached before plot progress so we have the
    # last row when plotting
    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=_write_training_metrics_handler,
        handler_config=handler_config,
    )
    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
        handler=_plot_progress_handler,
        handler_config=handler_config,
    )

    if cl_args.benchmark:
        trainer.add_event_handler(
            Events.STARTED,
            benchmark,
            config=handler_config.config,
            run_folder=handler_config.run_folder,
        )

    if cl_args.custom_lib:
        custom_handlers = _get_custom_handlers(handler_config)
        trainer = _attach_custom_handlers(trainer, handler_config, custom_handlers)
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
    cl_args = config.cl_args
    run_folder = Path("runs/", cl_args.run_name)
    pbar = ProgressBar()
    run_name = cl_args.run_name

    monitoring_metrics = get_monitoring_metrics(config.target_columns)

    handler_config = HandlerConfig(
        config, run_folder, run_name, pbar, monitoring_metrics
    )

    for handler in evaluation_handler, activation_analysis_handler:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
            handler,
            handler_config=handler_config,
        )

        trainer.add_event_handler(
            Events.COMPLETED, handler, handler_config=handler_config
        )

    if cl_args.lr_schedule is not None:
        lr_scheduler = set_up_scheduler(handler_config)
        attach_lr_scheduler(trainer, lr_scheduler, config)

    attach_metrics(trainer, handler_config=handler_config)
    pbar.attach(trainer, metric_names=["t_loss-average"])

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, log_stats, handler_config=handler_config
    )

    if handler_config.run_name:
        trainer = _attach_run_event_handlers(trainer, handler_config)

    return trainer


def get_monitoring_metrics(target_columns) -> List[Tuple[str, str]]:
    """
    TODO: Refactor. Possibly have loss in get_train_metrics.
    """

    target_columns_gen = get_target_columns_generator(target_columns)

    loss_average_metrics = tuple(["t_loss-average"] * 2)
    monitoring_metrics = [loss_average_metrics]
    for column_type, column_name in target_columns_gen:

        cur_metrics = get_train_metrics(
            column_type=column_type, prefix=f"t_{column_name}"
        )

        for metric in cur_metrics:
            cur_tuple = tuple([column_name, metric])
            monitoring_metrics.append(cur_tuple)

    return monitoring_metrics
