import atexit
import json
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Tuple, TYPE_CHECKING, Dict

import pandas as pd
from aislib.misc_utils import get_logger
from torch.utils.tensorboard import SummaryWriter
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage

from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.train_utils.activation_analysis import (
    activation_analysis_handler,
)
from human_origins_supervised.train_utils import H_PARAMS
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
    add_metrics_to_writer,
)
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train import Config
    from human_origins_supervised.train_utils.metric_funcs import al_step_metric_dict

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


def configure_trainer(trainer: Engine, config: "Config") -> Engine:
    """
    NOTE:
        **Important** the evaluate handler must be attached before the
        ``save_progress`` function, as it manually adds validation metrics
        to the trainer state. I.e. we need to make sure they have been
        calculated before calling ``save_progress`` during training.

    TODO:
        Check if there is a better way to address the above, e.g. reordering
        the handlers in this func in the end?
    """
    cl_args = config.cl_args
    run_folder = Path("runs/", cl_args.run_name)
    pbar = ProgressBar()
    run_name = cl_args.run_name

    monitoring_metrics = _get_monitoring_metrics(target_columns=config.target_columns)

    handler_config = HandlerConfig(
        config=config,
        run_folder=run_folder,
        run_name=run_name,
        pbar=pbar,
        monitoring_metrics=monitoring_metrics,
    )

    for handler in evaluation_handler, activation_analysis_handler:
        trainer.add_event_handler(
            event_name=Events.ITERATION_COMPLETED(every=cl_args.sample_interval),
            handler=handler,
            handler_config=handler_config,
        )

        trainer.add_event_handler(
            event_name=Events.COMPLETED, handler=handler, handler_config=handler_config
        )

    if cl_args.lr_schedule is not None:
        lr_scheduler = set_up_scheduler(handler_config=handler_config)
        attach_lr_scheduler(engine=trainer, lr_scheduler=lr_scheduler, config=config)

    _attach_metrics(engine=trainer, monitoring_metrics=monitoring_metrics)
    pbar.attach(engine=trainer, metric_names=["t_loss-average"])

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=_log_stats_to_pbar,
        handler_config=handler_config,
    )

    if handler_config.run_name:
        trainer = _attach_run_event_handlers(
            trainer=trainer, handler_config=handler_config
        )

    return trainer


def _get_monitoring_metrics(target_columns) -> List[Tuple[str, str]]:
    """
    The spec for the tuple here follows the metric dict spec, i.e. the tuple is:
    (column_name, metric).
    """
    target_columns_gen = get_target_columns_generator(target_columns)

    loss_average_metrics = tuple(["t_average", "t_loss-average"])
    perf_average_metrics = tuple(["t_average", "t_perf-average"])
    monitoring_metrics = [loss_average_metrics, perf_average_metrics]

    for column_type, column_name in target_columns_gen:

        cur_metrics = get_train_metrics(
            column_type=column_type, prefix=f"t_{column_name}"
        )

        for metric in cur_metrics:
            cur_tuple = tuple([column_name, metric])
            monitoring_metrics.append(cur_tuple)

    return monitoring_metrics


def _attach_metrics(engine: Engine, monitoring_metrics: List[Tuple[str, str]]) -> None:
    """
    For each metric, we create an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has it's own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for column_name, metric_name in monitoring_metrics:

        def output_transform(
            metric_dict_from_step: "al_step_metric_dict",
            column_name_key: str,
            metric_name_key: str,
        ) -> float:
            return metric_dict_from_step[column_name_key][metric_name_key]

        partial_func = partial(
            output_transform, column_name_key=column_name, metric_name_key=metric_name
        )

        RunningAverage(
            output_transform=partial_func, alpha=0.98, epoch_bound=False
        ).attach(engine, name=metric_name)


def _log_stats_to_pbar(engine: Engine, handler_config: HandlerConfig) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    key = "t_loss-average"
    value = engine.state.metrics[key]
    log_string += f" | {key}: {value:.4g}"

    handler_config.pbar.log_message(log_string)


def _attach_run_event_handlers(trainer: Engine, handler_config: HandlerConfig):
    cl_args = handler_config.config.cl_args

    checkpoint_handler = ModelCheckpoint(
        Path(handler_config.run_folder, "saved_models"),
        cl_args.run_name,
        create_dir=True,
        n_saved=100,
        save_as_state_dict=True,
    )

    _save_config(run_folder=handler_config.run_folder, cl_args=cl_args)

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=cl_args.checkpoint_interval),
        handler=checkpoint_handler,
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

    if cl_args.custom_lib:
        custom_handlers = _get_custom_handlers(handler_config)
        trainer = _attach_custom_handlers(trainer, handler_config, custom_handlers)

    tensorboard_hparam_func = partial(
        add_hparams_to_tensorboard,
        h_params=H_PARAMS,
        config=handler_config.config,
        writer=handler_config.config.writer,
    )
    atexit.register(tensorboard_hparam_func)

    return trainer


def _save_config(run_folder: Path, cl_args: Namespace):
    with open(str(run_folder / "cl_args.json"), "w") as config_file:
        config_dict = vars(cl_args)
        json.dump(config_dict, config_file, sort_keys=True, indent=4)


def _write_training_metrics_handler(engine: Engine, handler_config: HandlerConfig):
    """
    Note that trainer.state.metrics contains the running averages we are interested in.

    The main "problem" here is that we lose the structure of the metrics dict that
    we get from `train_utils.metric_funcs.calculate_batch_metrics`, so we have to
    filter all metrics for a given target column specifically, from the 1d array
    trainer.state.metrics gives us.
    """
    args = handler_config.config.cl_args
    writer = handler_config.config.writer
    iteration = engine.state.iteration
    target_columns = handler_config.config.target_columns

    engine_metrics_dict = engine.state.metrics

    run_folder = get_run_folder(run_name=args.run_name)

    is_first_iteration = True if iteration == 1 else False

    metrics_files = get_metrics_files(
        target_columns=target_columns, run_folder=run_folder, target_prefix="t_"
    )

    if is_first_iteration:
        ensure_metrics_paths_exists(metrics_files)

    for metrics_name, metrics_history_file in metrics_files.items():
        cur_metric_dict = filter_items_from_engine_metrics_dict(
            engine_metrics_dict=engine_metrics_dict, metrics_substring=metrics_name
        )

        add_metrics_to_writer(
            name=f"train/{metrics_name}",
            metric_dict=cur_metric_dict,
            iteration=iteration,
            writer=writer,
        )

        append_metrics_to_file(
            filepath=metrics_history_file,
            metrics=cur_metric_dict,
            iteration=iteration,
            write_header=is_first_iteration,
        )


def _plot_progress_handler(engine: Engine, handler_config: HandlerConfig) -> None:
    args = handler_config.config.cl_args
    hook_funcs = []

    run_folder = get_run_folder(args.run_name)

    for results_dir in (run_folder / "results").iterdir():
        target_column = results_dir.name

        train_history_df, valid_history_df = _get_metrics_dataframes(
            results_dir=results_dir, target_string=target_column
        )

        vf.generate_all_training_curves(
            training_history_df=train_history_df,
            valid_history_df=valid_history_df,
            output_folder=results_dir,
            hook_funcs=hook_funcs,
        )

    train_avg_history_df, valid_avg_history_df = _get_metrics_dataframes(
        results_dir=run_folder, target_string="average"
    )

    vf.generate_all_training_curves(
        training_history_df=train_avg_history_df,
        valid_history_df=valid_avg_history_df,
        output_folder=run_folder,
        hook_funcs=hook_funcs,
    )

    with open(Path(handler_config.run_folder, "model_info.txt"), "w") as mfile:
        mfile.write(str(handler_config.config.model))


def _get_metrics_dataframes(
    results_dir: Path, target_string: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_history_path = read_metrics_history_file(
        results_dir / f"t_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"v_{target_string}_history.log"
    )

    return train_history_path, valid_history_path


def add_hparams_to_tensorboard(
    h_params: List[str], config: "Config", writer: SummaryWriter
) -> None:

    logger.debug(
        "Exiting and logging best hyperparameters for best average loss "
        "to tensorboard."
    )

    c = config
    run_folder = get_run_folder(c.cl_args.run_name)

    metrics_files = get_metrics_files(
        target_columns=c.target_columns, run_folder=run_folder, target_prefix="v_"
    )

    try:
        best_overall_performance = _get_best_average_performance(
            val_metrics_files=metrics_files, target_columns=c.target_columns
        )
    except FileNotFoundError:
        logger.debug(
            "Could not find target column performance files at exit."
            "Tensorboard hyper parameters not logged."
        )
        return

    average_loss_file = metrics_files["v_loss-average"]
    try:
        average_loss_df = pd.read_csv(average_loss_file)
    except FileNotFoundError:
        logger.debug(
            "Could not find %s at exit. " "Tensorboard hyper parameters not logged.",
            average_loss_file,
        )
        return

    min_loss = average_loss_df["v_loss-average"].min()

    h_param_dict = {
        param_name: getattr(c.cl_args, param_name) for param_name in h_params
    }

    writer.add_hparams(
        h_param_dict,
        {
            "v_loss-overall_min": min_loss,
            "best_overall_performance": best_overall_performance,
        },
    )


def _get_best_average_performance(val_metrics_files: Dict[str, Path], target_columns):
    df_performances = _get_overall_performance(
        val_metrics_files=val_metrics_files, target_columns=target_columns
    )
    best_performance = df_performances["Performance_Average"].max()

    return best_performance


def _get_overall_performance(
    val_metrics_files: Dict[str, Path], target_columns
) -> pd.DataFrame:
    """
    With continuous columns, we use the distance the MSE is from 1 as "performance".
    """
    target_columns_gen = get_target_columns_generator(target_columns)

    df_performances = pd.DataFrame()
    for column_type, column_name in target_columns_gen:
        cur_metric_df = pd.read_csv(val_metrics_files[column_name])

        if column_type == "con":
            df_performances[column_name] = 1 - cur_metric_df[f"v_{column_name}_loss"]

        elif column_type == "cat":
            df_performances[column_name] = cur_metric_df[f"v_{column_name}_mcc"]

        else:
            raise ValueError()

    df_performances["Performance_Average"] = df_performances.mean(axis=1)

    return df_performances


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
