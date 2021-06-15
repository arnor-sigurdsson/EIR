import atexit
import json
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Callable, Union, Tuple, TYPE_CHECKING, Dict, overload

import pandas as pd
from aislib.misc_utils import get_logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine, events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import RunningAverage
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from eir.data_load.data_utils import get_target_columns_generator
from eir.data_load.label_setup import al_target_columns
from eir.interpretation.interpretation import activation_analysis_handler
from eir.train_utils import H_PARAMS
from eir.train_utils.evaluation import validation_handler
from eir.train_utils.lr_scheduling import (
    set_up_lr_scheduler,
    attach_lr_scheduler,
)
from eir.train_utils.metrics import (
    get_metrics_dataframes,
    persist_metrics,
    get_metrics_files,
    al_metric_record_dict,
    MetricRecord,
    read_metrics_history_file,
    get_average_history_filepath,
)
from eir.train_utils.utils import get_run_folder, validate_handler_dependencies
from eir.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from eir.train import Config
    from eir.train_utils.metrics import al_step_metric_dict

# Aliases
al_handler_and_event = Tuple[Callable[[Engine, "HandlerConfig"], None], Events]
al_sample_interval_handlers = Tuple[al_handler_and_event, ...]


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class HandlerConfig:
    config: "Config"
    run_folder: Path
    run_name: str
    monitoring_metrics: List[Tuple[str, str]]


def configure_trainer(trainer: Engine, config: "Config") -> Engine:

    ca = config.cl_args
    run_folder = get_run_folder(run_name=ca.run_name)

    monitoring_metrics = _get_monitoring_metrics(
        target_columns=config.target_columns, metric_record_dict=config.metrics
    )

    handler_config = HandlerConfig(
        config=config,
        run_folder=run_folder,
        run_name=ca.run_name,
        monitoring_metrics=monitoring_metrics,
    )

    _attach_running_average_metrics(
        engine=trainer, monitoring_metrics=monitoring_metrics
    )

    if not ca.no_pbar:
        pbar = ProgressBar()
        pbar.attach(engine=trainer, metric_names=["loss-average"])
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=_log_stats_to_pbar,
            pbar=pbar,
        )

    trainer = _attach_sample_interval_handlers(
        trainer=trainer, handler_config=handler_config
    )

    if ca.early_stopping_patience:
        trainer = _attach_early_stopping_handler(
            trainer=trainer, handler_config=handler_config
        )

    # TODO: Implement warmup for same LR scheduling
    if ca.lr_schedule != "same":
        lr_scheduler = set_up_lr_scheduler(handler_config=handler_config)
        attach_lr_scheduler(engine=trainer, lr_scheduler=lr_scheduler, config=config)
    elif ca.lr_schedule == "same" and ca.warmup_steps:
        raise NotImplementedError("Warmup not yet implemented for 'same' LR schedule.")

    if handler_config.run_name:
        trainer = _attach_run_event_handlers(
            trainer=trainer, handler_config=handler_config
        )

    return trainer


def _attach_sample_interval_handlers(
    trainer: Engine,
    handler_config: "HandlerConfig",
) -> Engine:

    config = handler_config.config
    cl_args = config.cl_args

    validation_handler_and_event = _get_validation_handler_and_event(
        sample_interval_base=cl_args.sample_interval,
        iter_per_epoch=len(config.train_loader),
        n_epochs=cl_args.n_epochs,
        early_stopping_patience=cl_args.early_stopping_patience,
    )
    all_handler_events = [validation_handler_and_event]

    if cl_args.get_acts:
        activation_handler_and_event = _get_activation_handler_and_event(
            iter_per_epoch=len(config.train_loader),
            n_epochs=cl_args.n_epochs,
            sample_interval_base=cl_args.sample_interval,
            act_every_sample_factor=cl_args.act_every_sample_factor,
            early_stopping_patience=cl_args.early_stopping_patience,
        )
        all_handler_events.append(activation_handler_and_event)

    for handler, event in all_handler_events:

        trainer.add_event_handler(
            event_name=event,
            handler=handler,
            handler_config=handler_config,
        )

    return trainer


def _get_validation_handler_and_event(
    sample_interval_base: int,
    iter_per_epoch: int,
    n_epochs: int,
    early_stopping_patience: int,
) -> al_handler_and_event:

    validation_handler_callable = validation_handler
    validation_event = Events.ITERATION_COMPLETED(every=sample_interval_base)

    do_run_when_training_complete = _do_run_completed_handler(
        iter_per_epoch=iter_per_epoch,
        n_epochs=n_epochs,
        sample_interval=sample_interval_base,
    )

    if do_run_when_training_complete and not early_stopping_patience:
        validation_event = validation_event | Events.COMPLETED

    return validation_handler_callable, validation_event


def _get_activation_handler_and_event(
    iter_per_epoch: int,
    n_epochs: int,
    sample_interval_base: int,
    act_every_sample_factor: int,
    early_stopping_patience,
) -> al_handler_and_event:

    activation_handler_callable = activation_analysis_handler

    if act_every_sample_factor == 0:
        activation_event = Events.COMPLETED
        logger.debug("Activations will be computed at run end.")

        return activation_handler_callable, activation_event

    activation_handler_interval = sample_interval_base * act_every_sample_factor
    activation_event = Events.ITERATION_COMPLETED(every=activation_handler_interval)

    do_run_when_training_complete = _do_run_completed_handler(
        iter_per_epoch=iter_per_epoch,
        n_epochs=n_epochs,
        sample_interval=activation_handler_interval,
    )

    if do_run_when_training_complete and not early_stopping_patience:
        activation_event = activation_event | Events.COMPLETED

    logger.debug(
        "Activations will be computed every %d iterations.",
        activation_handler_interval,
    )

    return activation_handler_callable, activation_event


def _do_run_completed_handler(iter_per_epoch: int, n_epochs: int, sample_interval: int):
    """
    We need this function to avoid running handlers twice in cases where the total
    number of iterations has a remainder of 0 w.r.t. the sample interval.
    """
    if (iter_per_epoch * n_epochs) % sample_interval != 0:
        return True

    return False


def _get_monitoring_metrics(
    target_columns: al_target_columns, metric_record_dict: al_metric_record_dict
) -> List[Tuple[str, str]]:
    """
    The spec for the tuple here follows the metric dict spec, i.e. the tuple is:
    (column_name, metric).
    """
    target_columns_gen = get_target_columns_generator(target_columns=target_columns)

    loss_average_metrics = tuple(["average", "loss-average"])
    perf_average_metrics = tuple(["average", "perf-average"])
    monitoring_metrics = [loss_average_metrics, perf_average_metrics]

    def _parse_target_metrics(metric_name: str, column_name_: str) -> str:
        return f"{column_name_}_{metric_name}"

    for column_type, column_name in target_columns_gen:

        cur_metric_records: Tuple[MetricRecord, ...] = metric_record_dict[column_type]

        for metric in cur_metric_records:
            if not metric.only_val:

                parsed_metric = _parse_target_metrics(
                    metric_name=metric.name, column_name_=column_name
                )
                cur_tuple = tuple([column_name, parsed_metric])
                monitoring_metrics.append(cur_tuple)

        # manually add loss record as it's not in metric records, but from criterions
        loss_name = _parse_target_metrics(metric_name="loss", column_name_=column_name)
        monitoring_metrics.append(tuple([column_name, loss_name]))

    return monitoring_metrics


@validate_handler_dependencies(
    [validation_handler],
)
def _attach_early_stopping_handler(trainer: Engine, handler_config: "HandlerConfig"):
    cl_args = handler_config.config.cl_args

    early_stopping_handler = _get_early_stopping_handler(
        trainer=trainer,
        handler_config=handler_config,
        patience_steps=cl_args.early_stopping_patience,
    )

    early_stopping_event_kwargs = _get_early_stopping_event_kwargs(
        early_stopping_iteration_buffer=cl_args.early_stopping_buffer,
        sample_interval=cl_args.sample_interval,
    )

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(
            **early_stopping_event_kwargs,
        ),
        handler=early_stopping_handler,
    )

    return trainer


def _get_early_stopping_handler(
    trainer: Engine, handler_config: HandlerConfig, patience_steps: int
):

    scoring_function = _get_latest_validation_value_score_function(
        run_folder=handler_config.run_folder, column="perf-average"
    )

    logger.info(
        "Setting early stopping patience to %d validation steps.", patience_steps
    )

    handler = EarlyStopping(
        patience=patience_steps, score_function=scoring_function, trainer=trainer
    )
    handler.logger = logger

    return handler


@overload
def _get_early_stopping_event_kwargs(
    early_stopping_iteration_buffer: None, sample_interval: int
) -> Dict[str, int]:
    ...


@overload
def _get_early_stopping_event_kwargs(
    early_stopping_iteration_buffer: int, sample_interval: int
) -> Dict[str, Callable[[Engine, int], bool]]:
    ...


def _get_early_stopping_event_kwargs(early_stopping_iteration_buffer, sample_interval):

    if early_stopping_iteration_buffer is None:
        return {"every": sample_interval}

    logger.info(
        "Early stopping checks will be activated after %d iterations.",
        early_stopping_iteration_buffer,
    )
    has_checked = False

    def _early_stopping_event_filter(engine: Engine, event: int) -> bool:
        iteration = event

        if iteration < early_stopping_iteration_buffer:
            return False

        nonlocal has_checked
        if not has_checked:
            logger.debug(
                "%d iterations done, early stopping checks activated from now on.",
                iteration,
            )
            has_checked = True

        if iteration % sample_interval == 0:
            return True

        return False

    return {"event_filter": _early_stopping_event_filter}


def _get_latest_validation_value_score_function(run_folder: Path, column: str):
    eval_history_fpath = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="validation_"
    )

    def scoring_function(engine):
        eval_df = read_metrics_history_file(eval_history_fpath)
        latest_val_loss = eval_df[column].iloc[-1]
        return latest_val_loss

    return scoring_function


def _attach_running_average_metrics(
    engine: Engine, monitoring_metrics: List[Tuple[str, str]]
) -> None:
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


def _log_stats_to_pbar(engine: Engine, pbar: ProgressBar) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    key = "loss-average"
    value = engine.state.metrics[key]
    log_string += f" | {key}: {value:.4g}"

    pbar.log_message(log_string)


def _attach_run_event_handlers(trainer: Engine, handler_config: HandlerConfig):
    c = handler_config.config
    cl_args = handler_config.config.cl_args

    _save_config(run_folder=handler_config.run_folder, cl_args=cl_args)

    if cl_args.checkpoint_interval is not None:
        trainer = _add_checkpoint_handler_wrapper(
            trainer=trainer,
            run_folder=handler_config.run_folder,
            run_name=Path(cl_args.run_name),
            n_to_save=cl_args.n_saved_models,
            checkpoint_interval=cl_args.checkpoint_interval,
            sample_interval=cl_args.sample_interval,
            model=c.model,
        )

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=_write_training_metrics_handler,
        handler_config=handler_config,
    )

    for plot_event in _get_plot_events(sample_interval=cl_args.sample_interval):

        if plot_event == Events.COMPLETED and not _do_run_completed_handler(
            iter_per_epoch=len(c.train_loader),
            n_epochs=cl_args.n_epochs,
            sample_interval=cl_args.sample_interval,
        ):
            continue

        trainer = _attach_plot_progress_handler(
            trainer=trainer, plot_event=plot_event, handler_config=handler_config
        )

    if c.hooks.custom_handler_attachers is not None:
        custom_handlers = _get_custom_handlers(handler_config=handler_config)
        trainer = _attach_custom_handlers(
            trainer=trainer,
            handler_config=handler_config,
            custom_handlers=custom_handlers,
        )

    log_tb_hparams_on_exit_func = partial(
        add_hparams_to_tensorboard,
        h_params=H_PARAMS,
        config=handler_config.config,
        writer=handler_config.config.writer,
    )
    atexit.register(log_tb_hparams_on_exit_func)

    return trainer


def _save_config(run_folder: Path, cl_args: Namespace):
    with open(str(run_folder / "cl_args.json"), "w") as config_file:
        config_dict = vars(cl_args)
        json.dump(config_dict, config_file, sort_keys=True, indent=4)


def _add_checkpoint_handler_wrapper(
    trainer: Engine,
    run_folder: Path,
    run_name: Path,
    n_to_save: Union[int, None],
    checkpoint_interval: int,
    sample_interval: int,
    model: nn.Module,
) -> Engine:

    checkpoint_score_function, score_name = None, None
    if n_to_save is not None:
        logger.debug(
            "Setting up scoring function for checkpoint interval, "
            "keeping top %d models.",
            n_to_save,
        )

        if checkpoint_interval % sample_interval != 0:
            raise ValueError(
                f"The checkpoint interval ({checkpoint_interval}) is not "
                f"a multiplication of the sample interval "
                f"({sample_interval}) and top n saved models are being "
                f"requested. This means the scoring functions would give"
                f"a wrong result for the checkpoint (as a validation step"
                f"is not performed at the checkpoint interval)."
            )

        score_name = "perf-average"
        checkpoint_score_function = _get_latest_validation_value_score_function(
            run_folder=run_folder, column=score_name
        )

    checkpoint_handler = _get_checkpoint_handler(
        run_folder=run_folder,
        run_name=run_name,
        n_to_save=n_to_save,
        score_function=checkpoint_score_function,
        score_name=score_name,
    )

    trainer = _attach_checkpoint_handler(
        trainer=trainer,
        checkpoint_handler=checkpoint_handler,
        checkpoint_interval=checkpoint_interval,
        model=model,
    )

    return trainer


def _get_checkpoint_handler(
    run_folder: Path,
    run_name: Path,
    n_to_save: int,
    score_function: Callable = None,
    score_name: str = None,
) -> ModelCheckpoint:
    def _default_global_step_transform(engine: Engine, event_name: str) -> int:
        return engine.state.iteration

    checkpoint_handler = ModelCheckpoint(
        dirname=Path(run_folder, "saved_models"),
        filename_prefix=run_name.name,
        create_dir=True,
        score_name=score_name,
        n_saved=n_to_save,
        score_function=score_function,
        global_step_transform=_default_global_step_transform,
    )

    return checkpoint_handler


@validate_handler_dependencies(
    [validation_handler],
)
def _attach_checkpoint_handler(
    trainer: Engine,
    checkpoint_handler: ModelCheckpoint,
    checkpoint_interval: int,
    model: nn.Module,
) -> Engine:

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=checkpoint_interval),
        handler=checkpoint_handler,
        to_save={"model": model},
    )

    return trainer


def _write_training_metrics_handler(engine: Engine, handler_config: HandlerConfig):
    """
    Note that trainer.state.metrics contains the *running averages* we are interested
    in.

    The main "problem" here is that we lose the structure of the metrics dict that
    we get from `train_utils.metrics.calculate_batch_metrics`, so we have to
    filter all metrics for a given target column specifically, from the 1d array
    trainer.state.metrics gives us.
    """

    iteration = engine.state.iteration

    is_first_iteration = True if iteration == 1 else False

    running_average_metrics = _unflatten_engine_metrics_dict(
        step_base=engine.state.output, engine_metrics_dict=engine.state.metrics
    )
    persist_metrics(
        handler_config=handler_config,
        metrics_dict=running_average_metrics,
        iteration=iteration,
        write_header=is_first_iteration,
        prefixes={"metrics": "train_", "writer": "train"},
    )


def _unflatten_engine_metrics_dict(
    step_base: "al_step_metric_dict", engine_metrics_dict: Dict[str, float]
) -> "al_step_metric_dict":
    """
    We need this to streamline the 1D dictionary that comes from engine.state.metrics.
    """
    unflattened_dict = {}
    for column_name, column_metric_dict in step_base.items():
        unflattened_dict[column_name] = {}

        for column_metric_name in column_metric_dict.keys():
            eng_run_avg_value = engine_metrics_dict[column_metric_name]
            unflattened_dict[column_name][column_metric_name] = eng_run_avg_value

    return unflattened_dict


def _get_plot_events(
    sample_interval: int,
) -> Tuple[events.CallableEventWithFilter, events.CallableEventWithFilter]:
    plot_events = (
        Events.ITERATION_COMPLETED(every=sample_interval),
        Events.COMPLETED,
    )
    return plot_events


@validate_handler_dependencies([_write_training_metrics_handler])
def _attach_plot_progress_handler(
    trainer: Engine,
    plot_event: events.CallableEventWithFilter,
    handler_config: HandlerConfig,
):
    trainer.add_event_handler(
        event_name=plot_event,
        handler=_plot_progress_handler,
        handler_config=handler_config,
    )

    return trainer


def _plot_progress_handler(engine: Engine, handler_config: HandlerConfig) -> None:
    cl_args = handler_config.config.cl_args

    # if no val data is available yet
    if engine.state.iteration < cl_args.sample_interval:
        return

    run_folder = get_run_folder(cl_args.run_name)

    for results_dir in (run_folder / "results").iterdir():
        target_column = results_dir.name

        train_history_df, valid_history_df = get_metrics_dataframes(
            results_dir=results_dir, target_string=target_column
        )

        vf.generate_all_training_curves(
            training_history_df=train_history_df,
            valid_history_df=valid_history_df,
            output_folder=results_dir,
            title_extra=target_column,
            plot_skip_steps=cl_args.plot_skip_steps,
        )

    train_avg_history_df, valid_avg_history_df = get_metrics_dataframes(
        results_dir=run_folder, target_string="average"
    )

    vf.generate_all_training_curves(
        training_history_df=train_avg_history_df,
        valid_history_df=valid_avg_history_df,
        output_folder=run_folder,
        title_extra="Multi Task Average",
        plot_skip_steps=cl_args.plot_skip_steps,
    )

    with open(Path(handler_config.run_folder, "model_info.txt"), "w") as mfile:
        mfile.write(str(handler_config.config.model))


def _get_custom_handlers(handler_config: "HandlerConfig"):

    custom_handlers = handler_config.config.hooks.custom_handler_attachers

    return custom_handlers


def _attach_custom_handlers(
    trainer: Engine, handler_config: "HandlerConfig", custom_handlers
):
    if not custom_handlers:
        return trainer

    for custom_handler_attacher in custom_handlers:
        trainer = custom_handler_attacher(trainer, handler_config)

    return trainer


def add_hparams_to_tensorboard(
    h_params: List[str], config: "Config", writer: SummaryWriter
) -> None:

    logger.debug(
        "Exiting and logging best hyperparameters for best average loss "
        "to tensorboard."
    )

    c = config
    run_folder = get_run_folder(run_name=c.cl_args.run_name)

    metrics_files = get_metrics_files(
        target_columns=c.target_columns,
        run_folder=run_folder,
        train_or_val_target_prefix="validation_",
    )

    try:
        average_loss_file = metrics_files["average"]
        average_loss_df = pd.read_csv(average_loss_file)

    except FileNotFoundError as e:
        logger.debug(
            "Could not find %s at exit. Tensorboard hyper parameters not logged.",
            e.filename,
        )
        return

    h_param_dict = _generate_h_param_dict(cl_args=c.cl_args, h_params=h_params)

    min_loss = average_loss_df["loss-average"].min()
    max_perf = average_loss_df["perf-average"].max()

    writer.add_hparams(
        h_param_dict,
        {"validation_loss-overall_min": min_loss, "best_overall_performance": max_perf},
    )


def _generate_h_param_dict(
    cl_args: Namespace, h_params: List[str]
) -> Dict[str, Union[str, float, int]]:

    h_param_dict = {}

    for param_name in h_params:
        param_value = getattr(cl_args, param_name)

        if isinstance(param_value, (tuple, list)):
            param_value = "_".join([str(p) for p in param_value])
        elif param_value is None:
            param_value = str(param_value)

        h_param_dict[param_name] = param_value

    return h_param_dict
