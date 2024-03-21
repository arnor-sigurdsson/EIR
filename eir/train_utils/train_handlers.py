import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import aislib.misc_utils
import numpy as np
import torch
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList, events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import RunningAverage
from ignite.metrics.metric import Metric, RunningBatchWise
from torch import nn

from eir.data_load.data_utils import get_output_info_generator
from eir.interpretation.interpretation import attribution_analysis_handler
from eir.setup.config_setup_modules.config_setup_utils import object_to_primitives
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.distributed import (
    AttrDelegatedDistributedDataParallel,
    only_call_on_master_node,
)
from eir.train_utils.evaluation import validation_handler
from eir.train_utils.lr_scheduling import attach_lr_scheduler, set_up_lr_scheduler
from eir.train_utils.metrics import (
    MetricRecord,
    al_metric_record_dict,
    get_average_history_filepath,
    get_buffered_metrics_writer,
    get_metrics_dataframes,
    get_metrics_files,
    persist_metrics,
    read_metrics_history_file,
)
from eir.train_utils.optim import AttrDelegatedSWAWrapper
from eir.train_utils.utils import get_run_folder, validate_handler_dependencies
from eir.utils.logging import get_logger
from eir.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from eir.setup.config import Configs
    from eir.train import Experiment
    from eir.train_utils.metrics import al_step_metric_dict

# Aliases
al_handler = Callable[[Engine, "HandlerConfig"], None]
al_event = EventsList | CallableEventWithFilter
al_handler_and_event = Tuple[al_handler, al_event]
al_sample_interval_handlers = Tuple[al_handler_and_event, ...]


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class HandlerConfig:
    experiment: "Experiment"
    run_folder: Path
    output_folder: str
    monitoring_metrics: List[Tuple[str, str, str]]


def configure_trainer(
    trainer: Engine,
    experiment: "Experiment",
    validation_handler_callable: al_handler = validation_handler,
) -> Engine:
    gc = experiment.configs.global_config
    run_folder = get_run_folder(output_folder=gc.output_folder)

    monitoring_metrics = _get_monitoring_metrics(
        outputs_as_dict=experiment.outputs, metric_record_dict=experiment.metrics
    )

    handler_config = HandlerConfig(
        experiment=experiment,
        run_folder=run_folder,
        output_folder=gc.output_folder,
        monitoring_metrics=monitoring_metrics,
    )

    train_monitoring_metrics = _parse_metrics_for_train_running_average(
        monitoring_metrics=monitoring_metrics
    )
    _call_and_undo_ignite_local_rank_side_effects(
        func=_attach_running_average_metrics,
        kwargs={"engine": trainer, "monitoring_metrics": train_monitoring_metrics},
    )

    _maybe_attach_progress_bar(trainer=trainer, do_not_attach=gc.no_pbar)

    _attach_sample_interval_handlers(
        trainer=trainer,
        handler_config=handler_config,
        validation_handler_callable=validation_handler_callable,
    )

    if gc.early_stopping_patience:
        _attach_early_stopping_handler(trainer=trainer, handler_config=handler_config)

    # TODO: Implement warmup for same LR scheduling
    if gc.lr_schedule != "same":
        lr_scheduler = set_up_lr_scheduler(handler_config=handler_config)
        attach_lr_scheduler(
            engine=trainer, lr_scheduler=lr_scheduler, experiment=experiment
        )
    elif gc.lr_schedule == "same" and gc.warmup_steps:
        raise NotImplementedError("Warmup not yet implemented for 'same' LR schedule.")

    if handler_config.output_folder:
        _attach_run_event_handlers(trainer=trainer, handler_config=handler_config)

    return trainer


@only_call_on_master_node
def _attach_sample_interval_handlers(
    trainer: Engine,
    handler_config: "HandlerConfig",
    validation_handler_callable: Callable = validation_handler,
) -> Engine:
    exp = handler_config.experiment
    gc = exp.configs.global_config

    validation_handler_and_event = _get_validation_handler_and_event(
        sample_interval_base=gc.sample_interval,
        iter_per_epoch=len(exp.train_loader),
        n_epochs=gc.n_epochs,
        early_stopping_patience=gc.early_stopping_patience,
        validation_handler_callable=validation_handler_callable,
    )
    all_handler_events = [validation_handler_and_event]

    if gc.compute_attributions:
        attribution_handler_and_event = _get_attribution_handler_and_event(
            iter_per_epoch=len(exp.train_loader),
            n_epochs=gc.n_epochs,
            sample_interval_base=gc.sample_interval,
            attributions_every_sample_factor=gc.attributions_every_sample_factor,
            early_stopping_patience=gc.early_stopping_patience,
        )
        all_handler_events.append(attribution_handler_and_event)

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
    validation_handler_callable: Callable,
) -> al_handler_and_event:
    validation_event: al_event = Events.ITERATION_COMPLETED(every=sample_interval_base)

    do_run_when_training_complete = _do_run_completed_handler(
        iter_per_epoch=iter_per_epoch,
        n_epochs=n_epochs,
        sample_interval=sample_interval_base,
    )

    if do_run_when_training_complete and not early_stopping_patience:
        validation_event = validation_event | Events.COMPLETED

    return validation_handler_callable, validation_event


def _get_attribution_handler_and_event(
    iter_per_epoch: int,
    n_epochs: int,
    sample_interval_base: int,
    attributions_every_sample_factor: int,
    early_stopping_patience: int,
) -> al_handler_and_event:
    attribution_handler_callable = attribution_analysis_handler

    if attributions_every_sample_factor == 0:
        attribution_event: al_event = Events.COMPLETED
        logger.debug("Attributions will be computed at run end.")

        return attribution_handler_callable, attribution_event

    attribution_handler_interval = (
        sample_interval_base * attributions_every_sample_factor
    )
    attribution_event = Events.ITERATION_COMPLETED(every=attribution_handler_interval)

    do_run_when_training_complete = _do_run_completed_handler(
        iter_per_epoch=iter_per_epoch,
        n_epochs=n_epochs,
        sample_interval=attribution_handler_interval,
    )

    if do_run_when_training_complete and not early_stopping_patience:
        attribution_event = attribution_event | Events.COMPLETED

    logger.debug(
        "Attributions will be computed every %d iterations.",
        attribution_handler_interval,
    )

    return attribution_handler_callable, attribution_event


def _do_run_completed_handler(iter_per_epoch: int, n_epochs: int, sample_interval: int):
    """
    We need this function to avoid running handlers twice in cases where the total
    number of iterations has a remainder of 0 w.r.t. the sample interval.
    """
    if (iter_per_epoch * n_epochs) % sample_interval != 0:
        return True

    return False


def _get_monitoring_metrics(
    outputs_as_dict: al_output_objects_as_dict,
    metric_record_dict: al_metric_record_dict,
) -> List[Tuple[str, str, str]]:
    """
    The spec for the tuple here follows the metric dict spec, i.e. the tuple is:
    (column_name, metric).
    """
    target_columns_gen = get_output_info_generator(outputs_as_dict=outputs_as_dict)

    loss_average_metrics: tuple[str, str, str] = ("average", "average", "loss-average")
    perf_average_metrics: tuple[str, str, str] = ("average", "average", "perf-average")
    monitoring_metrics: list[tuple[str, str, str]] = [
        loss_average_metrics,
        perf_average_metrics,
    ]

    def _parse_target_metrics(
        output_name_: str, metric_name: str, column_name_: str
    ) -> str:
        return f"{output_name_}_{column_name_}_{metric_name}"

    for output_name, output_target_type, column_name in target_columns_gen:
        if output_target_type in ("con", "cat"):
            cur_output_object = outputs_as_dict[output_name]
            cur_output_type = cur_output_object.output_config.output_info.output_type
            assert cur_output_type == "tabular"

            al_record = tuple[MetricRecord, ...]
            cur_record = metric_record_dict[output_target_type]
            assert isinstance(cur_record, tuple)
            cur_metric_records: al_record = cur_record

            for metric in cur_metric_records:
                if not metric.only_val:
                    parsed_metric = _parse_target_metrics(
                        output_name_=output_name,
                        column_name_=column_name,
                        metric_name=metric.name,
                    )
                    cur_tuple: tuple[str, str, str] = (
                        output_name,
                        column_name,
                        parsed_metric,
                    )
                    monitoring_metrics.append(cur_tuple)

        # manually add loss record as it's not in metric records, but from criteria
        loss_name = _parse_target_metrics(
            output_name_=output_name, metric_name="loss", column_name_=column_name
        )
        metrics_keys = (output_name, column_name, loss_name)
        monitoring_metrics.append(metrics_keys)

    return monitoring_metrics


@validate_handler_dependencies(
    [validation_handler],
)
@only_call_on_master_node
def _attach_early_stopping_handler(trainer: Engine, handler_config: "HandlerConfig"):
    gc = handler_config.experiment.configs.global_config

    early_stopping_handler = _get_early_stopping_handler(
        trainer=trainer,
        handler_config=handler_config,
        patience_steps=gc.early_stopping_patience,
    )

    early_stopping_event_kwargs = _get_early_stopping_event_filter_kwargs(
        early_stopping_iteration_buffer=gc.early_stopping_buffer,
        sample_interval=gc.sample_interval,
    )

    assert isinstance(early_stopping_event_kwargs, dict)
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

    logger.debug(
        "Setting early stopping patience to %d validation steps.", patience_steps
    )

    handler = EarlyStopping(
        patience=patience_steps, score_function=scoring_function, trainer=trainer
    )
    handler.logger = logger

    return handler


@overload
def _get_early_stopping_event_filter_kwargs(
    early_stopping_iteration_buffer: None, sample_interval: int
) -> dict[str, int]: ...


@overload
def _get_early_stopping_event_filter_kwargs(
    early_stopping_iteration_buffer: int, sample_interval: int
) -> dict[str, Callable[[Engine, int], bool]]: ...


def _get_early_stopping_event_filter_kwargs(
    early_stopping_iteration_buffer, sample_interval
):
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


def _get_latest_validation_value_score_function(
    run_folder: Path, column: str
) -> Callable[[Engine], float]:
    eval_history_fpath = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="validation_"
    )

    def scoring_function(engine: Engine) -> float:
        eval_df = read_metrics_history_file(eval_history_fpath)
        latest_val_loss = eval_df[column].iloc[-1]
        return latest_val_loss

    return scoring_function


def _parse_metrics_for_train_running_average(
    monitoring_metrics: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    """
    We skip computing performance averages on training batches,
    as some metrics (e.g. ROC-AUC, R2) can be undefined (e.g. batch with only
    positive or negative labels) or fluctuate a lot (e.g. R2).
    """
    to_skip = ("perf-average",)

    parsed_metrics: List[Tuple[str, str, str]] = []

    for output_name, column_name, metric_name in monitoring_metrics:
        if metric_name in to_skip:
            continue

        parsed_metrics.append((output_name, column_name, metric_name))

    return parsed_metrics


class MyRunningAverage(RunningAverage):
    def __init__(
        self,
        src: Optional[Metric] = None,
        alpha: float = 0.98,
        output_transform: Optional[Callable] = None,
        epoch_bound: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(
            src=src,
            alpha=alpha,
            output_transform=output_transform,
            epoch_bound=epoch_bound,
            device=device,
        )

    def update(self, output: Union[torch.Tensor, float]) -> None:
        """
        We have this checking for nan to avoid corrupting / making the running
        average always become nan if a NaN output value (due to an empty output
        batch for a given target) being encountered.
        """
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                return
        else:
            if np.isnan(output):
                return

        super().update(output)


def _attach_running_average_metrics(
    engine: Engine, monitoring_metrics: List[Tuple[str, str, str]]
) -> None:
    """
    For each metric, we create an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has its own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for output_name, column_name, metric_name in monitoring_metrics:

        def output_transform(
            metric_dict_from_step: "al_step_metric_dict",
            output_name_key: str,
            column_name_key: str,
            metric_name_key: str,
        ) -> float:
            value = metric_dict_from_step[output_name_key][column_name_key][
                metric_name_key
            ]
            return value

        partial_func = partial(
            output_transform,
            output_name_key=output_name,
            column_name_key=column_name,
            metric_name_key=metric_name,
        )

        running_average = MyRunningAverage(
            output_transform=partial_func,
            alpha=0.98,
        )

        running_average.attach(
            engine,
            name=metric_name,
            usage=RunningBatchWise(),
        )


def _call_and_undo_ignite_local_rank_side_effects(func: Callable, kwargs: Dict):
    """
    This weird function is needed in the case where a GPU is available, calling some
    functions will trigger obscure ignite side effects that change some environment
    variables without warning.
    """
    original_local_rank = int(os.environ.get("LOCAL_RANK", 0))

    result = func(**kwargs)

    cur_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if cur_local_rank != original_local_rank:
        logger.debug(
            "Enforcing local rank to be '%d' after ignite side effects",
            original_local_rank,
        )
        os.environ["LOCAL_RANK"] = str(original_local_rank)

    return result


@only_call_on_master_node
def _maybe_attach_progress_bar(trainer: Engine, do_not_attach: bool) -> None:
    do_attach = not do_not_attach
    if do_attach:
        pbar = ProgressBar()
        pbar.attach(engine=trainer, metric_names=["loss-average"])
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=_log_stats_to_pbar,
            pbar=pbar,
        )


def _log_stats_to_pbar(engine: Engine, pbar: ProgressBar) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    key = "loss-average"
    value = engine.state.metrics[key]
    log_string += f" | {key}: {value:.4g}"

    pbar.log_message(log_string)


@only_call_on_master_node
def _attach_run_event_handlers(trainer: Engine, handler_config: HandlerConfig):
    exp = handler_config.experiment
    gc = handler_config.experiment.configs.global_config

    _save_yaml_configs(run_folder=handler_config.run_folder, configs=exp.configs)

    if gc.checkpoint_interval is not None:
        trainer = _add_checkpoint_handler_wrapper(
            trainer=trainer,
            run_folder=handler_config.run_folder,
            output_folder=Path(gc.output_folder),
            n_to_save=gc.n_saved_models,
            checkpoint_interval=gc.checkpoint_interval,
            sample_interval=gc.sample_interval,
            model=exp.model,
        )

    metric_writing_funcs = _get_metric_writing_funcs(
        sample_interval=gc.sample_interval,
        outputs_as_dict=exp.outputs,
        run_folder=handler_config.run_folder,
    )
    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=_write_training_metrics_handler,
        handler_config=handler_config,
        writer_funcs=metric_writing_funcs,
    )

    for plot_event in _get_plot_events(sample_interval=gc.sample_interval):
        if plot_event == Events.COMPLETED and not _do_run_completed_handler(
            iter_per_epoch=len(exp.train_loader),
            n_epochs=gc.n_epochs,
            sample_interval=gc.sample_interval,
        ):
            continue

        trainer = _attach_plot_progress_handler(
            trainer=trainer, plot_event=plot_event, handler_config=handler_config
        )

    if exp.hooks.custom_handler_attachers is not None:
        custom_handlers = _get_custom_handlers(handler_config=handler_config)
        trainer = _attach_custom_handlers(
            trainer=trainer,
            handler_config=handler_config,
            custom_handlers=custom_handlers,
        )

    return trainer


def _save_yaml_configs(run_folder: Path, configs: "Configs"):
    for config_name, config_object in configs.__dict__.items():
        cur_output_path = Path(run_folder / "configs" / config_name).with_suffix(
            ".yaml"
        )
        aislib.misc_utils.ensure_path_exists(path=cur_output_path)

        config_object_as_primitives = object_to_primitives(obj=config_object)

        with open(str(cur_output_path), "w") as yaml_file_handle:
            yaml.dump(data=config_object_as_primitives, stream=yaml_file_handle)


def _add_checkpoint_handler_wrapper(
    trainer: Engine,
    run_folder: Path,
    output_folder: Path,
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

    assert n_to_save is not None
    assert checkpoint_score_function is not None
    assert score_name is not None
    checkpoint_handler = _get_checkpoint_handler(
        run_folder=run_folder,
        output_folder=output_folder,
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


def _get_metric_writing_funcs(
    sample_interval: int, outputs_as_dict: al_output_objects_as_dict, run_folder: Path
) -> Dict[str, Dict[str, Callable]]:
    buffer_interval = sample_interval // 2

    target_generator = get_output_info_generator(outputs_as_dict=outputs_as_dict)

    metrics_files = get_metrics_files(
        target_generator=target_generator,
        run_folder=run_folder,
        train_or_val_target_prefix="train_",
    )

    writer_funcs: Dict[str, Dict[str, Callable]] = {}
    for output_name, target_name_file_dict in metrics_files.items():
        writer_funcs[output_name] = {}

        for target_name, target_file in target_name_file_dict.items():
            writer_funcs[output_name][target_name] = get_buffered_metrics_writer(
                buffer_interval=buffer_interval
            )

    return writer_funcs


def _get_checkpoint_handler(
    run_folder: Path,
    output_folder: Path,
    n_to_save: int,
    score_function: Optional[Callable] = None,
    score_name: Optional[str] = None,
) -> ModelCheckpoint:
    def _default_global_step_transform(engine: Engine, event_name: str) -> int:
        return engine.state.iteration

    checkpoint_handler = ModelCheckpoint(
        dirname=str(Path(run_folder, "saved_models")),
        filename_prefix=output_folder.name,
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
    model: Union[
        nn.Module, AttrDelegatedSWAWrapper, AttrDelegatedDistributedDataParallel
    ],
) -> Engine:
    match model:
        case AttrDelegatedSWAWrapper() | AttrDelegatedDistributedDataParallel():
            model_to_save = model.module
        case _:
            model_to_save = model

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(every=checkpoint_interval),
        handler=checkpoint_handler,
        to_save={"model": model_to_save},
    )

    return trainer


def _write_training_metrics_handler(
    engine: Engine,
    handler_config: HandlerConfig,
    writer_funcs: Union[Dict[str, Callable], None] = None,
):
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

    engine_output = engine.state.output
    engine_metrics_dict = engine.state.metrics
    assert isinstance(engine_output, dict)
    assert isinstance(engine_metrics_dict, dict)

    running_average_metrics = _unflatten_engine_metrics_dict(
        step_base=engine_output,
        engine_metrics_dict=engine_metrics_dict,
    )
    persist_metrics(
        handler_config=handler_config,
        metrics_dict=running_average_metrics,
        iteration=iteration,
        write_header=is_first_iteration,
        prefixes={"metrics": "train_", "writer": "train"},
        writer_funcs=writer_funcs,
    )


def _unflatten_engine_metrics_dict(
    step_base: "al_step_metric_dict", engine_metrics_dict: Dict[str, float]
) -> "al_step_metric_dict":
    """
    We need this to streamline the 1D dictionary that comes from engine.state.metrics.
    """

    nested_dict: "al_step_metric_dict" = {}

    for output_name, output_metric_dict in step_base.items():
        nested_dict[output_name] = {}

        for target_name, target_metric_dict in output_metric_dict.items():
            nested_dict[output_name][target_name] = {}

            for target_metric_name in target_metric_dict.keys():
                eng_run_avg_value = engine_metrics_dict[target_metric_name]
                nested_dict[output_name][target_name][
                    target_metric_name
                ] = eng_run_avg_value

    return nested_dict


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


def _iterdir_ignore_hidden(path: Path) -> Iterator[Path]:
    """
    Mostly to avoid hidden files like .DS_Store on Macs.
    """
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        yield child


def _plot_progress_handler(engine: Engine, handler_config: HandlerConfig) -> None:
    ca = handler_config.experiment.configs.global_config

    # if no val data is available yet
    if engine.state.iteration < ca.sample_interval:
        return

    run_folder = get_run_folder(ca.output_folder)

    for output_dir in _iterdir_ignore_hidden(path=run_folder / "results"):
        for target_dir in _iterdir_ignore_hidden(path=output_dir):
            target_column = target_dir.name

            train_history_df, valid_history_df = get_metrics_dataframes(
                results_dir=target_dir, target_string=target_column
            )

            vf.generate_all_training_curves(
                training_history_df=train_history_df,
                valid_history_df=valid_history_df,
                output_folder=target_dir,
                title_extra=target_column,
                plot_skip_steps=ca.plot_skip_steps,
            )

    train_avg_history_df, valid_avg_history_df = get_metrics_dataframes(
        results_dir=run_folder, target_string="average"
    )

    vf.generate_all_training_curves(
        training_history_df=train_avg_history_df,
        valid_history_df=valid_avg_history_df,
        output_folder=run_folder,
        title_extra="Multi Task Average",
        plot_skip_steps=ca.plot_skip_steps,
    )


def _get_custom_handlers(handler_config: "HandlerConfig"):
    custom_handlers = handler_config.experiment.hooks.custom_handler_attachers

    return custom_handlers


def _attach_custom_handlers(
    trainer: Engine, handler_config: "HandlerConfig", custom_handlers
):
    if not custom_handlers:
        return trainer

    for custom_handler_attacher in custom_handlers:
        trainer = custom_handler_attacher(trainer, handler_config)

    return trainer
