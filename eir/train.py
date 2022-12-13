import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    TYPE_CHECKING,
    Callable,
    Iterable,
    Any,
    Sequence,
)

import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from eir.data_load import data_utils
from eir.data_load import datasets
from eir.data_load.data_augmentation import (
    hook_mix_loss,
    get_mix_data_hook,
)
from eir.data_load.data_utils import Batch, get_train_sampler
from eir.data_load.label_setup import (
    al_all_column_ops,
    al_label_dict,
    al_label_transformers,
    set_up_train_and_valid_tabular_data,
    gather_ids_from_tabular_file,
    gather_ids_from_data_source,
    split_ids,
    TabularFileInfo,
    save_transformer_set,
)
from eir.experiment_io.experiment_io import (
    serialize_experiment,
    get_default_experiment_keys_to_serialize,
    serialize_all_input_transformers,
    serialize_chosen_input_objects,
)
from eir.models import al_fusion_models
from eir.models import model_training_utils
from eir.models.model_setup import get_model, get_default_model_registry_per_input_type
from eir.models.model_training_utils import run_lr_find
from eir.models.tabular.tabular import (
    get_tabular_inputs,
)
from eir.setup import schemas
from eir.setup.config import (
    get_configs,
    Configs,
    get_all_tabular_targets,
)
from eir.setup.input_setup import al_input_objects_as_dict, set_up_inputs_for_training
from eir.setup.output_setup import (
    al_output_objects_as_dict,
    set_up_outputs_for_training,
)
from eir.train_utils import distributed
from eir.train_utils import utils
from eir.train_utils.metrics import (
    calculate_batch_metrics,
    calculate_prediction_losses,
    aggregate_losses,
    add_multi_task_average_metrics,
    get_average_history_filepath,
    get_default_metrics,
    hook_add_l1_loss,
    get_uncertainty_loss_hook,
    add_loss_to_metrics,
)
from eir.train_utils.optimizers import (
    get_optimizer,
    get_optimizer_backward_kwargs,
)
from eir.train_utils.train_handlers import HandlerConfig
from eir.train_utils.train_handlers import configure_trainer
from eir.train_utils.utils import (
    call_hooks_stage_iterable,
)

if TYPE_CHECKING:
    from eir.train_utils.metrics import (
        al_step_metric_dict,
        al_metric_record_dict,
    )

# aliases
al_criteria = Dict[str, Dict[str, Union[nn.CrossEntropyLoss, nn.MSELoss]]]
# these are all after being collated by torch dataloaders
# output name -> target column: value
al_training_labels_target = Dict[str, Dict[str, torch.Tensor]]
al_input_batch = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
al_ids = List[str]

al_dataloader_getitem_batch = Tuple[
    al_input_batch,
    al_training_labels_target,
    al_ids,
]

utils.seed_everything()
logger = get_logger(name=__name__, tqdm_compatible=True)


def main():
    torch.backends.cudnn.benchmark = True

    configs = get_configs()

    configs, local_rank = distributed.maybe_initialize_distributed_environment(
        configs=configs
    )

    utils.configure_root_logger(output_folder=configs.global_config.output_folder)

    default_hooks = get_default_hooks(configs=configs)
    default_experiment = get_default_experiment(
        configs=configs,
        hooks=default_hooks,
    )

    run_experiment(experiment=default_experiment)


def run_experiment(experiment: "Experiment") -> None:

    _log_model(model=experiment.model)

    gc = experiment.configs.global_config

    run_folder = utils.get_run_folder(output_folder=gc.output_folder)
    keys_to_serialize = get_default_experiment_keys_to_serialize()
    serialize_experiment(
        experiment=experiment,
        run_folder=run_folder,
        keys_to_serialize=keys_to_serialize,
    )

    train(experiment=experiment)


@dataclass(frozen=True)
class Experiment:
    configs: Configs
    inputs: al_input_objects_as_dict
    outputs: al_output_objects_as_dict
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: torch.utils.data.Dataset
    model: al_fusion_models
    optimizer: Optimizer
    criteria: al_criteria
    loss_function: Callable
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union["Hooks", None]


@dataclass
class MergedTargetLabels:
    train_labels: al_label_dict
    valid_labels: al_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return {**self.train_labels, **self.valid_labels}


def set_up_tabular_target_labels_wrapper(
    tabular_target_file_infos: Dict[str, TabularFileInfo],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> MergedTargetLabels:

    df_labels_train = pd.DataFrame(index=train_ids)
    df_labels_valid = pd.DataFrame(index=valid_ids)
    label_transformers = {}

    for output_name, tabular_info in tabular_target_file_infos.items():
        cur_labels = set_up_train_and_valid_tabular_data(
            tabular_file_info=tabular_info,
            custom_label_ops=custom_label_ops,
            train_ids=train_ids,
            valid_ids=valid_ids,
        )

        df_train_cur = pd.DataFrame.from_dict(cur_labels.train_labels, orient="index")
        df_valid_cur = pd.DataFrame.from_dict(cur_labels.valid_labels, orient="index")

        df_train_cur["Output Name"] = output_name
        df_valid_cur["Output Name"] = output_name

        df_labels_train = pd.concat((df_labels_train, df_train_cur))
        df_labels_valid = pd.concat((df_labels_valid, df_valid_cur))

        cur_transformers = cur_labels.label_transformers
        label_transformers[output_name] = cur_transformers

    df_labels_train = df_labels_train.set_index("Output Name", append=True)
    df_labels_valid = df_labels_valid.set_index("Output Name", append=True)

    df_labels_train = df_labels_train.dropna(how="all")
    df_labels_valid = df_labels_valid.dropna(how="all")

    train_labels_dict = df_to_nested_dict(df=df_labels_train)
    valid_labels_dict = df_to_nested_dict(df=df_labels_valid)

    labels_data_object = MergedTargetLabels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
    )

    return labels_data_object


def df_to_nested_dict(df: pd.DataFrame) -> Dict:

    """
    The df has a 2-level multi index, like so ['ID', output_name]

    We want to convert it to a nested dict like so:

        {'ID': {output_name: {target_output_column: target_column_value}}}
    """
    index_dict = df.to_dict(orient="index")

    parsed_dict = {}
    for key_tuple, value in index_dict.items():
        cur_id, cur_output_name = key_tuple

        if cur_id not in parsed_dict:
            parsed_dict[cur_id] = {}

        parsed_dict[cur_id][cur_output_name] = value

    return parsed_dict


def get_default_experiment(
    configs: Configs,
    hooks: Union["Hooks", None] = None,
) -> "Experiment":
    run_folder = _prepare_run_folder(output_folder=configs.global_config.output_folder)

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=configs.output_configs
    )
    manual_valid_ids = _read_manual_ids_if_exist(
        manual_valid_ids_file=configs.global_config.manual_valid_ids_file
    )

    train_ids, valid_ids = split_ids(
        ids=all_array_ids,
        valid_size=configs.global_config.valid_size,
        manual_valid_ids=manual_valid_ids,
    )

    logger.info("Setting up target labels.")
    target_labels_info = get_tabular_target_file_infos(
        output_configs=configs.output_configs
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    target_labels = set_up_tabular_target_labels_wrapper(
        tabular_target_file_infos=target_labels_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    save_transformer_set(
        transformers_per_source=target_labels.label_transformers, run_folder=run_folder
    )

    inputs_as_dict = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=hooks,
    )

    serialize_all_input_transformers(inputs_dict=inputs_as_dict, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs_as_dict, run_folder=run_folder)

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=configs.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs_as_dict,
        outputs_as_dict=outputs_as_dict,
    )

    train_sampler = get_train_sampler(
        columns_to_sample=configs.global_config.weighted_sampling_columns,
        train_dataset=train_dataset,
    )

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=configs.global_config.batch_size,
        num_workers=configs.global_config.dataloader_workers,
    )

    default_registry = get_default_model_registry_per_input_type()

    model = get_model(
        inputs_as_dict=inputs_as_dict,
        global_config=configs.global_config,
        fusion_config=configs.fusion_config,
        outputs_as_dict=outputs_as_dict,
        model_registry_per_input_type=default_registry,
        model_registry_per_output_type={},
    )

    criteria = _get_criteria(
        outputs_as_dict=outputs_as_dict,
    )

    writer = get_summary_writer(run_folder=run_folder)

    loss_func = _get_loss_callable(
        criteria=criteria,
    )

    optimizer = get_optimizer(
        model=model, loss_callable=loss_func, global_config=configs.global_config
    )

    metrics = get_default_metrics(target_transformers=target_labels.label_transformers)

    experiment = Experiment(
        configs=configs,
        inputs=inputs_as_dict,
        outputs=outputs_as_dict,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_func,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    return experiment


def gather_all_ids_from_output_configs(
    output_configs: Sequence[schemas.OutputConfig],
) -> Tuple[str, ...]:
    all_ids = set()
    for config in output_configs:
        cur_source = Path(config.output_info.output_source)
        if cur_source.suffix == ".csv":
            cur_ids = gather_ids_from_tabular_file(file_path=cur_source)
        elif cur_source.is_dir():
            cur_ids = gather_ids_from_data_source(data_source=cur_source)
        else:
            raise NotImplementedError()
        all_ids.update(cur_ids)

    return tuple(all_ids)


def _read_manual_ids_if_exist(
    manual_valid_ids_file: Union[None, str]
) -> Union[Sequence[str], None]:

    if not manual_valid_ids_file:
        return None

    with open(manual_valid_ids_file, "r") as infile:
        manual_ids = tuple(line.strip() for line in infile)

    return manual_ids


def get_tabular_target_file_infos(
    output_configs: Iterable[schemas.OutputConfig],
) -> Dict[str, TabularFileInfo]:

    tabular_infos = {}

    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            raise NotImplementedError()

        output_name = output_config.output_info.output_name
        tabular_info = TabularFileInfo(
            file_path=Path(output_config.output_info.output_source),
            con_columns=output_config.output_type_info.target_con_columns,
            cat_columns=output_config.output_type_info.target_cat_columns,
            parsing_chunk_size=output_config.output_type_info.label_parsing_chunk_size,
        )
        tabular_infos[output_name] = tabular_info

    return tabular_infos


def _prepare_run_folder(output_folder: str) -> Path:
    run_folder = utils.get_run_folder(output_folder=output_folder)
    history_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix="train_"
    )
    if history_file.exists():
        raise FileExistsError(
            f"There already exists a run with that name: {history_file}. Please choose "
            f"a different run name or delete the folder."
        )

    ensure_path_exists(path=run_folder, is_folder=True)

    return run_folder


def get_dataloaders(
    train_dataset: datasets.DatasetBase,
    train_sampler: Union[None, WeightedRandomSampler],
    valid_dataset: datasets.DatasetBase,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple:

    check_dataset_and_batch_size_compatiblity(
        dataset=train_dataset, batch_size=batch_size, name="Training"
    )

    check_dataset_and_batch_size_compatiblity(
        dataset=valid_dataset, batch_size=batch_size, name="Validation"
    )
    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    valid_dloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_dloader, valid_dloader


def check_dataset_and_batch_size_compatiblity(
    dataset: datasets.DatasetBase, batch_size: int, name: str = ""
):
    if len(dataset) < batch_size:
        raise ValueError(
            f"{name} dataset size ({len(dataset)}) can not be smaller than "
            f"batch size ({batch_size}). A fix can be increasing {name.lower()} sample "
            f"size or reducing the batch size. If predicting on few unknown samples, "
            f"a solution can be setting the batch size to 1 in the global configuration"
            f" passed to the prediction module. Future work includes making this "
            f"easier to work with."
        )


def _get_criteria(outputs_as_dict: al_output_objects_as_dict) -> al_criteria:
    criteria_dict = {}

    def get_criterion(
        column_type_: str, cat_label_smoothing_: float = 0.0
    ) -> Union[nn.CrossEntropyLoss, Callable]:

        if column_type_ == "con":
            assert cat_label_smoothing_ == 0.0
            return partial(_calc_mse, mse_loss_func=nn.MSELoss())
        elif column_type_ == "cat":
            return nn.CrossEntropyLoss(label_smoothing=cat_label_smoothing_)

    target_columns_gen = data_utils.get_output_info_generator(
        outputs_as_dict=outputs_as_dict
    )

    for output_name, column_type, column_name in target_columns_gen:

        label_smoothing = _get_label_smoothing(
            output_config=outputs_as_dict[output_name].output_config,
            column_type=column_type,
        )

        criterion = get_criterion(
            column_type_=column_type, cat_label_smoothing_=label_smoothing
        )

        if output_name not in criteria_dict:
            criteria_dict[output_name] = {}
        criteria_dict[output_name][column_name] = criterion

    return criteria_dict


def _get_label_smoothing(
    output_config: schemas.OutputConfig,
    column_type: str,
) -> float:

    if column_type == "con":
        return 0.0
    elif column_type == "cat":
        return output_config.output_type_info.cat_label_smoothing

    raise ValueError(f"Unknown column type: {column_type}")


def _calc_mse(input, target, mse_loss_func: nn.MSELoss):
    return mse_loss_func(input=input.squeeze(), target=target.squeeze())


def _get_loss_callable(criteria: al_criteria):

    single_task_loss_func = partial(calculate_prediction_losses, criteria=criteria)
    return single_task_loss_func


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_model(model: nn.Module) -> None:
    """
    TODO: Add summary of parameters
    TODO: Add verbosity option
    """
    no_params = sum(p.numel() for p in model.parameters())
    no_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Starting training with a %s parameter model. " "Num trainable parameters: %s.",
        format(no_params, ",.0f"),
        format(no_trainable_params, ",.0f"),
    )


def get_base_trainer(experiment: Experiment) -> Engine:
    step_hooks = experiment.hooks.step_func_hooks

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_target, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        experiment.model.train()
        experiment.optimizer.zero_grad()

        base_prepare_inputs_stage = step_hooks.base_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=base_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": loader_batch},
            state=None,
        )
        base_batch = state["batch"]

        post_prepare_inputs_stage = step_hooks.post_prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=post_prepare_inputs_stage,
            common_kwargs={"experiment": experiment, "loader_batch": base_batch},
            state=state,
        )
        batch = state["batch"]

        model_forward_loss_stage = step_hooks.model_forward
        state = call_hooks_stage_iterable(
            hook_iterable=model_forward_loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        loss_stage = step_hooks.loss
        state = call_hooks_stage_iterable(
            hook_iterable=loss_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        optimizer_backward_stage = step_hooks.optimizer_backward
        state = call_hooks_stage_iterable(
            hook_iterable=optimizer_backward_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        metrics_stage = step_hooks.metrics
        state = call_hooks_stage_iterable(
            hook_iterable=metrics_stage,
            common_kwargs={"experiment": experiment, "batch": batch},
            state=state,
        )

        return state["metrics"]

    trainer = Engine(process_function=step)

    return trainer


def train(experiment: Experiment) -> None:
    exp = experiment
    gc = experiment.configs.global_config

    trainer = get_base_trainer(experiment=experiment)

    if gc.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=exp.train_loader,
            model=exp.model,
            optimizer=exp.optimizer,
            output_folder=utils.get_run_folder(output_folder=gc.output_folder),
        )
        sys.exit(0)

    trainer = configure_trainer(trainer=trainer, experiment=experiment)

    trainer.run(data=exp.train_loader, max_epochs=gc.n_epochs)


def get_default_hooks(configs: Configs):
    step_func_hooks = _get_default_step_function_hooks(configs=configs)
    hooks_object = Hooks(step_func_hooks=step_func_hooks)

    return hooks_object


@dataclass
class Hooks:
    al_handler_attachers = Iterable[Callable[[Engine, HandlerConfig], Engine]]

    step_func_hooks: "StepFunctionHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None
    custom_handler_attachers: Union[None, al_handler_attachers] = None


def _get_default_step_function_hooks(configs: Configs):
    """
    TODO: Add validation, inspect that outputs have correct names.
    TODO: Refactor, split into smaller functions e.g. for L1, mixing and uncertainty.
    """

    init_kwargs = _get_default_step_function_hooks_init_kwargs(configs=configs)

    step_func_hooks = StepFunctionHookStages(**init_kwargs)

    return step_func_hooks


def _get_default_step_function_hooks_init_kwargs(
    configs: Configs,
) -> Dict[str, Sequence[Callable]]:

    init_kwargs = {
        "base_prepare_batch": [hook_default_prepare_batch],
        "post_prepare_batch": [],
        "model_forward": [hook_default_model_forward],
        "loss": [hook_default_per_target_loss],
        "optimizer_backward": [hook_default_optimizer_backward],
        "metrics": [hook_default_compute_metrics],
    }

    if configs.global_config.mixing_alpha:
        logger.debug(
            "Setting up hooks for mixing with with α=%.2g.",
            configs.global_config.mixing_alpha,
        )
        mix_hook = get_mix_data_hook(input_configs=configs.input_configs)

        init_kwargs["post_prepare_batch"].append(mix_hook)
        init_kwargs["loss"][0] = hook_mix_loss

    if _should_add_uncertainty_loss_hook(output_configs=configs.output_configs):
        uncertainty_hook = get_uncertainty_loss_hook(
            output_configs=configs.output_configs,
            device=configs.global_config.device,
        )
        init_kwargs["loss"].append(uncertainty_hook)

    init_kwargs["loss"].append(hook_default_aggregate_losses)

    init_kwargs = add_l1_loss_hook_if_applicable(
        step_function_hooks_init_kwargs=init_kwargs, configs=configs
    )

    grad_acc_steps = configs.global_config.gradient_accumulation_steps
    if grad_acc_steps and grad_acc_steps > 1:
        logger.debug(
            "Adding gradient accumulation hook with steps=%d.",
            configs.global_config.gradient_accumulation_steps,
        )
        init_kwargs["loss"].append(get_hook_iteration_counter())

    do_amp = configs.global_config.amp
    if do_amp:
        logger.debug("Setting up AMP training.")
        model_forward_with_amp_objects = [
            get_hook_amp_objects(device=configs.global_config.device)
        ] + init_kwargs["model_forward"]
        init_kwargs["model_forward"] = model_forward_with_amp_objects

    return init_kwargs


def _should_add_uncertainty_loss_hook(
    output_configs: Sequence[schemas.OutputConfig],
) -> bool:
    all_targets = get_all_tabular_targets(output_configs=output_configs)

    more_than_one_target = len(all_targets) > 1

    any_uncertainty_targets = any(
        c.output_type_info.uncertainty_weighted_mt_loss for c in output_configs
    )
    return more_than_one_target and any_uncertainty_targets


def add_l1_loss_hook_if_applicable(
    step_function_hooks_init_kwargs: Dict[str, Sequence[Callable]],
    configs: Configs,
) -> Dict[str, List[Callable]]:
    input_l1 = any(
        getattr(input_config.model_config.model_init_config, "l1", None)
        for input_config in configs.input_configs
    )
    preds_l1 = getattr(configs.fusion_config.model_config, "l1", None)
    if input_l1 or preds_l1:
        logger.info("Adding L1 loss hook.")
        step_function_hooks_init_kwargs["loss"].append(hook_add_l1_loss)

    return step_function_hooks_init_kwargs


@dataclass
class StepFunctionHookStages:

    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    base_prepare_batch: al_hooks
    post_prepare_batch: al_hooks
    model_forward: al_hooks
    loss: al_hooks
    optimizer_backward: al_hooks
    metrics: al_hooks


def hook_default_prepare_batch(
    experiment: "Experiment",
    loader_batch: al_dataloader_getitem_batch,
    *args,
    **kwargs,
) -> Dict:

    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        input_objects=experiment.inputs,
        output_objects=experiment.outputs,
        model=experiment.model,
        device=experiment.configs.global_config.device,
    )

    state_updates = {"batch": batch}

    return state_updates


def prepare_base_batch_default(
    loader_batch: al_dataloader_getitem_batch,
    input_objects: al_input_objects_as_dict,
    output_objects: al_output_objects_as_dict,
    model: nn.Module,
    device: str,
) -> Batch:

    inputs, target_labels, train_ids = loader_batch

    inputs_prepared = _prepare_inputs_for_model(
        batch_inputs=inputs, input_objects=input_objects, model=model, device=device
    )

    if target_labels:
        target_labels = model_training_utils.parse_target_labels(
            output_objects=output_objects,
            device=device,
            labels=target_labels,
        )

    batch = Batch(
        inputs=inputs_prepared,
        target_labels=target_labels,
        ids=train_ids,
    )

    return batch


def _prepare_inputs_for_model(
    batch_inputs: Dict[str, Any],
    input_objects: al_input_objects_as_dict,
    model: nn.Module,
    device: str,
) -> Dict[str, torch.Tensor]:
    inputs_prepared = {}
    for input_name, input_object in input_objects.items():
        input_type = input_object.input_config.input_info.input_type

        if input_type in ("omics", "image"):
            cur_tensor = batch_inputs[input_name]
            cur_tensor = cur_tensor.to(device=device)
            cur_tensor = cur_tensor.to(dtype=torch.float32)

            inputs_prepared[input_name] = cur_tensor

        elif input_type == "tabular":

            tabular_source_input = batch_inputs[input_name]
            for tabular_name, tensor in tabular_source_input.items():
                tabular_source_input[tabular_name] = tensor.to(device=device)

            tabular_input_type_info = input_object.input_config.input_type_info
            cat_columns = tabular_input_type_info.input_cat_columns
            con_columns = tabular_input_type_info.input_con_columns
            tabular = get_tabular_inputs(
                input_cat_columns=cat_columns,
                input_con_columns=con_columns,
                tabular_model=getattr(model.input_modules, input_name),
                tabular_input=tabular_source_input,
                device=device,
            )
            inputs_prepared[input_name] = tabular

        elif input_type in ("sequence", "bytes"):
            cur_seq = batch_inputs[input_name]
            cur_seq = cur_seq.to(device=device)
            cur_module = getattr(model.input_modules, input_name)
            cur_module_embedding = cur_module.embedding
            cur_embedding = cur_module_embedding(input=cur_seq)
            inputs_prepared[input_name] = cur_embedding
        else:
            raise ValueError(f"Unrecognized input type {input_name}.")

    return inputs_prepared


def hook_default_model_forward(
    experiment: "Experiment", state: Dict, batch: "Batch", *args, **kwargs
) -> Dict:

    inputs = batch.inputs

    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        train_outputs = experiment.model(inputs=inputs)

    state_updates = {"model_outputs": train_outputs}

    return state_updates


def get_amp_context_manager(device_type: str) -> autocast:
    return autocast(device_type=device_type)


def hook_default_optimizer_backward(
    experiment: "Experiment", state: Dict, *args, **kwargs
) -> Dict:

    optimizer_backward_kwargs = get_optimizer_backward_kwargs(
        optimizer_name=experiment.configs.global_config.optimizer
    )

    grad_acc_steps = experiment.configs.global_config.gradient_accumulation_steps

    if grad_acc_steps and grad_acc_steps > 1:
        loss = state["loss"] / grad_acc_steps
    else:
        loss = state["loss"]

    amp = experiment.configs.global_config.amp
    amp_gradient_scaler = None
    if amp and experiment.configs.global_config.device != "cpu":
        amp_gradient_scaler = state["amp_scaler"]
        loss = amp_gradient_scaler.scale(loss)

    loss.backward(**optimizer_backward_kwargs)

    gradient_noise = experiment.configs.global_config.gradient_noise
    if gradient_noise:
        for name, weight in experiment.model.named_parameters():
            weight.grad = weight.grad + torch.randn_like(weight.grad) * gradient_noise

    gradient_clipping = experiment.configs.global_config.gradient_clipping
    if gradient_clipping:
        clip_grad_norm_(
            parameters=experiment.model.parameters(),
            max_norm=gradient_clipping,
        )

    step_func = experiment.optimizer.step
    if amp and experiment.configs.global_config.device != "cpu":
        step_func = partial(amp_gradient_scaler.step, optimizer=experiment.optimizer)

    if grad_acc_steps and grad_acc_steps > 1:
        cur_step = state["iteration"]
        if cur_step % grad_acc_steps == 0:
            step_func()
    else:
        step_func()

    if amp and experiment.configs.global_config.device != "cpu":
        amp_gradient_scaler.update()

    return {}


def hook_default_compute_metrics(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
):

    train_batch_metrics = calculate_batch_metrics(
        outputs_as_dict=experiment.outputs,
        outputs=state["model_outputs"],
        labels=batch.target_labels,
        mode="train",
        metric_record_dict=experiment.metrics,
    )

    train_batch_metrics_w_loss = add_loss_to_metrics(
        outputs_as_dict=experiment.outputs,
        losses=state["per_target_train_losses"],
        metric_dict=train_batch_metrics,
    )

    train_batch_metrics_with_averages = add_multi_task_average_metrics(
        batch_metrics_dict=train_batch_metrics_w_loss,
        outputs_as_dict=experiment.outputs,
        loss=state["loss"].item(),
        performance_average_functions=experiment.metrics["averaging_functions"],
    )

    state_updates = {"metrics": train_batch_metrics_with_averages}

    return state_updates


def hook_default_per_target_loss(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
) -> Dict:

    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        per_target_train_losses = experiment.loss_function(
            inputs=state["model_outputs"], targets=batch.target_labels
        )

        state_updates = {"per_target_train_losses": per_target_train_losses}

    return state_updates


def hook_default_aggregate_losses(state: Dict, *args, **kwargs) -> Dict:
    context_manager = get_maybe_amp_context_manager_from_state(state=state)
    with context_manager:
        train_loss_avg = aggregate_losses(losses_dict=state["per_target_train_losses"])
        state_updates = {"loss": train_loss_avg}

    return state_updates


def get_maybe_amp_context_manager_from_state(
    state: Dict,
) -> Union[nullcontext, autocast]:
    context_manager = state.get("amp_context_manager", nullcontext())
    return context_manager


def get_hook_iteration_counter() -> Callable:
    iteration_count = 0

    def _counter_iterator(do_increment: bool = True, *args, **kwargs) -> Dict[str, int]:
        nonlocal iteration_count
        if do_increment:
            iteration_count += 1

        state_updates = {"iteration": iteration_count}
        return state_updates

    return _counter_iterator


def get_hook_amp_objects(device: str):
    device_type = "cpu" if device == "cpu" else "cuda"

    if device == "cpu":
        logger.warning("Using AMP is on a CPU, speedups will most likely be minimal.")

    scaler = None
    if device != "cpu":
        scaler = GradScaler()

    amp_context_manager = get_amp_context_manager(device_type=device_type)

    def _get_objects(*args, **kwargs) -> Dict[str, GradScaler]:
        state_updates = {
            "amp_context_manager": amp_context_manager,
        }
        if device != "cpu":
            state_updates["amp_scaler"] = scaler

        return state_updates

    return _get_objects


def hook_adjust_loss_for_gradient_accumulation(
    experiment: "Experiment", state: Dict, *args, **kwargs
) -> Dict:
    gradient_accumulation_steps = (
        experiment.configs.global_config.gradient_accumulation_steps
    )

    loss = state["loss"]
    loss_adjusted = loss / gradient_accumulation_steps

    state_updates = {"loss": loss_adjusted}

    return state_updates


if __name__ == "__main__":
    main()
