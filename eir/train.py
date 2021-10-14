import sys
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    TYPE_CHECKING,
    Callable,
    Iterable,
    Sequence,
    Any,
    Set,
    Type,
)

import dill
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from eir.data_load import data_utils
from eir.data_load import datasets
from eir.data_load.data_augmentation import hook_mix_loss, get_mix_data_hook
from eir.data_load.data_utils import Batch, get_train_sampler
from eir.data_load.label_setup import (
    al_target_columns,
    al_label_transformers,
    al_all_column_ops,
    set_up_train_and_valid_tabular_data,
    gather_ids_from_tabular_file,
    split_ids,
    TabularFileInfo,
    save_transformer_set,
    Labels,
)
from eir.models import fusion, fusion_mgmoe, fusion_linear, al_fusion_models
from eir.models import model_training_utils
from eir.models.model_training_utils import run_lr_find
from eir.models.omics.omics_models import (
    get_model_class,
    get_omics_model_init_kwargs,
    al_omics_model_configs,
)
from eir.models.sequence.transformer_basic import (
    BasicTransformerModelConfig,
    TransformerModel,
)
from eir.models.tabular.tabular import (
    get_tabular_inputs,
    SimpleTabularModel,
    get_unique_values_from_transformers,
)
from eir.setup import schemas
from eir.setup.config import (
    get_configs,
    Configs,
    get_all_targets,
)
from eir.setup.input_setup import (
    DataDimensions,
    serialize_all_input_transformers,
    serialize_all_sequence_inputs,
)
from eir.setup.input_setup import al_input_objects_as_dict, set_up_inputs_for_training
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
al_criterions = Dict[str, Union[nn.CrossEntropyLoss, nn.MSELoss]]
# these are all after being collated by torch dataloaders
al_training_labels_target = Dict[str, Union[torch.LongTensor, torch.Tensor]]
al_training_labels_extra = Dict[str, Union[List[str], torch.Tensor]]
al_training_labels_batch = Dict[
    str, Union[al_training_labels_target, al_training_labels_extra]
]
al_dataloader_getitem_batch = Tuple[
    Union[Dict[str, torch.Tensor], Dict[str, Any]],
    al_training_labels_target,
    List[str],
]
al_num_outputs_per_target = Dict[str, int]

utils.seed_everything()
logger = get_logger(name=__name__, tqdm_compatible=True)


def main():
    configs = get_configs()

    utils.configure_root_logger(run_name=configs.global_config.run_name)

    default_hooks = get_default_hooks(configs=configs)
    default_config = get_default_experiment(configs=configs, hooks=default_hooks)

    run_experiment(experiment=default_config)


def run_experiment(experiment: "Experiment") -> None:

    _log_model(model=experiment.model)

    gc = experiment.configs.global_config

    run_folder = utils.get_run_folder(run_name=gc.run_name)
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
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: torch.utils.data.Dataset
    target_transformers: al_label_transformers
    target_columns: al_target_columns
    num_outputs_per_target: al_num_outputs_per_target
    model: al_fusion_models
    optimizer: Optimizer
    criterions: al_criterions
    loss_function: Callable
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union["Hooks", None]


def serialize_experiment(
    experiment: "Experiment",
    run_folder: Path,
    keys_to_serialize: Union[Iterable[str], None],
) -> None:
    serialization_path = get_train_experiment_serialization_path(run_folder=run_folder)
    ensure_path_exists(path=serialization_path)

    filtered_experiment = filter_experiment_by_keys(
        experiment=experiment, keys=keys_to_serialize
    )
    serialize_namespace(namespace=filtered_experiment, output_path=serialization_path)


def get_train_experiment_serialization_path(run_folder: Path) -> Path:
    train_experiment_path = run_folder / "serializations" / "filtered_experiment.dill"

    return train_experiment_path


def get_default_experiment_keys_to_serialize() -> Iterable[str]:
    return (
        "configs",
        "num_outputs_per_target",
        "target_transformers",
        "target_columns",
        "metrics",
        "hooks",
    )


def filter_experiment_by_keys(
    experiment: "Experiment", keys: Union[None, Iterable[str]] = None
) -> SimpleNamespace:
    filtered = {}

    config_fields = (f.name for f in fields(experiment))
    iterable = keys if keys is not None else config_fields

    for k in iterable:
        filtered[k] = getattr(experiment, k)

    namespace = SimpleNamespace(**filtered)

    return namespace


def serialize_namespace(namespace: SimpleNamespace, output_path: Path) -> None:
    with open(output_path, "wb") as outfile:
        dill.dump(namespace, outfile)


def set_up_target_labels_wrapper(
    tabular_file_infos: Sequence[TabularFileInfo],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> Labels:
    """
    TODO:   Log if some were dropped on merge.
    TODO:   Decide if we want to keep this merging here, or have a key reference to
            target_name or something like that?
    """

    df_labels_train = pd.DataFrame(index=train_ids)
    df_labels_valid = pd.DataFrame(index=valid_ids)
    label_transformers = {}

    for tabular_info in tabular_file_infos:
        cur_labels = set_up_train_and_valid_tabular_data(
            tabular_file_info=tabular_info,
            custom_label_ops=custom_label_ops,
            train_ids=train_ids,
            valid_ids=valid_ids,
        )

        df_train_cur = pd.DataFrame.from_dict(cur_labels.train_labels, orient="index")
        df_valid_cur = pd.DataFrame.from_dict(cur_labels.valid_labels, orient="index")

        df_labels_train = pd.merge(
            df_labels_train, df_train_cur, left_index=True, right_index=True
        )
        df_labels_valid = pd.merge(
            df_labels_valid, df_valid_cur, left_index=True, right_index=True
        )

        cur_transformers = cur_labels.label_transformers
        label_transformers = {**label_transformers, **cur_transformers}

    train_labels_dict = df_labels_train.to_dict("index")
    valid_labels_dict = df_labels_valid.to_dict("index")

    labels_data_object = Labels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
    )

    return labels_data_object


def get_default_experiment(
    configs: Configs, hooks: Union["Hooks", None] = None
) -> "Experiment":
    run_folder = _prepare_run_folder(run_name=configs.global_config.run_name)

    all_array_ids = gather_all_ids_from_target_configs(
        target_configs=configs.target_configs
    )
    train_ids, valid_ids = split_ids(
        ids=all_array_ids, valid_size=configs.global_config.valid_size
    )

    logger.info("Setting up target labels.")
    target_labels_info = get_tabular_target_file_infos(
        target_configs=configs.target_configs
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    target_labels = set_up_target_labels_wrapper(
        tabular_file_infos=target_labels_info,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    save_transformer_set(
        transformers=target_labels.label_transformers, run_folder=run_folder
    )

    inputs = set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=hooks,
    )
    serialize_all_input_transformers(inputs_dict=inputs, run_folder=run_folder)
    serialize_all_sequence_inputs(inputs_dict=inputs, run_folder=run_folder)

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
    )

    batch_size = _modify_bs_for_multi_gpu(
        multi_gpu=configs.global_config.multi_gpu,
        batch_size=configs.global_config.batch_size,
    )

    train_sampler = get_train_sampler(
        columns_to_sample=configs.global_config.weighted_sampling_columns,
        train_dataset=train_dataset,
    )

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=configs.global_config.dataloader_workers,
    )

    num_outputs_per_target = set_up_num_outputs_per_target(
        target_transformers=target_labels.label_transformers
    )

    model = get_model(
        inputs_as_dict=inputs,
        global_config=configs.global_config,
        predictor_config=configs.predictor_config,
        num_outputs_per_target=num_outputs_per_target,
    )

    criterions = _get_criterions(
        target_columns=train_dataset.target_columns,
    )

    writer = get_summary_writer(run_folder=run_folder)

    loss_func = _get_loss_callable(
        criterions=criterions,
    )

    optimizer = get_optimizer(
        model=model, loss_callable=loss_func, global_config=configs.global_config
    )

    metrics = get_default_metrics(target_transformers=target_labels.label_transformers)

    experiment = Experiment(
        configs=configs,
        inputs=inputs,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        target_transformers=target_labels.label_transformers,
        num_outputs_per_target=num_outputs_per_target,
        target_columns=train_dataset.target_columns,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        loss_function=loss_func,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    return experiment


def gather_all_ids_from_target_configs(
    target_configs: Sequence[schemas.TargetConfig],
) -> Tuple[str, ...]:
    all_ids = set()
    for config in target_configs:
        cur_label_file = Path(config.label_file)
        cur_ids = gather_ids_from_tabular_file(file_path=cur_label_file)
        all_ids.update(cur_ids)

    return tuple(all_ids)


def get_tabular_target_file_infos(
    target_configs: Iterable[schemas.TargetConfig],
) -> Sequence[TabularFileInfo]:

    tabular_infos = []

    for target_config in target_configs:

        tabular_info = TabularFileInfo(
            file_path=Path(target_config.label_file),
            con_columns=target_config.target_con_columns,
            cat_columns=target_config.target_cat_columns,
            parsing_chunk_size=target_config.label_parsing_chunk_size,
        )
        tabular_infos.append(tabular_info)

    return tabular_infos


def set_up_num_outputs_per_target(
    target_transformers: al_label_transformers,
) -> al_num_outputs_per_target:

    num_outputs_per_target_dict = {}
    for target_column, transformer in target_transformers.items():
        if isinstance(transformer, StandardScaler):
            num_outputs = 1
        else:
            num_outputs = len(transformer.classes_)

            if num_outputs < 2:
                logger.warning(
                    f"Only {num_outputs} unique values found in categorical label "
                    f"column {target_column} (returned by {transformer}). This means "
                    f"that most likely an error will be raised if e.g. using "
                    f"nn.CrossEntropyLoss as it expects an output dimension of >=2."
                )

        num_outputs_per_target_dict[target_column] = num_outputs

    return num_outputs_per_target_dict


def _prepare_run_folder(run_name: str) -> Path:
    run_folder = utils.get_run_folder(run_name=run_name)
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


def _modify_bs_for_multi_gpu(multi_gpu: bool, batch_size: int) -> int:
    if multi_gpu:
        batch_size = torch.cuda.device_count() * batch_size
        logger.info(
            "Batch size set to %d to account for %d GPUs.",
            batch_size,
            torch.cuda.device_count(),
        )

    return batch_size


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
        drop_last=True,
    )

    return train_dloader, valid_dloader


def check_dataset_and_batch_size_compatiblity(
    dataset: datasets.DatasetBase, batch_size: int, name: str = ""
):
    if len(dataset) < batch_size:
        raise ValueError(
            f"{name} dataset size ({len(dataset)}) can not be smaller than "
            f"batch size ({batch_size}). A fix can be increasing {name.lower()} sample "
            f"size or reducing the batch size. If predicting on few unknown samples,"
            f"a solution can be setting the batch size to 1 in the global configuration"
            f"passed to the prediction module. Future work includes making this "
            f"easier to work with."
        )


class GetAttrDelegatedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_model(
    inputs_as_dict: al_input_objects_as_dict,
    predictor_config: schemas.PredictorConfig,
    global_config: schemas.GlobalConfig,
    num_outputs_per_target: al_num_outputs_per_target,
) -> Union[nn.Module, nn.DataParallel]:

    fusion_class = get_fusion_class(fusion_model_type=predictor_config.model_type)
    fusion_kwargs = get_fusion_kwargs_from_cl_args(
        global_config=global_config,
        predictor_config=predictor_config,
        inputs=inputs_as_dict,
        num_outputs_per_target=num_outputs_per_target,
    )
    fusion_model = fusion_class(**fusion_kwargs)
    fusion_model = fusion_model.to(device=global_config.device)

    if global_config.multi_gpu:
        fusion_model = GetAttrDelegatedDataParallel(module=fusion_model)

    return fusion_model


def get_modules_to_fuse_from_inputs(inputs: al_input_objects_as_dict, device: str):
    models = nn.ModuleDict()

    for input_name, inputs_object in inputs.items():

        if input_name.startswith("omics_"):
            cur_omics_model = get_omics_model_from_model_config(
                model_type=inputs_object.input_config.input_type_info.model_type,
                model_config=inputs_object.input_config.model_config,
                data_dimensions=inputs_object.data_dimensions,
                device=device,
            )

            models[input_name] = cur_omics_model

        elif input_name.startswith("tabular_"):

            transformers = inputs_object.labels.label_transformers
            cat_columns = inputs_object.input_config.input_type_info.extra_cat_columns
            con_columns = inputs_object.input_config.input_type_info.extra_con_columns

            unique_tabular_values = get_unique_values_from_transformers(
                transformers=transformers,
                keys_to_use=cat_columns,
            )

            tabular_model = get_tabular_model(
                cat_columns=cat_columns,
                con_columns=con_columns,
                device=device,
                unique_label_values=unique_tabular_values,
            )
            models[input_name] = tabular_model

        elif input_name.startswith("sequence_"):
            sequence_model = get_sequence_model(
                model_type=inputs_object.input_config.input_type_info.model_type,
                model_config=inputs_object.input_config.model_config,
                num_tokens=len(inputs_object.vocab),
                max_length=inputs_object.computed_max_length,
                device=device,
            )
            models[input_name] = sequence_model

    return models


def get_sequence_model(
    model_type: str,
    model_config: BasicTransformerModelConfig,
    num_tokens: int,
    max_length: int,
    device: str,
):
    sequence_model = TransformerModel(
        model_config=model_config,
        num_tokens=num_tokens,
        max_length=max_length,
    ).to(device=device)

    return sequence_model


def get_tabular_model(
    cat_columns: Sequence[str],
    con_columns: Sequence[str],
    device: str,
    unique_label_values: Dict[str, Set[str]],
) -> SimpleTabularModel:
    tabular_model = SimpleTabularModel(
        cat_columns=cat_columns,
        con_columns=con_columns,
        unique_label_values_per_column=unique_label_values,
        device=device,
    )

    return tabular_model


def get_omics_model_from_model_config(
    model_config: al_omics_model_configs,
    data_dimensions: DataDimensions,
    model_type: str,
    device: str,
):

    omics_model_class = get_model_class(model_type=model_type)
    model_init_kwargs = get_omics_model_init_kwargs(
        model_type=model_type,
        model_config=model_config,
        data_dimensions=data_dimensions,
    )
    omics_model = omics_model_class(**model_init_kwargs)

    if model_type == "cnn":
        assert omics_model.data_size_after_conv >= 8

    omics_model = omics_model.to(device=device)

    return omics_model


def get_fusion_class(
    fusion_model_type: str,
) -> Type[nn.Module]:
    if fusion_model_type == "mgmoe":
        return fusion_mgmoe.MGMoEModel
    elif fusion_model_type == "default":
        return fusion.FusionModel
    elif fusion_model_type == "linear":
        return fusion_linear.LinearFusionModel
    raise ValueError(f"Unrecognized fusion model type: {fusion_model_type}.")


def get_fusion_kwargs_from_cl_args(
    global_config: schemas.GlobalConfig,
    predictor_config: schemas.PredictorConfig,
    inputs: al_input_objects_as_dict,
    num_outputs_per_target: al_num_outputs_per_target,
) -> Dict[str, Any]:

    kwargs = {}
    modules_to_fuse = get_modules_to_fuse_from_inputs(
        inputs=inputs, device=global_config.device
    )
    kwargs["modules_to_fuse"] = modules_to_fuse
    kwargs["num_outputs_per_target"] = num_outputs_per_target
    kwargs["model_config"] = predictor_config.model_config

    return kwargs


def _get_criterions(target_columns: al_target_columns) -> al_criterions:
    criterions_dict = {}

    def get_criterion(column_type_):

        if column_type_ == "con":
            return partial(_calc_mse, mse_loss_func=nn.MSELoss())
        elif column_type_ == "cat":
            return nn.CrossEntropyLoss()

    target_columns_gen = data_utils.get_target_columns_generator(
        target_columns=target_columns
    )

    for column_type, column_name in target_columns_gen:
        criterion = get_criterion(column_type_=column_type)
        criterions_dict[column_name] = criterion

    return criterions_dict


def _calc_mse(input, target, mse_loss_func: nn.MSELoss):
    return mse_loss_func(input=input.squeeze(), target=target.squeeze())


def _get_loss_callable(criterions: al_criterions):

    single_task_loss_func = partial(calculate_prediction_losses, criterions=criterions)
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
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def train(experiment: Experiment) -> None:
    e = experiment
    gc = experiment.configs.global_config
    step_hooks = e.hooks.step_func_hooks

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_batch, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        e.model.train()
        e.optimizer.zero_grad()

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

    if gc.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=e.train_loader,
            model=e.model,
            optimizer=e.optimizer,
            output_folder=utils.get_run_folder(run_name=gc.run_name),
        )
        sys.exit(0)

    trainer = configure_trainer(trainer=trainer, experiment=experiment)

    trainer.run(data=e.train_loader, max_epochs=gc.n_epochs)


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

    if configs.global_config.mixing_type is not None:
        logger.debug(
            "Setting up hooks for mixing with %s with α=%.2g.",
            configs.global_config.mixing_type,
            configs.global_config.mixing_alpha,
        )
        mix_hook = get_mix_data_hook(mixing_type=configs.global_config.mixing_type)

        init_kwargs["post_prepare_batch"].append(mix_hook)
        init_kwargs["loss"][0] = hook_mix_loss

    all_targets = get_all_targets(targets_configs=configs.target_configs)
    if len(all_targets) > 1:
        logger.debug(
            "Setting up hook for uncertainty weighted loss for multi task modelling."
        )
        uncertainty_hook = get_uncertainty_loss_hook(
            target_cat_columns=all_targets.cat_targets,
            target_con_columns=all_targets.con_targets,
            device=configs.global_config.device,
        )
        init_kwargs["loss"].append(uncertainty_hook)

    init_kwargs["loss"].append(hook_default_aggregate_losses)

    init_kwargs = add_l1_loss_hook_if_applicable(
        step_function_hooks_init_kwargs=init_kwargs, configs=configs
    )

    return init_kwargs


def add_l1_loss_hook_if_applicable(
    step_function_hooks_init_kwargs: Dict[str, Sequence[Callable]],
    configs: Configs,
) -> Dict[str, Sequence[Callable]]:
    input_l1 = any(
        getattr(input_config.model_config, "l1", None)
        for input_config in configs.input_configs
    )
    preds_l1 = getattr(configs.predictor_config.model_config, "l1", None)
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
        target_columns=experiment.target_columns,
        model=experiment.model,
        device=experiment.configs.global_config.device,
    )

    state_updates = {"batch": batch}

    return state_updates


def prepare_base_batch_default(
    loader_batch: al_dataloader_getitem_batch,
    input_objects: al_input_objects_as_dict,
    target_columns: al_target_columns,
    model: fusion.FusionModel,
    device: str,
):

    inputs, target_labels, train_ids = loader_batch

    inputs_prepared = {}
    for input_name, input_object in input_objects.items():

        if input_name.startswith("omics_"):
            cur_omics = inputs[input_name]
            cur_omics = cur_omics.to(device=device)
            cur_omics = cur_omics.to(dtype=torch.float32)

            inputs_prepared[input_name] = cur_omics

        elif input_name.startswith("tabular_"):

            tabular_source_input: Dict[str, torch.Tensor] = inputs[input_name]
            for name, tensor in tabular_source_input.items():
                tabular_source_input[name] = tensor.to(device=device)

            tabular_input_type_info = input_object.input_config.input_type_info
            cat_columns = tabular_input_type_info.extra_cat_columns
            con_columns = tabular_input_type_info.extra_con_columns
            tabular = get_tabular_inputs(
                extra_cat_columns=cat_columns,
                extra_con_columns=con_columns,
                tabular_model=model.modules_to_fuse[input_name],
                tabular_input=tabular_source_input,
                device=device,
            )
            inputs_prepared[input_name] = tabular

        elif input_name.startswith("sequence_"):
            cur_seq = inputs[input_name]
            cur_seq = cur_seq.to(device=device)
            cur_module = model.modules_to_fuse[input_name]
            cur_embedding = cur_module.embed_tokens(input=cur_seq)
            inputs_prepared[input_name] = cur_embedding

        else:
            raise ValueError(f"Unrecognized input type {input_name}.")

    if target_labels:
        target_labels = model_training_utils.parse_target_labels(
            target_columns=target_columns,
            device=device,
            labels=target_labels,
        )

    batch = Batch(
        inputs=inputs_prepared,
        target_labels=target_labels,
        ids=train_ids,
    )

    return batch


def hook_default_model_forward(
    experiment: "Experiment", batch: "Batch", *args, **kwargs
) -> Dict:

    inputs = batch.inputs
    train_outputs = experiment.model(inputs=inputs)

    state_updates = {"model_outputs": train_outputs}

    return state_updates


def hook_default_optimizer_backward(
    experiment: "Experiment", state: Dict, *args, **kwargs
) -> Dict:

    optimizer_backward_kwargs = get_optimizer_backward_kwargs(
        optimizer_name=experiment.configs.global_config.optimizer
    )

    state["loss"].backward(**optimizer_backward_kwargs)
    experiment.optimizer.step()

    return {}


def hook_default_compute_metrics(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
):

    train_batch_metrics = calculate_batch_metrics(
        target_columns=experiment.target_columns,
        outputs=state["model_outputs"],
        labels=batch.target_labels,
        mode="train",
        metric_record_dict=experiment.metrics,
    )

    train_batch_metrics_w_loss = add_loss_to_metrics(
        target_columns=experiment.target_columns,
        losses=state["per_target_train_losses"],
        metric_dict=train_batch_metrics,
    )

    train_batch_metrics_with_averages = add_multi_task_average_metrics(
        batch_metrics_dict=train_batch_metrics_w_loss,
        target_columns=experiment.target_columns,
        loss=state["loss"].item(),
        performance_average_functions=experiment.metrics["averaging_functions"],
    )

    state_updates = {"metrics": train_batch_metrics_with_averages}

    return state_updates


def hook_default_per_target_loss(
    experiment: "Experiment", batch: "Batch", state: Dict, *args, **kwargs
) -> Dict:

    per_target_train_losses = experiment.loss_function(
        inputs=state["model_outputs"], targets=batch.target_labels
    )

    state_updates = {"per_target_train_losses": per_target_train_losses}

    return state_updates


def hook_default_aggregate_losses(state: Dict, *args, **kwargs) -> Dict:

    train_loss_avg = aggregate_losses(losses_dict=state["per_target_train_losses"])
    state_updates = {"loss": train_loss_avg}

    return state_updates


if __name__ == "__main__":
    main()
