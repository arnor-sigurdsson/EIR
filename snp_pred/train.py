import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Union,
    Tuple,
    List,
    Dict,
    overload,
    TYPE_CHECKING,
    Callable,
    Any,
    Iterable,
)

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from snp_pred.configuration import get_default_cl_args
from snp_pred.data_load import data_utils
from snp_pred.data_load import datasets
from snp_pred.data_load.data_augmentation import hook_mix_loss, get_mix_data_hook
from snp_pred.data_load.data_loading_funcs import get_weighted_random_sampler
from snp_pred.data_load.data_utils import Batch
from snp_pred.data_load.datasets import al_num_classes
from snp_pred.data_load.label_setup import (
    al_target_columns,
    al_label_transformers,
    al_all_column_ops,
)
from snp_pred.models import model_training_utils
from snp_pred.models.extra_inputs_module import (
    set_up_and_save_embeddings_dict,
    get_extra_inputs,
    al_emb_lookup_dict,
)
from snp_pred.models.model_training_utils import run_lr_find
from snp_pred.models.models import al_models
from snp_pred.models.models import get_model_class
from snp_pred.train_utils import utils
from snp_pred.train_utils.metrics import (
    calculate_batch_metrics,
    calculate_prediction_losses,
    aggregate_losses,
    add_multi_task_average_metrics,
    get_average_history_filepath,
    get_default_metrics,
    hook_add_l1_loss,
    get_uncertainty_loss_hook,
)
from snp_pred.train_utils.optimizers import (
    get_optimizer,
    get_optimizer_backward_kwargs,
)
from snp_pred.train_utils.train_handlers import HandlerConfig
from snp_pred.train_utils.train_handlers import configure_trainer

if TYPE_CHECKING:
    from snp_pred.train_utils.metrics import (
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
al_dataloader_getitem_batch = Tuple[torch.Tensor, al_training_labels_batch, List[str]]

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass(frozen=True)
class Config:
    """
    The idea of this class is to keep track of objects that need to be used
    in multiple contexts in different parts of the code (e.g. the train
    dataloader is used to load samples during training, but also as background
    for SHAP activation calculations).
    """

    cl_args: argparse.Namespace
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    valid_dataset: torch.utils.data.Dataset
    model: Union[al_models, nn.DataParallel]
    optimizer: Optimizer
    criterions: al_criterions
    loss_function: Callable
    labels_dict: Dict
    target_transformers: al_label_transformers
    target_columns: al_target_columns
    data_dimension: "DataDimension"
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union["Hooks", None]


def get_default_config(
    cl_args: argparse.Namespace, hooks: Union["Hooks", None] = None
) -> "Config":
    run_folder = _prepare_run_folder(run_name=cl_args.run_name)

    train_dataset, valid_dataset = datasets.set_up_datasets(
        cl_args=cl_args, custom_label_ops=hooks.custom_column_label_parsing_ops
    )

    data_dimensions = _get_data_dimensions(
        dataset=train_dataset, target_width=cl_args.target_width
    )

    cl_args.target_width = data_dimensions.width

    batch_size = _modify_bs_for_multi_gpu(
        multi_gpu=cl_args.multi_gpu, batch_size=cl_args.batch_size
    )

    train_sampler = get_train_sampler(
        columns_to_sample=cl_args.weighted_sampling_column, train_dataset=train_dataset
    )

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=cl_args.dataloader_workers,
    )

    embedding_dict = set_up_and_save_embeddings_dict(
        embedding_columns=cl_args.extra_cat_columns,
        labels_dict=train_dataset.labels_dict,
        run_folder=run_folder,
    )

    model = get_model(
        cl_args=cl_args,
        num_classes=train_dataset.num_classes,
        embedding_dict=embedding_dict,
    )

    criterions = _get_criterions(
        target_columns=train_dataset.target_columns, model_type=cl_args.model_type
    )

    writer = get_summary_writer(run_folder=run_folder)

    loss_func = _get_loss_callable(
        criterions=criterions,
    )

    optimizer = get_optimizer(model=model, loss_callable=loss_func, cl_args=cl_args)

    metrics = get_default_metrics(target_transformers=train_dataset.target_transformers)

    config = Config(
        cl_args=cl_args,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        loss_function=loss_func,
        labels_dict=train_dataset.labels_dict,
        target_transformers=train_dataset.target_transformers,
        target_columns=train_dataset.target_columns,
        data_dimension=data_dimensions,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    return config


@dataclass
class DataDimension:
    channels: int
    height: int
    width: int


def _get_data_dimensions(
    dataset: torch.utils.data.Dataset, target_width: Union[int, None]
) -> DataDimension:
    sample, *_ = dataset[0]
    channels, height, width = sample.shape

    if target_width is not None:
        width = target_width

    return DataDimension(channels=channels, height=height, width=width)


def main(cl_args: argparse.Namespace, config: Config) -> None:

    _log_model(model=config.model, l1_weight=cl_args.l1)

    if cl_args.debug:
        breakpoint()

    train(config=config)


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


@overload
def get_train_sampler(
    columns_to_sample: None, train_dataset: datasets.ArrayDatasetBase
) -> None:
    ...


@overload
def get_train_sampler(
    columns_to_sample: List[str], train_dataset: datasets.ArrayDatasetBase
) -> WeightedRandomSampler:
    ...


def get_train_sampler(columns_to_sample, train_dataset):
    if columns_to_sample is None:
        return None

    loaded_target_columns = (
        train_dataset.target_columns["con"] + train_dataset.target_columns["cat"]
    )

    is_sample_column_loaded = set(columns_to_sample).issubset(
        set(loaded_target_columns)
    )
    is_sample_all_cols = columns_to_sample == ["all"]

    if not is_sample_column_loaded and not is_sample_all_cols:
        raise ValueError(
            "Weighted sampling from non-loaded columns not supported yet "
            f"(could not find {columns_to_sample})."
        )

    if is_sample_all_cols:
        columns_to_sample = train_dataset.target_columns["cat"]

    train_sampler = get_weighted_random_sampler(
        train_dataset=train_dataset, target_columns=columns_to_sample
    )
    return train_sampler


def get_dataloaders(
    train_dataset: datasets.ArrayDatasetBase,
    train_sampler: Union[None, WeightedRandomSampler],
    valid_dataset: datasets.ArrayDatasetBase,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple:

    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
        pin_memory=False,
    )

    valid_dloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_dloader, valid_dloader


class GetAttrDelegatedDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_model(
    cl_args: argparse.Namespace,
    num_classes: al_num_classes,
    embedding_dict: Union[al_emb_lookup_dict, None],
) -> Union[nn.Module, nn.DataParallel]:

    model_class = get_model_class(model_type=cl_args.model_type)
    model = model_class(
        cl_args=cl_args,
        num_classes=num_classes,
        embeddings_dict=embedding_dict,
        extra_continuous_inputs_columns=cl_args.extra_con_columns,
    )

    if cl_args.model_type == "cnn":
        assert model.data_size_after_conv >= 8

    if cl_args.multi_gpu:
        model = GetAttrDelegatedDataParallel(module=model)

    if model_class == "linear":
        _check_linear_model_columns(cl_args=cl_args)

    model = model.to(device=cl_args.device)

    return model


def _check_linear_model_columns(cl_args: argparse.Namespace) -> None:
    num_label_cols = len(cl_args.target_cat_columns + cl_args.target_con_columns)
    if num_label_cols != 1:
        raise NotImplementedError(
            "Linear model only supports one target column currently."
        )

    num_extra_cols = len(cl_args.extra_cat_columns + cl_args.extra_con_columns)
    if num_extra_cols != 0:
        raise NotImplementedError(
            "Extra columns not supported for linear model currently."
        )


def _get_criterions(
    target_columns: al_target_columns, model_type: str
) -> al_criterions:
    criterions_dict = {}

    def calc_bce(input, target):
        # note we use input and not e.g. input_ here because torch uses name "input"
        # in loss functions for compatibility
        bce_loss_func = nn.BCELoss()
        return bce_loss_func(input[:, 1], target.to(dtype=torch.float))

    def get_criterion(column_type_):

        if model_type == "linear":
            if column_type_ == "cat":
                return calc_bce
            else:
                return nn.MSELoss(reduction="mean")

        if column_type_ == "con":
            return nn.MSELoss()
        elif column_type_ == "cat":
            return nn.CrossEntropyLoss()

    target_columns_gen = data_utils.get_target_columns_generator(
        target_columns=target_columns
    )

    for column_type, column_name in target_columns_gen:
        criterion = get_criterion(column_type_=column_type)
        criterions_dict[column_name] = criterion

    return criterions_dict


def _get_loss_callable(criterions: al_criterions):

    single_task_loss_func = partial(calculate_prediction_losses, criterions=criterions)
    return single_task_loss_func


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_model(model: nn.Module, l1_weight: float) -> None:
    """
    TODO: Add summary of parameters
    TODO: Add verbosity option
    """
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(
        "Penalizing weights of shape %s with L1 loss with weight %f.",
        model.l1_penalized_weights.shape,
        l1_weight,
    )

    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def train(config: Config) -> None:
    c = config
    cl_args = config.cl_args
    step_hooks = c.hooks.step_func_hooks

    optimizer_backward_kwargs = get_optimizer_backward_kwargs(
        optimizer_name=cl_args.optimizer
    )

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_batch, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        c.model.train()

        loaded_inputs_stage = step_hooks.prepare_batch
        state = call_hooks_stage_iterable(
            hook_iterable=loaded_inputs_stage,
            common_kwargs={"config": config, "loader_batch": loader_batch},
            state=None,
        )
        batch = state["batch"]

        c.optimizer.zero_grad()

        # TODO: Can be a hook (model forward)
        train_outputs = c.model(x=batch.inputs, extra_inputs=batch.extra_inputs)
        state["model_outputs"] = train_outputs

        per_target_loss_stage = step_hooks.per_target_loss
        state = call_hooks_stage_iterable(
            hook_iterable=per_target_loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        # TODO: Can be added to previous hook step
        train_loss_avg = aggregate_losses(losses_dict=state["train_losses"])
        state["loss"] = train_loss_avg

        final_loss_stage = step_hooks.final_loss
        state = call_hooks_stage_iterable(
            hook_iterable=final_loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        # TODO: Can be a hook (optimizer backward)
        state["loss"].backward(**optimizer_backward_kwargs)
        c.optimizer.step()

        # TODO: Can be a hook (metrics finalizer)
        train_batch_metrics = calculate_batch_metrics(
            target_columns=c.target_columns,
            losses=state["train_losses"],
            outputs=train_outputs,
            labels=batch.target_labels,
            mode="train",
            metric_record_dict=c.metrics,
        )

        train_batch_metrics_with_averages = add_multi_task_average_metrics(
            batch_metrics_dict=train_batch_metrics,
            target_columns=c.target_columns,
            loss=train_loss_avg.item(),
            performance_average_functions=c.metrics["averaging_functions"],
        )

        return train_batch_metrics_with_averages

    trainer = Engine(process_function=step)

    if cl_args.find_lr:
        logger.info("Running LR find and exiting.")
        run_lr_find(
            trainer_engine=trainer,
            train_dataloader=c.train_loader,
            model=c.model,
            optimizer=c.optimizer,
            output_folder=utils.get_run_folder(run_name=cl_args.run_name),
        )
        sys.exit(0)

    trainer = configure_trainer(trainer=trainer, config=config)

    trainer.run(data=c.train_loader, max_epochs=cl_args.n_epochs)


def get_default_hooks(cl_args_: argparse.Namespace):
    step_func_hooks = _get_step_func_hooks(cl_args=cl_args_)
    hooks_object = Hooks(step_func_hooks=step_func_hooks)

    return hooks_object


@dataclass
class Hooks:
    al_handler_attachers = Iterable[Callable[[Engine, HandlerConfig], Engine]]

    step_func_hooks: "StepFunctionHookStages"
    custom_column_label_parsing_ops: Union[None, al_all_column_ops] = None
    custom_handler_attachers: Union[None, al_handler_attachers] = None


def _get_step_func_hooks(cl_args: argparse.Namespace):
    """
    TODO: Add validation, inspect that outputs have correct names.
    TODO: Refactor, split into smaller functions e.g. for L1, mixing and uncertainty.
    """

    init_kwargs = {
        "prepare_batch": [hook_default_prepare_batch],
        "per_target_loss": [hook_default_loss],
        "final_loss": [],
    }

    if cl_args.l1 is not None:
        init_kwargs["final_loss"].append(hook_add_l1_loss)

    if cl_args.mixing_type is not None:
        logger.debug(
            "Setting up hooks for mixing with %s with α=%.2g.",
            cl_args.mixing_type,
            cl_args.mixing_alpha,
        )
        mix_hook = get_mix_data_hook(mixing_type=cl_args.mixing_type)

        init_kwargs["prepare_batch"].append(mix_hook)
        init_kwargs["per_target_loss"] = [hook_mix_loss]

    if len(cl_args.target_con_columns + cl_args.target_cat_columns) > 1:
        logger.debug(
            "Setting up hook for uncertainty weighted loss for multi task modelling."
        )
        uncertainty_hook = get_uncertainty_loss_hook(
            target_cat_columns=cl_args.target_cat_columns,
            target_con_columns=cl_args.target_con_columns,
            device=cl_args.device,
        )
        init_kwargs["per_target_loss"].append(uncertainty_hook)

    step_func_hooks = StepFunctionHookStages(**init_kwargs)

    return step_func_hooks


@dataclass
class StepFunctionHookStages:
    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    prepare_batch: al_hooks
    per_target_loss: al_hooks
    final_loss: al_hooks


def call_hooks_stage_iterable(
    hook_iterable: Iterable[Callable],
    common_kwargs: Dict,
    state: Union[None, Dict[str, Any]],
):
    for hook in hook_iterable:
        _, state = state_registered_hook_call(
            hook_func=hook, **common_kwargs, state=state
        )

    return state


def state_registered_hook_call(
    hook_func: Callable,
    state: Union[Dict[str, Any], None],
    *args,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:

    if state is None:
        state = {}

    state_updates = hook_func(state=state, *args, **kwargs)

    state = {**state, **state_updates}

    return state_updates, state


def hook_default_model_inputs(state, batch: "Batch", *args, **kwargs) -> Dict:
    state_updates = {"model_inputs": batch.inputs, "extra_inputs": batch.extra_inputs}
    return state_updates


def hook_default_prepare_batch(
    config: "Config",
    loader_batch: al_dataloader_getitem_batch,
    state: Dict,
    *args,
    **kwargs,
) -> Dict:

    cl_args = config.cl_args

    train_seqs, labels, train_ids = loader_batch
    train_seqs = train_seqs.to(device=cl_args.device)
    train_seqs = train_seqs.to(dtype=torch.float32)

    target_labels = model_training_utils.parse_target_labels(
        target_columns=config.target_columns,
        device=cl_args.device,
        labels=labels["target_labels"],
    )

    extra_inputs = get_extra_inputs(
        cl_args=cl_args, model=config.model, labels=labels["extra_labels"]
    )

    batch = Batch(
        inputs=train_seqs,
        target_labels=target_labels,
        extra_inputs=extra_inputs,
        ids=train_ids,
    )

    state_updates = {"batch": batch}

    return state_updates


def hook_default_loss(
    config: "Config", batch: "Batch", state: Dict, *args, **kwargs
) -> Dict:

    train_losses = config.loss_function(
        inputs=state["model_outputs"], targets=batch.target_labels
    )

    state_updates = {"train_losses": train_losses}

    return state_updates


if __name__ == "__main__":

    default_cl_args = get_default_cl_args()
    utils.configure_root_logger(run_name=default_cl_args.run_name)

    default_hooks = get_default_hooks(cl_args_=default_cl_args)
    default_config = get_default_config(cl_args=default_cl_args, hooks=default_hooks)

    main(cl_args=default_cl_args, config=default_config)
