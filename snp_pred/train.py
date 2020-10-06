import argparse
import sys
from dataclasses import dataclass
from functools import partial
from os.path import abspath
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

import configargparse
import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from snp_pred.data_load import data_utils
from snp_pred.data_load import datasets
from snp_pred.data_load.data_augmentation import (
    mixup_data,
    calc_all_mixed_losses,
)
from snp_pred.data_load.data_loading_funcs import get_weighted_random_sampler
from snp_pred.data_load.datasets import al_num_classes
from snp_pred.data_load.label_setup import (
    al_target_columns,
    al_label_transformers,
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
    UncertaintyMultiTaskLoss,
    get_extra_loss_term_functions,
    add_extra_losses,
    get_average_history_filepath,
    get_default_metrics,
)
from snp_pred.train_utils.optimizers import (
    get_optimizer,
    get_base_optimizers_dict,
    get_optimizer_backward_kwargs,
)
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


# Have to have clearly defined stages here
# TODO: Make step_func_hooks a dataclass of stages
@dataclass
class Hooks:
    step_func_hooks: "StepFunctionHookStages"


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
    hook_func: Callable, state: Union[Dict[str, Any], None], *args, **kwargs,
) -> Tuple[Any, Dict[str, Any]]:
    """
    TODO: Add inspection of hook signature
    TODO: Even better, do it when hooks are initialized for the first time.
    """

    if state is None:
        state = {}

    state_updates = hook_func(state=state, *args, **kwargs)

    state = {**state, **state_updates}

    return state_updates, state


def hook_mix_data(config: "Config", state: Dict, *args, **kwargs) -> Dict:

    batch = state["batch"]

    mixed_object = mixup_data(
        inputs=batch.inputs,
        targets=batch.target_labels,
        target_columns=config.target_columns,
        alpha=config.cl_args.mixing_alpha,
        mixing_type=config.cl_args.mixing_type,
    )

    batch_mixed = Batch(
        inputs=mixed_object.inputs,
        target_labels=batch.target_labels,
        extra_inputs=batch.extra_inputs,
        ids=batch.ids,
    )

    state_updates = {"batch": batch_mixed, "mixed_data": mixed_object}

    return state_updates


def hook_mix_loss(config: "Config", state: Dict, *args, **kwargs) -> Dict:

    mixed_losses = calc_all_mixed_losses(
        target_columns=config.target_columns,
        criterions=config.criterions,
        outputs=state["model_outputs"],
        mixed_object=state["mixed_data"],
    )

    state_updates = {"train_losses": mixed_losses}

    return state_updates


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

    mixed_losses = config.loss_function(
        inputs=state["model_outputs"], targets=batch.target_labels
    )

    state_updates = {"train_losses": mixed_losses}

    return state_updates


def hook_final_loss(config: "Config", state: Dict, *args, **kwargs) -> Dict:
    """
    TODO: Separate the extra loss term functions into own hooks?
    """

    extra_loss_functions = get_extra_loss_term_functions(
        model=config.model, l1_weight=config.cl_args.l1
    )

    train_loss_final = add_extra_losses(
        total_loss=state["loss_average"], extra_loss_functions=extra_loss_functions
    )

    state_updates = {"loss_final": train_loss_final}

    return state_updates


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
    data_width: int
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    hooks: Union[Hooks, None]


def main(cl_args: argparse.Namespace, hooks: Union[Hooks, None] = None) -> None:
    run_folder = _prepare_run_folder(run_name=cl_args.run_name)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args=cl_args)

    cl_args.target_width = train_dataset[0][0].shape[2]

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

    # TODO: Actually add support for mixup in this function
    loss_func = _get_loss_callable(
        target_columns=train_dataset.target_columns,
        criterions=criterions,
        device=cl_args.device,
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
        data_width=train_dataset.data_width,
        writer=writer,
        metrics=metrics,
        hooks=hooks,
    )

    _log_num_params(model=model)

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
    num_workers: int = 8,
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


def _get_loss_callable(
    target_columns: al_target_columns, criterions: al_criterions, device: str,
):
    num_tasks = len(target_columns["con"] + target_columns["cat"])
    if num_tasks > 1:
        multi_task_loss_module = UncertaintyMultiTaskLoss(
            target_columns=target_columns, criterions=criterions, device=device
        )
        return multi_task_loss_module
    elif num_tasks == 1:
        single_task_loss_func = partial(
            calculate_prediction_losses, criterions=criterions
        )
        return single_task_loss_func


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_num_params(model: nn.Module) -> None:
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


@dataclass
class Batch:
    inputs: torch.Tensor
    target_labels: Dict[str, torch.Tensor]
    extra_inputs: Union[Dict[str, torch.Tensor], None]
    ids: List[str]


def _prepare_batch(
    loader_batch: Tuple[torch.Tensor, al_training_labels_batch, List[str]],
    config,
    cl_args,
) -> Batch:

    train_seqs, labels, train_ids = loader_batch
    train_seqs = train_seqs.to(device=cl_args.device)
    train_seqs = train_seqs.to(dtype=torch.float32)

    target_labels = model_training_utils.parse_target_labels(
        target_columns=config.target_columns,
        device=cl_args.device,
        labels=labels["target_labels"],
    )

    # TODO: We will have to mix extra inputs as well
    # TODO: Here we have extra_inputs hook
    extra_inputs = get_extra_inputs(
        cl_args=cl_args, model=config.model, labels=labels["extra_labels"]
    )

    batch = Batch(
        inputs=train_seqs,
        target_labels=target_labels,
        extra_inputs=extra_inputs,
        ids=train_ids,
    )

    return batch


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

        train_outputs = c.model(x=batch.inputs, extra_inputs=batch.extra_inputs)
        state["model_outputs"] = train_outputs

        per_target_loss_stage = step_hooks.per_target_loss
        state = call_hooks_stage_iterable(
            hook_iterable=per_target_loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        train_loss_avg = aggregate_losses(losses_dict=state["train_losses"])
        state["loss_average"] = train_loss_avg

        final_loss_stage = step_hooks.final_loss
        state = call_hooks_stage_iterable(
            hook_iterable=final_loss_stage,
            common_kwargs={"config": config, "batch": batch},
            state=state,
        )

        state["loss_final"].backward(**optimizer_backward_kwargs)
        c.optimizer.step()

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


def _get_train_argument_parser() -> configargparse.ArgumentParser:

    parser_ = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser_.add_argument(
        "--config_file",
        is_config_file=True,
        required=False,
        help="path to .yaml config file if using one",
    )

    parser_.add_argument(
        "--n_epochs", type=int, default=5, help="number of epochs of training"
    )
    parser_.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )

    parser_.add_argument(
        "--dataloader_workers",
        type=int,
        default=8,
        help="Number of workers for training and validation dataloaders.",
    )
    parser_.add_argument(
        "--lr", type=float, default=1e-3, help="Base learning rate for optimizer."
    )

    parser_.add_argument(
        "--find_lr",
        action="store_true",
        help="Whether to perform a range test of different learning rates, with "
        "the lower limit being what is passed in for the --lr flag. "
        "Produces a plot and exits with status 0 before training if this flag "
        "is active.",
    )

    parser_.add_argument(
        "--lr_schedule",
        type=str,
        default="same",
        choices=["cycle", "plateau", "same", "cosine"],
        help="Whether to use cyclical or reduce on plateau learning rate schedule. "
        "Otherwise keeps same learning rate.",
    )

    # TODO: Change this to patience steps, so it is configurable
    parser_.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Whether to terminate training early if performance stops improving.",
    )

    parser_.add_argument(
        "--warmup_steps", type=str, default=0, help="How many steps to use in warmup."
    )

    parser_.add_argument(
        "--lr_lb",
        type=float,
        default=0.0,
        help="Lower bound for learning rate when using LR scheduling.",
    )

    parser_.add_argument(
        "--optimizer",
        type=str,
        choices=_get_optimizer_cl_arg_choices(),
        default="adamw",
        help="Whether to use AdamW or SGDM optimizer.",
    )

    parser_.add_argument(
        "--b1",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )

    parser_.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of second order momentum of gradient",
    )
    parser_.add_argument("--wd", type=float, default=0.00, help="Weight decay.")

    parser_.add_argument(
        "--l1", type=float, default=0.00, help="L1 regularization for chosen layer."
    )

    parser_.add_argument(
        "--fc_repr_dim",
        type=int,
        default=512,
        help="dimensionality of first fc layer in the network, this is the last shared"
        "when running multi task training, first fc layer after convolutions when"
        "running cnn model, and first fc layer when running mlp",
    )

    parser_.add_argument(
        "--fc_task_dim",
        type=int,
        default=128,
        help="dimensionality of (a) specific task branches in multi task setting, (b)"
        "successive fc layers in cnn model after the first fc after the "
        "convolutions and (c) successive fc layers in mlp after first fc layer",
    )

    parser_.add_argument(
        "--mg_num_experts", type=int, default=8, help="Number of experts to use."
    )

    parser_.add_argument(
        "--split_mlp_num_splits",
        type=int,
        default=50,
        help="Number of splits in split MLP layer.",
    )

    parser_.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "mlp", "mlp-split", "mlp-fully-split", "mlp-mgmoe", "linear"],
        help="whether to use a convolutional neural network (cnn) or multilayer "
        "perceptron (mlp)",
    )

    parser_.add_argument(
        "--kernel_width",
        type=int,
        default=12,
        help="base width of the conv kernels used.",
    )

    parser_.add_argument(
        "--down_stride",
        type=int,
        default=4,
        help="down stride to use common over the network.",
    )

    parser_.add_argument(
        "--dilation_factor",
        type=int,
        default=1,
        help="factor to dilate convolutions by in each successive block",
    )

    parser_.add_argument(
        "--first_kernel_expansion",
        type=int,
        default=1,
        help="factor by which to expand the first kernel in the network",
    )

    parser_.add_argument(
        "--first_stride_expansion",
        type=int,
        default=1,
        help="factor by which to expand the first stride in the network",
    )

    parser_.add_argument(
        "--first_channel_expansion",
        type=int,
        default=1,
        help="factor by which to expand the first stride in the network",
    )

    parser_.add_argument(
        "--channel_exp_base",
        type=int,
        default=5,
        help="Exponential base for channels in first layer (i.e. default is 2**5)",
    )

    # TODO: Better help message.
    parser_.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Number of layers in models where it applies.",
    )

    parser_.add_argument(
        "--rb_do", type=float, default=0.0, help="Dropout in residual blocks."
    )

    parser_.add_argument(
        "--fc_do",
        type=float,
        default=0.0,
        help="Dropout before last fully connected layer.",
    )

    parser_.add_argument(
        "--sa",
        action="store_true",
        help="Whether to add self attention to the network.",
    )

    parser_.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Total width of input sequence after padding.",
    )
    parser_.add_argument(
        "--data_source",
        type=str,
        required=True,
        help="Data source to load inputs from. Can either be (a) a folder in which"
        "files will be gathered from the folder recursively and (b) a simple text"
        "file with each line having a path for a sample array.",
    )

    parser_.add_argument(
        "--valid_size",
        type=float,
        default=0.05,
        help="Size if the validaton set, if float then uses a percentage. If int, "
        "then raw counts.",
    )

    parser_.add_argument(
        "--weighted_sampling_column",
        type=str,
        default=None,
        nargs="*",
        help="Target column to apply weighted sampling on.",
    )

    parser_.add_argument(
        "--na_augment_perc",
        default=0.0,
        type=float,
        help="Percentage of array to make missing when using na_augmentation.",
    )

    parser_.add_argument(
        "--na_augment_prob",
        default=0.5,
        type=float,
        help="Probability of applying an na_augmentation of percentage as given in "
        "--na_augment_perc.",
    )

    parser_.add_argument(
        "--mixing_alpha",
        default=0.0,
        type=float,
        help="Alpha parameter used for mixing (higher means more mixing).",
    )

    parser_.add_argument(
        "--mixing_type",
        default=None,
        type=str,
        choices=["mixup", "cutmix-block", "cutmix-uniform"],
        help="Type of mixing to apply when using mixup and similar approaches.",
    )

    parser_.add_argument(
        "--label_file", type=str, required=True, help="Which file to load labels from."
    )

    parser_.add_argument(
        "--target_con_columns",
        nargs="*",
        default=[],
        help="Continuous target columns in label file.",
    )
    parser_.add_argument(
        "--target_cat_columns",
        nargs="*",
        default=[],
        help="Categorical target columns in label file.",
    )

    parser_.add_argument(
        "--extra_cat_columns",
        type=str,
        nargs="+",
        default=[],
        help="What columns of categorical variables to add to fully connected layer at "
        "end of model.",
    )

    parser_.add_argument(
        "--extra_con_columns",
        type=str,
        nargs="+",
        default=[],
        help="What columns of continuous variables to add to fully connected layer at "
        "end of model.",
    )

    parser_.add_argument(
        "--snp_file",
        type=str,
        default="infer",
        help="File to load SNPs from (.snp format).",
    )

    parser_.add_argument(
        "--memory_dataset",
        action="store_true",
        help="Whether to load all sample into memory during " "training.",
    )

    parser_.add_argument(
        "--sample_interval",
        type=int,
        default=None,
        help="Epoch interval to sample generated seqs.",
    )
    parser_.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Epoch to checkpoint model.",
    )
    parser_.add_argument(
        "--run_name",
        required=True,
        type=str,
        help="Name of the current run, specifying will save " "run info and models.",
    )

    parser_.add_argument(
        "--gpu_num", type=str, default="0", help="Which GPU to run (according to CUDA)."
    )

    parser_.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Whether to run the training on " "multiple GPUs for the current node.",
    )

    parser_.add_argument(
        "--get_acts", action="store_true", help="Whether to generate activation maps."
    )

    parser_.add_argument(
        "--act_classes",
        default=None,
        nargs="+",
        help="Classes to use for activation maps.",
    )

    parser_.add_argument(
        "--max_acts_per_class",
        default=None,
        type=int,
        help="Maximum number of samples per class to gather for activation analysis. "
        "Good to use when modelling on imbalanced data.",
    )

    parser_.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (w. breakpoint).",
    )

    parser_.add_argument(
        "--no_pbar",
        action="store_true",
        help="Whether to run in debug mode (w. breakpoint).",
    )

    parser_.add_argument(
        "--custom_lib",
        type=str,
        default=None,
        help="Path to custom library if using one.",
    )

    parser_.add_argument(
        "--plot_skip_steps",
        type=int,
        default=200,
        help="How many iterations to skip in in plots.",
    )
    return parser_


def _get_optimizer_cl_arg_choices():
    """
    Currently just going to hardcode the main default optimizers. Later we can do
    something fancy with inspect and issubclass of Optimizer to get names of all
    PyTorch built-in optimizers.
    """
    base_optimizer_dict = get_base_optimizers_dict()
    default = list(base_optimizer_dict.keys())
    external = _get_custom_opt_names()
    return default + external


def _get_custom_opt_names():
    # import here to keep separated from main codebase
    from torch_optimizer import _NAME_OPTIM_MAP as CUSTOM_OPT_NAME_MAP

    custom_optim_list = list(CUSTOM_OPT_NAME_MAP.keys())
    custom_optim_list = [i for i in custom_optim_list if i != "lookahead"]
    return custom_optim_list


def _modify_train_arguments(cl_args: argparse.Namespace) -> argparse.Namespace:
    if cl_args.valid_size > 1.0:
        cl_args.valid_size = int(cl_args.valid_size)

    cl_args.device = "cuda:" + cl_args.gpu_num if torch.cuda.is_available() else "cpu"

    # to make sure importlib gets absolute paths
    if cl_args.custom_lib is not None:
        cl_args.custom_lib = abspath(cl_args.custom_lib)

    # benchmark breaks if we run it with multiple GPUs
    if not cl_args.multi_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        logger.debug("Setting device to cuda:0 since running with multiple GPUs.")
        cl_args.device = "cuda:0"

    return cl_args


@dataclass
class StepFunctionHookStages:
    prepare_batch: List[Callable]
    per_target_loss: List[Callable]
    final_loss: List[Callable]


def _get_step_func_hooks(cl_args: argparse.Namespace):
    """
    TODO: Add validation, inspect that outputs have correct names.
    """

    init_kwargs = {
        "prepare_batch": [hook_default_prepare_batch],
        "per_target_loss": [hook_default_loss],
        "final_loss": [hook_final_loss],
    }

    if cl_args.mixing_type is not None:
        logger.debug("Setting up hooks for mixing.")
        init_kwargs["prepare_batch"].append(hook_mix_data)
        init_kwargs["per_target_loss"] = [hook_mix_loss]

    step_func_hooks = StepFunctionHookStages(**init_kwargs)

    return step_func_hooks


def _get_hooks(cl_args_: argparse.Namespace):
    step_func_hooks = _get_step_func_hooks(cl_args=cl_args_)
    hooks_object = Hooks(step_func_hooks=step_func_hooks)

    return hooks_object


if __name__ == "__main__":

    parser = _get_train_argument_parser()
    cur_cl_args = parser.parse_args()
    cur_cl_args = _modify_train_arguments(cl_args=cur_cl_args)

    utils.configure_root_logger(run_name=cur_cl_args.run_name)

    hooks = _get_hooks(cl_args_=cur_cl_args)

    main(cl_args=cur_cl_args, hooks=hooks)
