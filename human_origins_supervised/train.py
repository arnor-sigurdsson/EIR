import copy
import argparse
import sys
from dataclasses import dataclass
from functools import partial
from os.path import abspath
from pathlib import Path
from sys import platform
from typing import Union, Tuple, List, Dict, overload, TYPE_CHECKING, Callable

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

from human_origins_supervised.data_load import data_utils
from human_origins_supervised.data_load import datasets
from human_origins_supervised.data_load.data_loading_funcs import (
    get_weighted_random_sampler,
)
from human_origins_supervised.data_load.datasets import al_num_classes
from human_origins_supervised.data_load.label_setup import (
    al_target_columns,
    al_label_transformers,
)
from human_origins_supervised.models import model_utils
from human_origins_supervised.models.extra_inputs_module import (
    set_up_and_save_embeddings_dict,
    get_extra_inputs,
    al_emb_lookup_dict,
)
from human_origins_supervised.models.model_utils import run_lr_find
from human_origins_supervised.models.models import get_model_class, al_models
from human_origins_supervised.train_utils import utils
from human_origins_supervised.train_utils.metrics import (
    calculate_batch_metrics,
    calculate_prediction_losses,
    aggregate_losses,
    add_multi_task_average_metrics,
    UncertaintyMultiTaskLoss,
    get_extra_loss_term_functions,
    add_extra_losses,
    get_average_history_filepath,
    calc_mcc,
    calc_roc_auc_ovr,
    calc_acc,
    calc_rmse,
    calc_pcc,
    calc_r2,
    calc_average_precision_ovr,
    MetricRecord,
)
from human_origins_supervised.train_utils.optimizers import (
    get_optimizer,
    get_base_optimizers_dict,
    get_optimizer_backward_kwargs,
)
from human_origins_supervised.train_utils.train_handlers import configure_trainer

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.metrics import (
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
al_averaging_functions_dict = Dict[
    str, Callable[["al_step_metric_dict", str, str], float]
]

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class CustomHooks:
    metrics: Dict


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
    model: al_models
    optimizer: Optimizer
    criterions: al_criterions
    loss_function: Callable
    labels_dict: Dict
    target_transformers: al_label_transformers
    target_columns: al_target_columns
    data_width: int
    writer: SummaryWriter
    metrics: "al_metric_record_dict"
    custom_hooks: Union[CustomHooks, None]


def main(
    cl_args: argparse.Namespace, custom_hooks: Union[CustomHooks, None] = None
) -> None:
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
        target_columns=train_dataset.target_columns,
        criterions=criterions,
        device=cl_args.device,
        mixup=True,
    )

    optimizer = get_optimizer(model=model, loss_callable=loss_func, cl_args=cl_args)

    metrics = _get_default_metrics(
        target_transformers=train_dataset.target_transformers
    )

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
        custom_hooks=custom_hooks,
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
) -> Tuple:

    # Currently as bug with OSX in torch 1.3.0:
    # https://github.com/pytorch/pytorch/issues/2125
    nw = 0 if platform == "darwin" else 8
    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=nw,
        pin_memory=False,
    )

    valid_dloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=False,
    )

    return train_dloader, valid_dloader


class MyDataParallel(nn.DataParallel):
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
        model = MyDataParallel(module=model)
        breakpoint()

    if model_class == "linear":
        _check_linear_model_columns(cl_args=cl_args)

    model = model.to(device=cl_args.device)

    return model


def _check_linear_model_columns(cl_args: argparse.Namespace) -> None:
    if (
        len(
            cl_args.target_cat_columns
            + cl_args.target_con_columns
            + cl_args.extra_cat_columns
            + cl_args.extra_con_columns
        )
        != 1
    ):
        raise NotImplementedError(
            "Linear model only supports one target column currently."
        )

    if len(cl_args.extra_cat_columns + cl_args.extra_con_columns) != 0:
        raise NotImplementedError(
            "Extra columns not supported for linear model currently."
        )


def _get_criterions(
    target_columns: al_target_columns, model_type: str
) -> al_criterions:
    criterions_dict = {}

    # def calc_bce(input, target):
    #     # note we use input and not e.g. input_ here because torch uses name "input"
    #     # in loss functions for compatibility
    #     bce_loss_func = nn.BCELoss()
    #     return bce_loss_func(input[:, 1], target.to(dtype=torch.float))

    def get_criterion(column_type_):

        if model_type == "linear":
            if column_type_ == "cat":
                return nn.CrossEntropyLoss()
                # return calc_bce
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
    target_columns: al_target_columns,
    criterions: al_criterions,
    device: str,
    mixup: bool = False,
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


def _get_default_metrics(
    target_transformers: al_label_transformers,
) -> "al_metric_record_dict":
    mcc = MetricRecord(name="mcc", function=calc_mcc)
    acc = MetricRecord(name="acc", function=calc_acc)
    rmse = MetricRecord(
        name="rmse",
        function=partial(calc_rmse, target_transformers=target_transformers),
        minimize_goal=True,
    )

    default_metrics = {
        "cat": (mcc, acc),
        "con": (rmse,),
        "averaging_functions": {"con": "loss", "cat": "acc"},
    }

    # TODO: Remove and use default metrics, currently temporary for testing
    roc_auc_macro = MetricRecord(
        name="roc-auc-macro", function=calc_roc_auc_ovr, only_val=True
    )
    ap_macro = MetricRecord(
        name="ap-macro", function=calc_average_precision_ovr, only_val=True
    )
    r2 = MetricRecord(name="r2", function=calc_r2, only_val=True)
    pcc = MetricRecord(name="pcc", function=calc_pcc, only_val=True)

    averaging_functions = _get_default_performance_averaging_functions()
    default_metrics = {
        "cat": (mcc, acc, roc_auc_macro, ap_macro),
        "con": (rmse, r2, pcc),
        "averaging_functions": averaging_functions,
    }
    return default_metrics


def _get_default_performance_averaging_functions() -> al_averaging_functions_dict:
    def _calc_cat_averaging_value(
        metric_dict: "al_step_metric_dict", column_name: str, metric_name: str
    ) -> float:
        return metric_dict[column_name].get(f"{column_name}_{metric_name}", 0)

    def _calc_con_averaging_value(
        metric_dict: "al_step_metric_dict", column_name: str, metric_name: str
    ) -> float:
        return 1 - metric_dict[column_name][f"{column_name}_{metric_name}"]

    performance_averaging_functions = {
        "cat": partial(_calc_cat_averaging_value, metric_name="roc-auc-macro"),
        "con": partial(_calc_con_averaging_value, metric_name="loss"),
    }

    return performance_averaging_functions


def _log_num_params(model: nn.Module) -> None:
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def mixup_data(
    inputs: torch.Tensor,
    targets: al_training_labels_target,
    target_columns: al_target_columns,
    alpha: float = 1.0,
    mixing_type: str = "mixup",
):
    """
    :param inputs:
    :param targets:
    :param target_columns:
    :param alpha:
    :param mixing_type:
    :return:
    """

    if alpha > 0:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = 1.0

    batch_size = inputs.size()[0]
    random_index_for_mixing = get_random_index_for_mixing(batch_size=batch_size)
    targets_permuted = mixup_targets(
        targets=targets,
        random_index_for_mixing=random_index_for_mixing,
        target_columns=target_columns,
    )

    if mixing_type == "mixup":
        mixing_func = mixup_input
    elif mixing_type == "cutmix-block":
        mixing_func = block_cutmix_input
    elif mixing_type == "cutmix-uniform":
        mixing_func = uniform_cutmix_input
    else:
        raise ValueError()

    mixed_inputs = mixing_func(
        input_=inputs, lambda_=lambda_, random_index_for_mixing=random_index_for_mixing
    )

    return mixed_inputs, targets, targets_permuted, lambda_


def get_random_index_for_mixing(batch_size: int) -> torch.Tensor:
    return torch.randperm(batch_size)


def mixup_input(
    input_: torch.Tensor, lambda_: float, random_index_for_mixing: torch.Tensor
) -> torch.Tensor:
    mixed_x = lambda_ * input_ + (1 - lambda_) * input_[random_index_for_mixing, :]
    return mixed_x


def block_cutmix_input(
    input_: torch.Tensor, lambda_: float, random_index_for_mixing: torch.Tensor
) -> torch.Tensor:
    """
    We could even do the mixing in multiple places, even multiple SNPs like in
    the make random snps missing? Does not necessarily have to be one block at a time
    .
    """
    cut_start, cut_end = get_block_cutmix_indices(
        input_length=input_.shape[-1], lambda_=lambda_
    )
    target_to_cut = input_[random_index_for_mixing, :]
    cut_part = target_to_cut[..., cut_start:cut_end]
    cutmixed_x = input_
    cutmixed_x[..., cut_start:cut_end] = cut_part
    return cutmixed_x


def get_block_cutmix_indices(input_length: int, lambda_: float):
    mixin_coefficient = 1 - lambda_
    num_snps_to_mix = int(input_length * mixin_coefficient)
    random_index_start = np.random.choice(max(1, input_length - num_snps_to_mix))
    random_index_end = random_index_start + num_snps_to_mix
    return random_index_start, random_index_end


def uniform_cutmix_input(
    input_: torch.Tensor, lambda_: float, random_index_for_mixing: torch.Tensor
) -> torch.Tensor:
    """
    We could even do the mixing in multiple places, even multiple SNPs like in
    the make random snps missing? Does not necessarily have to be one block at a time
    .
    """

    target_to_mix = input_[random_index_for_mixing, :]

    random_snp_indices_to_mix = get_uniform_cutmix_indices(
        input_length=input_.shape[-1], lambda_=lambda_
    )
    cut_part = target_to_mix[..., random_snp_indices_to_mix]

    cutmixed_x = input_
    cutmixed_x[..., random_snp_indices_to_mix] = cut_part
    return cutmixed_x


def get_uniform_cutmix_indices(input_length: int, lambda_):
    mixin_coefficient = 1 - lambda_
    num_snps_to_mix = (int(input_length * mixin_coefficient),)
    random_to_mix = np.random.choice(input_length, num_snps_to_mix, replace=False)
    random_to_mix = torch.tensor(random_to_mix, dtype=torch.long)

    return random_to_mix


def mixup_targets(
    targets: al_training_labels_target,
    random_index_for_mixing: torch.Tensor,
    target_columns: al_target_columns,
) -> al_training_labels_target:
    targets_permuted = copy.copy(targets)

    for cat_target_col in target_columns["cat"]:

        cur_targets = targets_permuted[cat_target_col]
        cur_targets_permuted = cur_targets[random_index_for_mixing]
        targets_permuted[cat_target_col] = cur_targets_permuted

    return targets_permuted


def mixup_criterion(
    criterion: nn.CrossEntropyLoss,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    targets_permuted: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:

    base_loss = lambda_ * criterion(input=outputs, target=targets)
    permuted_loss = (1.0 - lambda_) * criterion(input=outputs, target=targets_permuted)

    total_loss = base_loss + permuted_loss

    return total_loss


def train(config: Config) -> None:
    c = config
    cl_args = config.cl_args

    extra_loss_functions = get_extra_loss_term_functions(
        model=c.model, l1_weight=cl_args.l1
    )
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

        train_seqs, labels, train_ids = loader_batch
        train_seqs = train_seqs.to(device=cl_args.device, non_blocking=True)
        train_seqs = train_seqs.to(dtype=torch.float32)

        target_labels = model_utils.parse_target_labels(
            target_columns=c.target_columns,
            device=cl_args.device,
            labels=labels["target_labels"],
        )

        # ----------- TMP -----------
        # TODO: Some kind of hook here, or in dataloader? Then parse_target labels
        # TODO: must be aware of that, better to have hook here and return DTO
        train_seqs_mixed, targets_a, targets_b, lambda_ = mixup_data(
            inputs=train_seqs,
            targets=target_labels,
            target_columns=c.target_columns,
            alpha=cl_args.mixing_alpha,
            mixing_type=cl_args.mixing_type,
        )
        # ----------- TMP -----------

        # TODO: We will have to mix extra inputs as well
        extra_inputs = get_extra_inputs(
            cl_args=cl_args, model=c.model, labels=labels["extra_labels"]
        )

        c.optimizer.zero_grad()

        train_outputs = c.model(x=train_seqs_mixed, extra_inputs=extra_inputs)

        # train_losses = c.loss_function(inputs=train_outputs, targets=target_labels)

        # ----------- TMP -----------
        tmp_target = c.target_columns["cat"][0]
        train_losses = mixup_criterion(
            criterion=c.criterions[tmp_target],
            outputs=train_outputs[tmp_target],
            targets=targets_a[tmp_target],
            targets_permuted=targets_b[tmp_target],
            lambda_=lambda_,
        )
        train_losses = {tmp_target: train_losses}
        # ----------- TMP -----------

        train_loss_avg = aggregate_losses(losses_dict=train_losses)
        train_loss_final = add_extra_losses(
            total_loss=train_loss_avg, extra_loss_functions=extra_loss_functions
        )

        train_loss_final.backward(**optimizer_backward_kwargs)
        c.optimizer.step()

        train_batch_metrics = calculate_batch_metrics(
            target_columns=c.target_columns,
            losses=train_losses,
            outputs=train_outputs,
            labels=target_labels,
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
    parser_.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")

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
        choices=["cnn", "mlp", "mlp-split", "mlp-mgmoe", "linear"],
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
        "--debug",
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


if __name__ == "__main__":

    parser = _get_train_argument_parser()
    cur_cl_args = parser.parse_args()
    cur_cl_args = _modify_train_arguments(cl_args=cur_cl_args)

    utils.configure_root_logger(run_name=cur_cl_args.run_name)

    main(cl_args=cur_cl_args)
