import argparse
import sys
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from sys import platform
from typing import Union, Tuple, List, Dict, overload, TYPE_CHECKING

import configargparse
import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
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
from human_origins_supervised.models.model_utils import get_model_params, test_lr_range
from human_origins_supervised.models.models import get_model_class, al_models
from human_origins_supervised.train_utils import utils
from human_origins_supervised.train_utils.metrics import (
    calculate_batch_metrics,
    calculate_losses,
    aggregate_losses,
    add_multi_task_average_metrics,
)
from human_origins_supervised.train_utils.train_handlers import configure_trainer

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.metrics import al_step_metric_dict

# aliases
al_criterions = Dict[str, Union[nn.CrossEntropyLoss, nn.MSELoss]]
# these are all after being collated by torch dataloaders
al_training_labels_target = Dict[str, Union[torch.LongTensor, torch.Tensor]]
al_training_labels_extra = Dict[str, Union[List[str], torch.Tensor]]
al_training_labels_batch = Dict[
    str, Union[al_training_labels_target, al_training_labels_extra]
]

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
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
    labels_dict: Dict
    target_transformers: al_label_transformers
    target_columns: al_target_columns
    data_width: int
    writer: SummaryWriter


def main(cl_args: argparse.Namespace) -> None:
    run_folder = _prepare_run_folder(run_name=cl_args.run_name)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args=cl_args)

    cl_args.target_width = train_dataset[0][0].shape[2]

    batch_size = _modify_bs_for_multi_gpu(
        multi_gpu=cl_args.multi_gpu, batch_size=cl_args.batch_size
    )

    train_sampler = get_train_sampler(
        column_to_sample=cl_args.weighted_sampling_column, train_dataset=train_dataset
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

    optimizer = get_optimizer(model=model, cl_args=cl_args)

    criterions = _get_criterions(
        target_columns=train_dataset.target_columns, model_type=cl_args.model_type
    )

    writer = get_summary_writer(run_folder=run_folder)

    config = Config(
        cl_args=cl_args,
        train_loader=train_dloader,
        valid_loader=valid_dloader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        labels_dict=train_dataset.labels_dict,
        target_transformers=train_dataset.target_transformers,
        target_columns=train_dataset.target_columns,
        data_width=train_dataset.data_width,
        writer=writer,
    )

    _log_num_params(model=model)

    if cl_args.find_lr:
        test_lr_range(config=config)
        sys.exit(0)

    if cl_args.debug:
        breakpoint()

    train(config=config)


def _prepare_run_folder(run_name: str) -> Path:
    run_folder = utils.get_run_folder(run_name=run_name)
    history_file = run_folder / "t_average_history.log"
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
    column_to_sample: None, train_dataset: datasets.ArrayDatasetBase
) -> None:
    ...


@overload
def get_train_sampler(
    column_to_sample: str, train_dataset: datasets.ArrayDatasetBase
) -> WeightedRandomSampler:
    ...


def get_train_sampler(column_to_sample, train_dataset):
    if column_to_sample is None:
        return None

    loaded_target_columns = (
        train_dataset.target_columns["con"] + train_dataset.target_columns["cat"]
    )
    if column_to_sample not in loaded_target_columns:
        raise ValueError("Weighted sampling from non-loaded columns not supported yet.")

    if column_to_sample is not None:
        train_sampler = get_weighted_random_sampler(
            train_dataset=train_dataset, target_column=column_to_sample
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


def get_model(
    cl_args: argparse.Namespace,
    num_classes: al_num_classes,
    embedding_dict: Union[al_emb_lookup_dict, None],
) -> Union[nn.Module, nn.DataParallel]:
    model_class = get_model_class(cl_args.model_type)
    model = model_class(cl_args, num_classes, embedding_dict, cl_args.extra_con_columns)

    if cl_args.model_type == "cnn":
        assert model.data_size_after_conv >= 8

    if cl_args.multi_gpu:
        model = nn.DataParallel(module=model)

    if model_class == "logreg":
        if (
            len(
                cl_args.target_cat_columns
                + cl_args.target_con_columns
                + cl_args.extra_cat_columns
                + cl_args.extra_con_columns
            )
            != 1
        ):
            raise ValueError()

        if len(cl_args.target_cat_columns != 1):
            raise ValueError()

    model = model.to(device=cl_args.device)

    return model


def get_optimizer(model: nn.Module, cl_args: argparse.Namespace) -> Optimizer:
    params = get_model_params(model=model, wd=cl_args.wd)

    if cl_args.optimizer == "adamw":
        optimizer = AdamW(
            params=params, lr=cl_args.lr, betas=(cl_args.b1, cl_args.b2), amsgrad=True
        )
    elif cl_args.optimizer == "sgdm":
        optimizer = SGD(params=params, lr=cl_args.lr, momentum=0.9)
    else:
        raise ValueError()

    return optimizer


def _get_criterions(
    target_columns: al_target_columns, model_type: str
) -> al_criterions:
    criterions_dict = {}

    bce_loss_func = nn.BCELoss()

    def calc_bce(input, target):
        return bce_loss_func(input[:, 1], target.to(dtype=torch.float))

    def get_criterion(column_type_):

        if model_type == "logreg":
            return calc_bce

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


def get_summary_writer(run_folder: Path) -> SummaryWriter:
    log_dir = Path(run_folder / "tensorboard_logs")
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def _log_num_params(model: nn.Module) -> None:
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def train(config: Config) -> None:
    c = config
    cl_args = config.cl_args

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, al_training_labels_batch, List[str]],
    ) -> "al_step_metric_dict":
        """
        The output here goes to trainer.output.
        """
        c.model.train()

        train_seqs, labels, train_ids = loader_batch
        train_seqs = train_seqs.to(device=cl_args.device, dtype=torch.float32)

        target_labels = model_utils.parse_target_labels(
            target_columns=c.target_columns,
            device=cl_args.device,
            labels=labels["target_labels"],
        )

        extra_inputs = get_extra_inputs(
            cl_args=cl_args, model=c.model, labels=labels["extra_labels"]
        )

        c.optimizer.zero_grad()
        train_outputs = c.model(train_seqs, extra_inputs)

        train_losses = calculate_losses(
            criterions=c.criterions, labels=target_labels, outputs=train_outputs
        )
        train_loss_avg = aggregate_losses(train_losses)
        l1_loss = torch.norm(c.model.fc_1.weight, p=1) * 0.1
        train_loss_avg += l1_loss
        train_loss_avg.backward()
        c.optimizer.step()

        batch_metrics_dict = calculate_batch_metrics(
            target_columns=c.target_columns,
            target_transformers=c.target_transformers,
            losses=train_losses,
            outputs=train_outputs,
            labels=target_labels,
            prefix="t_",
        )

        batch_metrics_dict_w_avgs = add_multi_task_average_metrics(
            batch_metrics_dict=batch_metrics_dict,
            target_columns=c.target_columns,
            prefix="t_",
            loss=train_loss_avg.item(),
        )

        return batch_metrics_dict_w_avgs

    trainer = Engine(process_function=step)

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
        choices=["cycle", "plateau", "same"],
        help="Whether to use cyclical or reduce on plateau learning rate schedule. "
        "Otherwise keeps same learning rate.",
    )

    parser_.add_argument(
        "--warmup_steps",
        type=str,
        default="auto",
        help="How many steps to use in warmup.",
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
        choices=["adamw", "sgdm"],
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
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "mlp", "logreg"],
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

    parser_.add_argument(
        "--resblocks",
        type=int,
        nargs="+",
        help="Number of hidden convolutional layers.",
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
        "--data_folder",
        type=str,
        required=True,
        help="Data folder to load inputs from.",
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
