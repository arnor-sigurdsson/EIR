import argparse
import sys
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from sys import platform
from typing import Union, Tuple, List, Dict

import numpy as np
import torch
from aislib.misc_utils import ensure_path_exists
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import SGD
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler

from human_origins_supervised.data_load import datasets
from human_origins_supervised.data_load.data_loading_funcs import (
    get_weighted_random_sampler,
)
from human_origins_supervised.models import model_utils
from human_origins_supervised.models.extra_inputs_module import (
    set_up_and_save_embeddings_dict,
    get_extra_inputs,
    al_emb_lookup_dict,
)
from human_origins_supervised.models.model_utils import get_model_params, test_lr_range
from human_origins_supervised.models.models import get_model_class
from human_origins_supervised.train_utils.metric_funcs import select_metric_func
from human_origins_supervised.train_utils.train_handlers import configure_trainer
from human_origins_supervised.train_utils.utils import get_run_folder

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
    model: nn.Module
    optimizer: Optimizer
    criterion: Union[nn.CrossEntropyLoss, nn.MSELoss]
    labels_dict: Dict
    target_transformer: Union[LabelEncoder, StandardScaler]
    data_width: int


def train_ignite(config: Config) -> None:
    c = config
    args = config.cl_args

    metric_func = select_metric_func(args.model_task, c.target_transformer)

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, Union[int, float, List[None]], List[str]],
    ) -> Dict[str, float]:
        """
        The output here goes to engine.output.
        """
        c.model.train()

        train_seqs, train_labels, train_ids = loader_batch
        train_seqs = train_seqs.to(device=args.device, dtype=torch.float32)

        train_labels = train_labels.to(device=args.device)
        train_labels = model_utils.cast_labels(args.model_task, train_labels)

        extra_inputs = get_extra_inputs(args, train_ids, c.labels_dict, c.model)

        c.optimizer.zero_grad()
        train_outputs = c.model(train_seqs, extra_inputs)
        train_loss = c.criterion(train_outputs, train_labels)
        train_loss.backward()
        c.optimizer.step()

        train_loss = train_loss.item()
        metric_dict = metric_func(
            outputs=train_outputs, labels=train_labels, prefix="t"
        )
        metric_dict["t_loss"] = train_loss

        return metric_dict

    trainer = Engine(step)

    trainer = configure_trainer(trainer, config)

    trainer.run(c.train_loader, args.n_epochs)


def _prepare_run_folder(run_name: str) -> Path:
    run_folder = get_run_folder(run_name)
    history_file = run_folder / "training_history.log"
    if history_file.exists():
        raise FileExistsError(
            f"There already exists a run with that name: {history_file}. Please choose "
            f"a different run name or delete the folder."
        )

    ensure_path_exists(run_folder, is_folder=True)

    return run_folder


def get_train_sampler(
    cl_args: argparse.Namespace, train_dataset: datasets.ArrayDatasetBase
) -> Union[WeightedRandomSampler, None]:

    if cl_args.model_task == "cls" and cl_args.weighted_sampling:
        train_sampler = get_weighted_random_sampler(train_dataset)
        return train_sampler

    return None


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


def _modify_bs_for_multi_gpu(multi_gpu: bool, batch_size: int) -> int:
    if multi_gpu:
        batch_size = torch.cuda.device_count() * batch_size
        logger.info(
            "Batch size set to %d to account for %d GPUs.",
            batch_size,
            torch.cuda.device_count(),
        )

    return batch_size


def get_optimizer(model: nn.Module, cl_args: argparse.Namespace) -> Optimizer:
    params = get_model_params(model, cl_args.wd)

    if cl_args.optimizer == "adamw":
        optimizer = AdamW(
            params, lr=cl_args.lr, betas=(cl_args.b1, cl_args.b2), amsgrad=True
        )
    elif cl_args.optimizer == "sgdm":
        optimizer = SGD(params, lr=cl_args.lr, momentum=0.9)
    else:
        raise ValueError()

    return optimizer


def get_model(
    cl_args: argparse.Namespace,
    num_classes: int,
    embedding_dict: Union[al_emb_lookup_dict, None],
) -> Union[nn.Module, nn.DataParallel]:

    model_class = get_model_class(cl_args.model_type)
    model = model_class(cl_args, num_classes, embedding_dict, cl_args.contn_columns)

    if cl_args.model_type == "cnn":
        assert model.data_size_after_conv >= 8

    if cl_args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device=cl_args.device)

    return model


def get_criterion(
    cl_args: argparse.Namespace
) -> Union[nn.CrossEntropyLoss, nn.MSELoss]:
    if cl_args.model_task == "cls":
        return nn.CrossEntropyLoss()

    return nn.MSELoss()


def _log_params(model: nn.Module) -> None:
    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )


def main(cl_args: argparse.Namespace) -> None:
    run_folder = _prepare_run_folder(cl_args.run_name)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    cl_args.target_width = train_dataset[0][0].shape[2]
    cl_args.data_width = train_dataset.data_width

    batch_size = _modify_bs_for_multi_gpu(cl_args.multi_gpu, cl_args.batch_size)

    train_sampler = get_train_sampler(cl_args, train_dataset)

    train_dloader, valid_dloader = get_dataloaders(
        train_dataset=train_dataset,
        train_sampler=train_sampler,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
    )

    embedding_dict = set_up_and_save_embeddings_dict(
        cl_args.embed_columns, train_dataset.labels_dict, run_folder
    )

    model = get_model(cl_args, train_dataset.num_classes, embedding_dict)

    optimizer = get_optimizer(model, cl_args)

    criterion = get_criterion(cl_args)

    config = Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        train_dataset.labels_dict,
        train_dataset.target_transformer,
        train_dataset.data_width,
    )

    _log_params(model)

    if cl_args.find_lr:
        test_lr_range(config)
        sys.exit(0)

    if cl_args.debug:
        breakpoint()

    train_ignite(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_task",
        type=str,
        default="cls",
        choices=["cls", "reg"],
        help="Whether the task is a regression or classification.",
    )

    parser.add_argument(
        "--n_epochs", type=int, default=5, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")

    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Whether to perform a range test of different learning rates, with "
        "the lower limit being what is passed in for the --lr flag. "
        "Produces a plot and exits with status 0 before training if this flag "
        "is active.",
    )

    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=None,
        choices=["cycle", "plateau"],
        help="Whether to use cyclical or reduce on plateau learning rate schedule. "
        "Otherwise keeps same learning rate.",
    )

    parser.add_argument(
        "--lr_lb",
        type=float,
        default=0.0,
        help="Lower bound for learning rate when using LR scheduling.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "sgdm"],
        default="adamw",
        help="Whether to use AdamW or SGDM optimizer.",
    )

    parser.add_argument(
        "--b1",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )

    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of second order momentum of gradient",
    )
    parser.add_argument("--wd", type=float, default=0.00, help="Weight decay.")

    parser.add_argument(
        "--fc_dim",
        type=int,
        default=128,
        help="base dimensionality of fully connected layers at the end of the network",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "mlp"],
        help="base dimensionality of fully connected layers at the end of the network",
    )

    parser.add_argument(
        "--kernel_width",
        type=int,
        default=12,
        help="base width of the conv kernels used.",
    )

    parser.add_argument(
        "--down_stride",
        type=int,
        default=4,
        help="down stride to use common over the network.",
    )

    parser.add_argument(
        "--first_kernel_expansion",
        type=int,
        default=1,
        help="factor by which to expand the first kernel in the network",
    )

    parser.add_argument(
        "--first_stride_expansion",
        type=int,
        default=1,
        help="factor by which to expand the first stride in the network",
    )

    parser.add_argument(
        "--channel_exp_base",
        type=int,
        default=5,
        help="Exponential base for channels in first layer (i.e. default is 2**5)",
    )

    parser.add_argument(
        "--resblocks",
        type=int,
        nargs="+",
        help="Number of hidden convolutional layers.",
    )

    parser.add_argument(
        "--rb_do", type=float, default=0.0, help="Dropout in residual blocks."
    )

    parser.add_argument(
        "--fc_do",
        type=float,
        default=0.0,
        help="Dropout before last fully connected layer.",
    )

    parser.add_argument(
        "--sa",
        action="store_true",
        help="Whether to add self attention to the network.",
    )

    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Total width of input sequence after padding.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Data folder to load inputs from.",
    )

    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.05,
        help="Size if the validaton set, if float then uses a percentage. If int, "
        "then raw counts.",
    )

    parser.add_argument(
        "--weighted_sampling",
        action="store_true",
        help="Whether to use weighted sampling when doing classification.",
    )

    parser.add_argument(
        "--na_augment",
        default=0.0,
        type=float,
        help="Percentage of SNPs to convert to NA in training set as data augmentation",
    )

    parser.add_argument(
        "--label_file", type=str, required=True, help="Which file to load labels from."
    )

    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="What column in label file to treat as target variable"
        "(i.e. to predict on).",
    )

    parser.add_argument(
        "--embed_columns",
        type=str,
        nargs="+",
        default=[],
        help="What columns to embed and add to fully connected layer at end of model.",
    )

    parser.add_argument(
        "--contn_columns",
        type=str,
        nargs="+",
        default=[],
        help="What columns of continuous variables to add to fully connected layer at "
        "end of model.",
    )

    parser.add_argument(
        "--snp_file",
        type=str,
        default="infer",
        help="File to load SNPs from (.snp format).",
    )

    parser.add_argument(
        "--memory_dataset",
        action="store_true",
        help="Whether to load all sample into memory during " "training.",
    )

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=None,
        help="Epoch interval to sample generated seqs.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Epoch to checkpoint model.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the current run, specifying will save " "run info and models.",
    )

    parser.add_argument(
        "--gpu_num", type=str, default="0", help="Which GPU to run (according to CUDA)."
    )

    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Whether to run the training on " "multiple GPUs for the current node.",
    )

    parser.add_argument(
        "--get_acts", action="store_true", help="Whether to generate activation maps."
    )

    parser.add_argument(
        "--act_classes",
        default=None,
        nargs="+",
        help="Classes to use for activation maps.",
    )

    parser.add_argument("--benchmark", dest="benchmark", action="store_true")
    parser.add_argument("--no_benchmark", dest="benchmark", action="store_false")
    parser.set_defaults(benchmark=False)

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (w. breakpoint).",
    )

    parser.add_argument(
        "--custom_lib",
        type=str,
        default=None,
        help="Path to custom library if using one.",
    )

    cur_cl_args = parser.parse_args()

    if cur_cl_args.valid_size > 1.0:
        cur_cl_args.valid_size = int(cur_cl_args.valid_size)

    cur_cl_args.device = (
        "cuda:" + cur_cl_args.gpu_num if torch.cuda.is_available() else "cpu"
    )

    # to make sure importlib gets absolute paths
    if cur_cl_args.custom_lib is not None:
        cur_cl_args.custom_lib = abspath(cur_cl_args.custom_lib)

    # benchmark breaks if we run it with multiple GPUs
    if not cur_cl_args.multi_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        logger.debug("Setting device to cuda:0 since running with multiple GPUs.")
        cur_cl_args.device = "cuda:0"

    main(cur_cl_args)
