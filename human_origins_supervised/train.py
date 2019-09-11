import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from ignite.engine import Engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from human_origins_supervised.data_load import datasets
from human_origins_supervised.models import model_utils
from human_origins_supervised.models.models import Model
from human_origins_supervised.train_utils.metric_funcs import select_metric_func
from human_origins_supervised.train_utils.train_handlers import configure_trainer

torch.manual_seed(0)
np.random.seed(0)

logger = get_logger(__name__)


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
    criterion: nn.CrossEntropyLoss
    label_encoder: Union[LabelEncoder, StandardScaler]
    data_width: int


def train_ignite(config) -> None:
    c = config
    args = config.cl_args

    metric_func = select_metric_func(args.model_task, c.label_encoder)

    def step(
        engine: Engine,
        loader_batch: Tuple[torch.Tensor, Union[int, float, List[None]], List[str]],
    ) -> Dict[str, float]:
        """
        The output here goes to engine.output.
        """
        c.model.train()

        train_seqs, train_labels, *_ = loader_batch
        train_seqs = train_seqs.to(device=args.device, dtype=torch.float32)

        train_labels = train_labels.to(device=args.device)
        train_labels = model_utils.cast_labels(args.model_task, train_labels)

        c.optimizer.zero_grad()
        train_outputs = c.model(train_seqs)
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


def main(cl_args):
    run_folder = Path("runs", cl_args.run_name)
    if run_folder.exists():
        raise FileExistsError(
            "There already exists a run with that name, please"
            " choose a different one."
        )
    ensure_path_exists(run_folder, is_folder=True)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    cl_args.target_width = train_dataset[0][0].shape[2]
    cl_args.data_width = train_dataset.data_width

    train_dloader = DataLoader(
        train_dataset,
        batch_size=cl_args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    valid_dloader = DataLoader(
        valid_dataset,
        batch_size=cl_args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = Model(cl_args, train_dataset.num_classes).to(cl_args.device)
    assert model.data_size_after_conv > 8

    if cl_args.debug:
        breakpoint()

    optimizer = Adam(
        model.parameters(),
        lr=cl_args.lr,
        betas=(cl_args.b1, cl_args.b2),
        weight_decay=cl_args.wd,
        amsgrad=True,
    )

    criterion = nn.CrossEntropyLoss() if cl_args.model_task == "cls" else nn.MSELoss()

    config = Config(
        cl_args,
        train_dloader,
        valid_dloader,
        valid_dataset,
        model,
        optimizer,
        criterion,
        train_dataset.label_encoder,
        train_dataset.data_width,
    )

    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Starting training with a %s parameter model.", format(no_params, ",.0f")
    )

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
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay for adam.")

    parser.add_argument(
        "--kernel_width",
        type=int,
        default=12,
        help="base width of the conv kernels used.",
    )
    parser.add_argument(
        "--resblocks",
        type=int,
        nargs="+",
        help="Number of hidden convolutional layers.",
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
        "--label_file", type=str, required=True, help="Which file to load labels from."
    )

    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
        help="What column in label file to model on.",
    )

    parser.add_argument(
        "--snp_file",
        type=str,
        default="infer",
        help="File to load SNPs from (.snp format).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="uint8",
        choices=["packbits", "uint8"],
        help="Format of the data being passed in.",
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
        default=1000,
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
    parser.set_defaults(benchmark=True)

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (w. breakpoint).",
    )

    cur_cl_args = parser.parse_args()
    cur_cl_args.device = (
        "cuda:" + cur_cl_args.gpu_num if torch.cuda.is_available() else "cpu"
    )

    torch.backends.cudnn.benchmark = True

    main(cur_cl_args)
