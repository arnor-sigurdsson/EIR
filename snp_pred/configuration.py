import argparse

import configargparse
import torch
from aislib.misc_utils import get_logger

from snp_pred.train_utils.optimizers import get_base_optimizers_dict

logger = get_logger(name=__name__)


def get_default_cl_args():
    parser = get_train_argument_parser()
    cl_args = parser.parse_args()
    cl_args = modify_train_arguments(cl_args=cl_args)

    return cl_args


def get_train_argument_parser() -> configargparse.ArgumentParser:

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
        default="adam",
        help="What optimizer to use.",
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
        "--label_parsing_chunk_size",
        type=int,
        default=None,
        help="Number of rows to load at a time from label file before processing.",
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
        default=None,
        help="Iteration to checkpoint model.",
    )

    parser_.add_argument(
        "--n_saved_models",
        type=int,
        default=None,
        help="Iteration to checkpoint model.",
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
        "--act_every_sample_factor",
        type=int,
        default=1,
        help="Number of rows to load at a time from label file before processing.",
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


def modify_train_arguments(cl_args: argparse.Namespace) -> argparse.Namespace:
    if cl_args.valid_size > 1.0:
        cl_args.valid_size = int(cl_args.valid_size)

    cl_args.device = "cuda:" + cl_args.gpu_num if torch.cuda.is_available() else "cpu"

    # benchmark breaks if we run it with multiple GPUs
    if not cl_args.multi_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        logger.debug("Setting device to cuda:0 since running with multiple GPUs.")
        cl_args.device = "cuda:0"

    return cl_args
