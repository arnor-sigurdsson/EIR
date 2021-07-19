import argparse
from copy import copy

import configargparse
import torch
from aislib.misc_utils import get_logger

from eir.train_utils.optimizers import get_base_optimizers_dict

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
        help="path to .yaml config file if using one.",
    )

    parser_.add_argument(
        "--n_epochs", type=int, default=5, help="Number of epochs of training."
    )
    parser_.add_argument(
        "--batch_size", type=int, default=64, help="Size of the batches."
    )

    parser_.add_argument(
        "--dataloader_workers",
        type=int,
        default=8,
        help="Number of workers for multi-process training and validation data "
        "loading.",
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
        help="Whether to use cyclical, cosine or reduce on plateau learning rate "
        "schedule. Otherwise keeps same learning rate.",
    )

    parser_.add_argument(
        "--lr_plateau_patience",
        type=int,
        default=10,
        help="Number of validation performance steps without improvement over "
        "best performance before reducing LR (only relevant when --lr_schedule is "
        "'plateau'.",
    )

    parser_.add_argument(
        "--lr_plateau_factor",
        type=float,
        default=0.1,
        help="Factor to reduce LR when running with plateau schedule.",
    )

    parser_.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of validation performance steps without improvement over "
        "best performance before terminating run.",
    )

    parser_.add_argument(
        "--early_stopping_buffer",
        type=int,
        default=None,
        help="Number of iterations to run before activation early stopping checks, "
        "useful if networks take a while to 'kick into gear'.",
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
        help="Decay of first order momentum of gradient for relevant optimizers.",
    )

    parser_.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="Decay of second order momentum of gradient for relevant optimizers.",
    )
    parser_.add_argument("--wd", type=float, default=0.00, help="Weight decay.")

    parser_.add_argument(
        "--l1", type=float, default=0.00, help="L1 regularization for chosen layer."
    )

    parser_.add_argument(
        "--fc_repr_dim",
        type=int,
        default=512,
        help="Dimensionality of first fc layer in the network, this is the last shared"
        "when running multi task training, first fc layer after convolutions when"
        "running cnn model, and first fc layer when running mlp.",
    )

    parser_.add_argument(
        "--fc_task_dim",
        type=int,
        default=128,
        help="Dimensionality of (a) specific task branches in multi task setting, (b)"
        "successive fc layers in cnn model after the first fc after the "
        "convolutions and (c) successive fc layers in mlp after first fc layer.",
    )

    parser_.add_argument(
        "--mg_num_experts",
        type=int,
        default=8,
        help="Number of experts to use when using the MGMoE fusion model.",
    )

    parser_.add_argument(
        "--split_mlp_num_splits",
        type=int,
        default=50,
        help="Number of splits in split MLP layer. Warning: Will be deprecated.",
    )

    parser_.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "mlp", "mlp-split", "genome-local-net", "linear"],
        help="Model type for omics model.",
    )

    parser_.add_argument(
        "--fusion_model_type",
        type=str,
        default="default",
        choices=["default", "mgmoe"],
        help="What type of fusion model to use.",
    )

    parser_.add_argument(
        "--kernel_width",
        type=int,
        default=12,
        help="Base width of the convolutional / locally-connected kernels used.",
    )

    parser_.add_argument(
        "--down_stride",
        type=int,
        default=4,
        help="Down stride to use common over the network when using CNN models.",
    )

    parser_.add_argument(
        "--dilation_factor",
        type=int,
        default=1,
        help="Factor to dilate convolutions by in each successive block when using CNN "
        "models.",
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
        help="Factor by which to expand the first stride in the network. "
        "Used for CNN and locally-connected models.",
    )

    parser_.add_argument(
        "--channel_exp_base",
        type=int,
        default=5,
        help="Exponential base for channels in first layer (i.e. default is 2**5). "
        "Used for CNN and locally-connected models.",
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
        help="Whether to add self attention to the network, only applies to CNN.",
    )

    parser_.add_argument(
        "--omics_sources",
        type=str,
        nargs="*",
        help="Which one-hot omics sources to load samples from for training. Can "
        "either be (a) a folder in which files will be gathered from the folder "
        "recursively or (b) a simple text file with each line having a path for "
        "a sample array",
    )

    parser_.add_argument(
        "--omics_names",
        type=str,
        nargs="*",
        help="Names for the omics sources passed in the --omics_sources argument.",
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
        help="Target column to apply weighted sampling on. Only applies to categorical "
        "columns. Passing in 'all' here will use an average of all the target columns.",
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
        help="What categorical columns from --label_file to include as input.",
    )

    parser_.add_argument(
        "--extra_con_columns",
        type=str,
        nargs="+",
        default=[],
        help="What continuous columns from --label_file to include as input.",
    )

    parser_.add_argument(
        "--snp_file",
        type=str,
        default="infer",
        help="File to load SNPs from (.bim format).",
    )

    parser_.add_argument(
        "--memory_dataset",
        action="store_true",
        help="Whether to load all sample into memory during training.",
    )

    parser_.add_argument(
        "--sample_interval",
        type=int,
        default=None,
        help="Iteration interval to perform validation.",
    )

    parser_.add_argument(
        "--checkpoint_interval",
        type=int,
        default=None,
        help="Iteration interval to checkpoint model.",
    )

    parser_.add_argument(
        "--n_saved_models",
        type=int,
        default=None,
        help="Number of top N models to saved during training.",
    )

    parser_.add_argument(
        "--run_name",
        required=True,
        type=str,
        help="Name of the current run.",
    )

    parser_.add_argument(
        "--gpu_num", type=str, default="0", help="Which GPU to run (according to CUDA)."
    )

    parser_.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Whether to run the training on multiple GPUs for the current node.",
    )

    parser_.add_argument(
        "--get_acts",
        action="store_true",
        help="Whether to compute activations w.r.t. inputs.",
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
        help="Controls whether the activations are computed at every sample interval "
        "(=1), every other sample interval (=2), etc. Useful when computing the "
        "activations takes a long time and we don't want to do it every time we "
        "evaluate.",
    )

    parser_.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode.",
    )

    parser_.add_argument(
        "--no_pbar",
        action="store_true",
        help="Whether to not use progress bars. Useful when stdout/stderr is written "
        "to files.",
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

    cl_args_copy = copy(cl_args)

    if cl_args_copy.valid_size > 1.0:
        cl_args_copy.valid_size = int(cl_args_copy.valid_size)

    cl_args_copy.device = (
        "cuda:" + cl_args_copy.gpu_num if torch.cuda.is_available() else "cpu"
    )

    # benchmark breaks if we run it with multiple GPUs
    if not cl_args_copy.multi_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        logger.debug("Setting device to cuda:0 since running with multiple GPUs.")
        cl_args_copy.device = "cuda:0"

    cl_args_copy = append_data_source_prefixes(cl_args=cl_args_copy)

    return cl_args_copy


def append_data_source_prefixes(cl_args: argparse.Namespace) -> argparse.Namespace:
    cl_args_copy = copy(cl_args)

    keys = cl_args.__dict__.keys()
    keys_to_change = [i for i in keys if i.endswith("_names")]

    for key in keys_to_change:
        new_names = []
        prev_names = cl_args_copy.__getattribute__(key)

        if not prev_names:
            continue

        key_type_name = key.split("_names")[0]

        for name in prev_names:
            new_name = key_type_name + "_" + name
            new_names.append(new_name)

        cl_args_copy.__setattr__(key, new_names)

    return cl_args_copy
