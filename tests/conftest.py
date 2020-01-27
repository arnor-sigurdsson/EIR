from pathlib import Path
from random import shuffle
from shutil import rmtree
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest
from aislib.misc_utils import ensure_path_exists
from torch import cuda
from torch.utils.data import DataLoader

from human_origins_supervised.data_load import datasets
from human_origins_supervised.models.models import CNNModel
from human_origins_supervised.train import get_optimizer

np.random.seed(0)


def pytest_addoption(parser):
    parser.addoption("--keep_outputs", action="store_true")
    parser.addoption(
        "--num_samples_per_class",
        type=int,
        default=1000,
        help="Number of samples per class.",
    )
    parser.addoption(
        "--num_snps", type=int, default=1000, help="Number of SNPs per sample."
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.keep_outputs
    if "keep_outputs" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("keep_outputs", [option_value])


@pytest.fixture
def parse_test_cl_args(request):
    n_per_class = request.config.getoption("--num_samples_per_class")
    num_snps = request.config.getoption("--num_snps")

    parsed_args = {"n_per_class": n_per_class, "n_snps": num_snps}

    return parsed_args


@pytest.fixture
def args_config():
    config = SimpleNamespace(
        **{
            "act_classes": None,
            "b1": 0.9,
            "b2": 0.999,
            "batch_size": 64,
            "benchmark": True,
            "channel_exp_base": 5,
            "checkpoint_interval": 100,
            "contn_columns": [],
            "custom_lib": None,
            "data_folder": "REPLACE_ME",
            "data_width": 1000,
            "device": "cuda:0" if cuda.is_available() else "cpu",
            "down_stride": 4,
            "embed_columns": [],
            "fc_dim": 128,
            "fc_do": 0.0,
            "first_kernel_expansion": 1,
            "first_stride_expansion": 1,
            "get_acts": True,
            "gpu_num": "0",
            "kernel_width": 12,
            "label_file": "REPLACE_ME",
            "lr": 1e-2,
            "lr_lb": 1e-5,
            "lr_schedule": "plateau",
            "memory_dataset": False,
            "model_type": "cnn",
            "multi_gpu": False,
            "n_cpu": 8,
            "n_epochs": 10,
            "na_augment": 0.0,
            "optimizer": "adamw",
            "rb_do": 0.0,
            "resblocks": None,
            "run_name": "test_run",
            "sa": False,
            "sample_interval": 20,
            "target_column": "Origin",
            "target_con_columns": [],
            "target_cat_columns": ["Origin"],
            "target_width": 1000,
            "valid_size": 0.05,
            "wd": 0.00,
            "weighted_sampling": False,
        }
    )

    return config


@pytest.fixture()
def create_test_cl_args(request, args_config, create_test_data):
    test_path, test_data_params = create_test_data

    model_task = "reg" if test_data_params["class_type"] == "regression" else "cls"
    n_snps = request.config.getoption("--num_snps")

    args_config.data_folder = str(test_path / "test_arrays")
    args_config.snp_file = str(test_path / "test_snps.bim")
    args_config.model_task = model_task
    args_config.label_file = str(test_path / "labels.csv")
    args_config.n_epochs = 5

    args_config.rb_do = 0.00
    args_config.fc_do = 0.00
    args_config.wd = 0.00
    args_config.na_augment = 0.00

    args_config.sample_interval = 100
    args_config.target_width = n_snps
    args_config.data_width = n_snps
    args_config.run_name = args_config.run_name + "_" + test_data_params["class_type"]

    # If tests need to have their own config different from the base defined above,
    # only supporting custom_cl_args hardcoded for now
    if hasattr(request, "param"):
        assert "custom_cl_args" in request.param.keys()
        custom_cl_args = request.param["custom_cl_args"]
        for k, v in custom_cl_args.items():
            setattr(args_config, k, v)

    # This is done after in case tests modify run_name
    args_config.run_name += "_" + args_config.model_type

    return args_config


@pytest.fixture()
def create_test_data(request, tmp_path, parse_test_cl_args):
    """
    Create a folder of data in same format, i.e. with ID_-_Class.
    Numpy arrays, can be in non-packbits format.
    Have different indexes for SNP locs in question.

    Also create SNP file. folder/arrays/ and folder/snp_file

    TODO:   Refactor this into different fixtures so that we don't have rewrite the
            test data to the disk multiple times
    """

    target_classes = {"Asia": 1, "Europe": 2}
    test_data_params = request.param

    if test_data_params["class_type"] in ("multi", "regression"):
        target_classes["Africa"] = 0

    n_per_class = parse_test_cl_args["n_per_class"]
    n_snps = parse_test_cl_args["n_snps"]

    test_task = "reg" if test_data_params["class_type"] == "regression" else "cls"

    label_file = open(str(tmp_path / "labels.csv"), "w")
    # extra col for testing extra inputs
    label_file.write("ID,Origin,OriginExtraCol,ExtraTarget\n")

    array_folder = set_up_test_data_array_outpath(tmp_path)

    for cls, snp_row_idx in target_classes.items():
        for sample_idx in range(n_per_class):

            outpath = array_folder / f"{sample_idx}_{cls}.npy"

            base_array, snp_idxs_candidates = set_up_test_array_base_params(n_snps)

            cur_test_array, snps_this_sample = create_test_array(
                test_task=test_task,
                base_array=base_array,
                snp_idxs_candidates=snp_idxs_candidates,
                snp_row_idx=snp_row_idx,
            )

            np.save(str(outpath), cur_test_array)

            if test_task == "reg":
                value_base = 100 + (5 * len(snps_this_sample)) + np.random.randn()
                values = [value_base] * 3 + [value_base - 10]
            else:
                values = [cls] * 4
            line_string = get_label_file_string(str(sample_idx), values)
            label_file.write(line_string)

    label_file.close()

    write_test_data_snp_file(tmp_path, n_snps)

    if test_data_params.get("split_to_test", False):
        split_test_array_folder(tmp_path)

    return tmp_path, request.param


def create_test_array(
    test_task: str,
    base_array: np.ndarray,
    snp_idxs_candidates: np.ndarray,
    snp_row_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # make samples have the reference, otherwise might have alleles chosen
    # below by random, without having the phenotype
    base_array[:, snp_idxs_candidates] = 0
    base_array[3, snp_idxs_candidates] = 1

    lower_bound = 0 if test_task == "reg" else 5
    np.random.shuffle(snp_idxs_candidates)
    num_snps_this_sample = np.random.randint(lower_bound, 10)
    snp_idxs = sorted(snp_idxs_candidates[:num_snps_this_sample])

    # zero out snp_idxs
    base_array[:, snp_idxs] = 0

    # assign form 2 for those snps for additive
    if test_task == "reg":
        base_array[2, snp_idxs] = 1
    else:
        base_array[snp_row_idx, snp_idxs] = 1

    base_array = base_array.astype(np.uint8)
    return base_array, snp_idxs


def set_up_test_data_array_outpath(base_folder: Path) -> Path:
    array_folder = base_folder / "test_arrays"
    array_folder.mkdir()

    return array_folder


def set_up_test_array_base_params(n_snps: int) -> Tuple[np.ndarray, np.ndarray]:
    # create random one hot array
    base_array = np.eye(4)[np.random.choice(4, n_snps)].T
    # set up 10 candidates
    step_size = n_snps // 10
    snp_idxs_candidates = np.array(range(50, n_snps, step_size))

    return base_array, snp_idxs_candidates


def get_label_file_string(sample_id: str, column_values: List[str]) -> str:
    values_part = ",".join(column_values)
    return f"{sample_id}_{values_part}\n"


def write_test_data_snp_file(base_folder: Path, n_snps: int) -> None:
    snp_file = base_folder / "test_snps.bim"
    base_snp_string_list = ["1", "REPLACE_W_IDX", "0.1", "10", "A", "T"]

    with open(str(snp_file), "w") as snpfile:
        for snp_idx in range(n_snps):
            cur_snp_list = base_snp_string_list[:]
            cur_snp_list[1] = str(snp_idx)

            cur_snp_string = "\t".join(cur_snp_list)
            snpfile.write(cur_snp_string + "\n")


def split_test_array_folder(test_folder: Path) -> None:
    test_array_test_set_folder = test_folder / "test_arrays_test_set"
    test_array_test_set_folder.mkdir()

    all_arrays = [i for i in (test_folder / "test_arrays").iterdir()]
    shuffle(all_arrays)

    test_arrays_test_set = all_arrays[:200]
    for array_file in test_arrays_test_set:
        array_file.replace(test_array_test_set_folder / array_file.name)


@pytest.fixture()
def create_test_model(create_test_cl_args, create_test_datasets):
    cl_args = create_test_cl_args
    train_dataset, _ = create_test_datasets

    model = CNNModel(
        cl_args,
        train_dataset.num_classes,
        extra_continuous_inputs_columns=cl_args.contn_columns,
    ).to(device=cl_args.device)

    return model


def cleanup(run_path):
    rmtree(run_path)


@pytest.fixture()
def create_test_datasets(create_test_data, create_test_cl_args):
    path, test_data_params = create_test_data

    cl_args = create_test_cl_args
    cl_args.data_folder = str(path / "test_arrays")

    run_path = Path(f"runs/{cl_args.run_name}/")

    # TODO: Use better logic here, to do the cleanup. Should not be in this fixture.
    if run_path.exists():
        cleanup(run_path)

    ensure_path_exists(run_path, is_folder=True)

    train_dataset, valid_dataset = datasets.set_up_datasets(cl_args)

    return train_dataset, valid_dataset


@pytest.fixture()
def create_test_dloaders(create_test_cl_args, create_test_datasets):
    cl_args = create_test_cl_args
    train_dataset, valid_dataset = create_test_datasets

    train_dloader = DataLoader(
        train_dataset, batch_size=cl_args.batch_size, shuffle=True
    )

    valid_dloader = DataLoader(
        valid_dataset, batch_size=cl_args.batch_size, shuffle=False
    )

    return train_dloader, valid_dloader, train_dataset, valid_dataset


@pytest.fixture()
def create_test_optimizer(create_test_cl_args, create_test_model):
    cl_args = create_test_cl_args
    model = create_test_model

    optimizer = get_optimizer(model, cl_args)

    return optimizer
