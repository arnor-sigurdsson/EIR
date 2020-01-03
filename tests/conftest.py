from pathlib import Path
from random import shuffle
from shutil import rmtree
from types import SimpleNamespace

import numpy as np
import pytest
from aislib.misc_utils import ensure_path_exists
from torch import cuda
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from human_origins_supervised.data_load import datasets
from human_origins_supervised.models.model_utils import get_model_params
from human_origins_supervised.models.models import CNNModel

np.random.seed(0)


def pytest_addoption(parser):
    parser.addoption("--keep_outputs", action="store_true")
    parser.addoption(
        "--num_samples_per_class",
        type=int,
        default=2000,
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
def args_config():
    config = SimpleNamespace(
        **{
            "b1": 0.9,
            "b2": 0.999,
            "batch_size": 32,
            "checkpoint_interval": 100,
            "data_folder": "REPLACE_ME",
            "valid_size": 0.05,
            "label_file": "REPLACE_ME",
            "target_column": "Origin",
            "data_width": 1000,
            "resblocks": None,
            "device": "cuda:0" if cuda.is_available() else "cpu",
            "gpu_num": "0",
            "lr": 1e-2,
            "lr_lb": 1e-4,
            "cycle_lr": True,
            "wd": 0.0,
            "n_cpu": 8,
            "n_epochs": 10,
            "run_name": "test_run",
            "sample_interval": 20,
            "target_width": 1000,
            "act_classes": None,
            "get_acts": True,
            "benchmark": True,
            "kernel_width": 12,
            "fc_dim": 128,
            "down_stride": 4,
            "first_kernel_expansion": 1,
            "first_stride_expansion": 1,
            "channel_exp_base": 5,
            "sa": False,
            "rb_do": 0.0,
            "fc_do": 0.0,
            "memory_dataset": False,
            "embed_columns": [],
            "contn_columns": [],
            "na_augment": 0.0,
            "model_type": "cnn",
            "custom_lib": None,
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
    args_config.wd = 1e-4
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


def create_test_array(test_task, base_array, snp_idxs_candidates, snp_row_idx):
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

    return base_array, snp_idxs


def split_test_array_folder(test_folder: Path):
    test_array_test_set_folder = test_folder / "test_arrays_test_set"
    test_array_test_set_folder.mkdir()

    all_arrays = [i for i in (test_folder / "test_arrays").iterdir()]
    shuffle(all_arrays)

    test_arrays_test_set = all_arrays[:200]
    for array_file in test_arrays_test_set:
        array_file.replace(test_array_test_set_folder / array_file.name)


@pytest.fixture()
def create_test_data(request, tmp_path):
    """
    Create a folder of data in same format, i.e. with ID_-_Class.
    Numpy arrays, can be in non-packbits format.
    Have different indexes for SNP locs in question.

    Also create SNP file. folder/arrays/ and folder/snp_file
    """

    target_classes = {"Asia": 1, "Europe": 2}
    test_data_params = request.param

    if test_data_params["class_type"] in ("multi", "regression"):
        target_classes["Africa"] = 0

    n_per_class = request.config.getoption("--num_samples_per_class")
    n_snps = request.config.getoption("--num_snps")
    test_task = "reg" if test_data_params["class_type"] == "regression" else "cls"

    array_folder = tmp_path / "test_arrays"
    array_folder.mkdir()
    label_file = open(tmp_path / "labels.csv", "w")
    # extra col for testing extra inputs
    label_file.write("ID,Origin,OriginExtraCol\n")

    for cls, snp_row_idx in target_classes.items():
        for sample_idx in range(n_per_class):

            outpath = array_folder / f"{sample_idx}_{cls}.npy"

            # create random one hot array
            base_array = np.eye(4)[np.random.choice(4, n_snps)].T
            # set up 10 candidates
            step_size = n_snps // 10
            snp_idxs_candidates = np.array(range(50, n_snps, step_size))

            cur_test_array, snps_this_sample = create_test_array(
                test_task, base_array, snp_idxs_candidates, snp_row_idx
            )

            arr_to_save = cur_test_array.astype(np.uint8)
            np.save(outpath, arr_to_save)

            if test_data_params["class_type"] in ("binary", "multi"):
                label_file.write(f"{sample_idx}_{cls},{cls},{cls}\n")
            else:
                value = 100 + (5 * len(snps_this_sample)) + np.random.randn()
                label_file.write(f"{sample_idx}_{cls},{value},{value}\n")

    label_file.close()

    snp_file = tmp_path / "test_snps.bim"
    base_snp_string_list = ["1", "REPLACE_W_IDX", "0.1", "10", "A", "T"]
    with open(snp_file, "w") as snpfile:
        for snp_idx in range(n_snps):
            cur_snp_list = base_snp_string_list[:]
            cur_snp_list[1] = str(snp_idx)

            cur_snp_string = "\t".join(cur_snp_list)
            snpfile.write(cur_snp_string + "\n")

    if test_data_params.get("split_to_test", False):
        split_test_array_folder(tmp_path)

    return tmp_path, request.param


@pytest.fixture()
def create_test_model(create_test_cl_args, create_test_dataset):
    cl_args = create_test_cl_args
    train_dataset, _ = create_test_dataset

    model = CNNModel(
        cl_args,
        train_dataset.num_classes,
        extra_continuous_inputs_columns=cl_args.contn_columns,
    ).to(device=cl_args.device)

    return model


def cleanup(run_path):
    rmtree(run_path)


@pytest.fixture()
def create_test_dataset(create_test_data, create_test_cl_args):
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
def create_test_dloaders(create_test_dataset):
    train_dataset, valid_dataset = create_test_dataset

    train_dloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    valid_dloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_dloader, valid_dloader, train_dataset, valid_dataset


@pytest.fixture()
def create_test_optimizer(create_test_cl_args, create_test_model):
    cl_args = create_test_cl_args
    model = create_test_model

    params = get_model_params(model, cl_args.wd)
    optimizer = AdamW(params, betas=(cl_args.b1, cl_args.b2), amsgrad=True)

    return optimizer
