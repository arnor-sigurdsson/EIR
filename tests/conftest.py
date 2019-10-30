from types import SimpleNamespace

import numpy as np
from torch import cuda
import pytest

np.random.seed(0)


def pytest_addoption(parser):
    parser.addoption("--keep_outputs", action="store_true")


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
            "checkpoint_interval": 120,
            "data_folder": "REPLACE_ME",
            "valid_size": 0.10,
            "label_file": "REPLACE_ME",
            "label_column": "Origin",
            "data_type": "packbits",
            "data_width": 1000,
            "resblocks": None,
            "device": "cuda:0" if cuda.is_available() else "cpu",
            "gpu_num": "0",
            "lr": 1e-3,
            "wd": 5e-4,
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
            "rb_do": 0.0,
            "fc_do": 0.0,
            "memory_dataset": False,
            "embed_columns": [],
            "contn_columns": [],
            "na_augment": 0.0,
        }
    )

    return config


@pytest.fixture()
def create_test_cl_args(args_config, create_test_data):
    test_path, test_data_params = create_test_data

    model_task = "reg" if test_data_params["class_type"] == "regression" else "cls"

    args_config.data_folder = str(test_path / "test_arrays")
    args_config.snp_file = str(test_path / "test_snps.snp")
    args_config.model_task = model_task
    args_config.label_file = str(test_path / "labels.csv")
    args_config.n_epochs = 5
    args_config.rb_do = 0.00
    args_config.fc_do = 0.00
    args_config.na_augment = 0.00
    args_config.sample_interval = 100
    args_config.target_width = 1000
    args_config.data_width = 1000
    args_config.run_name = (
        args_config.run_name
        + "_"
        + test_data_params["class_type"]
        + "_"
        + test_data_params["data_type"]
    )

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

    n_per_class = 1000
    n_snps = 1000
    test_task = "reg" if test_data_params["class_type"] == "regression" else "cls"

    array_folder = tmp_path / "test_arrays"
    array_folder.mkdir()
    label_file = open(tmp_path / "labels.csv", "w")
    label_file.write("ID,Origin\n")

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
            if test_data_params["data_type"] == "packbits":
                arr_to_save = np.packbits(arr_to_save)

            np.save(outpath, arr_to_save)

            if test_data_params["class_type"] in ("binary", "multi"):
                label_file.write(f"{sample_idx}_{cls},{cls}\n")
            else:
                value = 100 + (5 * len(snps_this_sample)) + np.random.randn()
                label_file.write(f"{sample_idx}_{cls},{value}\n")

    label_file.close()

    snp_file = tmp_path / "test_snps.snp"
    with open(snp_file, "w") as snpfile:
        for snp_idx in range(n_snps):
            snpfile.write(f"rs{snp_idx}" + "\n")

    return tmp_path, request.param
