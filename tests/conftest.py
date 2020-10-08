import csv
import warnings
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from shutil import rmtree
from types import SimpleNamespace
from typing import List, Tuple, Dict

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from aislib.misc_utils import ensure_path_exists
from torch import cuda
from torch import nn
from torch.utils.data import DataLoader

from snp_pred import train
from snp_pred.data_load import datasets
from snp_pred.models.extra_inputs_module import set_up_and_save_embeddings_dict
from snp_pred.train import Config, get_model
from snp_pred.train_utils import optimizers, metrics
from snp_pred.train_utils.utils import (
    configure_root_logger,
    get_run_folder,
)

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


@pytest.fixture(scope="session")
def parse_test_cl_args(request):
    n_per_class = request.config.getoption("--num_samples_per_class")
    num_snps = request.config.getoption("--num_snps")

    parsed_args = {"n_per_class": n_per_class, "n_snps": num_snps}

    return parsed_args


@pytest.fixture
def args_config():
    test_cl_args = SimpleNamespace(
        **{
            "act_classes": None,
            "b1": 0.9,
            "b2": 0.999,
            "batch_size": 32,
            "channel_exp_base": 5,
            "checkpoint_interval": 10000,
            "extra_con_columns": [],
            "custom_lib": None,
            "data_source": "REPLACE_ME",
            "device": "cuda:0" if cuda.is_available() else "cpu",
            "dilation_factor": 1,
            "dataloader_workers": 1,
            "down_stride": 4,
            "extra_cat_columns": [],
            "early_stopping_patience": None,
            "fc_repr_dim": 64,
            "fc_task_dim": 32,
            "fc_do": 0.0,
            "find_lr": False,
            "first_kernel_expansion": 1,
            "first_stride_expansion": 1,
            "first_channel_expansion": 1,
            "get_acts": True,
            "gpu_num": "0",
            "kernel_width": 12,
            "label_file": "REPLACE_ME",
            "l1": 0.0,
            "lr": 1e-2,
            "lr_lb": 1e-5,
            "lr_schedule": "plateau",
            "max_acts_per_class": None,
            "memory_dataset": True,
            "mixing_type": None,
            "mixing_alpha": 0.0,
            "mg_num_experts": 3,
            "model_type": "cnn",
            "multi_gpu": False,
            "n_cpu": 8,
            "n_epochs": 5,
            "na_augment_perc": 0.05,
            "na_augment_prob": 0.20,
            "no_pbar": False,
            "optimizer": "adam",
            "plot_skip_steps": 50,
            "rb_do": 0.25,
            "layers": [1, 1],
            "run_name": "test_run",
            "sa": False,
            "sample_interval": 200,
            "split_mlp_num_splits": 16,
            "target_cat_columns": ["Origin"],
            "target_con_columns": [],
            "target_width": 1000,
            "valid_size": 0.05,
            "warmup_steps": 100,
            "wd": 0.00,
            "weighted_sampling_column": None,
        }
    )

    return test_cl_args


@pytest.fixture(scope="module")
def create_test_data(request, tmp_path_factory, parse_test_cl_args) -> "TestDataConfig":
    c = _create_test_data_config(request, tmp_path_factory, parse_test_cl_args)

    fieldnames = ["ID", "Origin", "Height", "OriginExtraCol", "ExtraTarget"]

    label_file_handle, label_file_writer = _set_up_label_file_writing(
        path=c.scoped_tmp_path, fieldnames=fieldnames
    )

    array_outfolder = _set_up_test_data_array_outpath(c.scoped_tmp_path)

    for cls, snp_row_idx in c.target_classes.items():
        for sample_idx in range(c.n_per_class):

            sample_outpath = array_outfolder / f"{sample_idx}_{cls}"

            num_active_snps_in_sample = _save_test_array_to_disk(
                test_data_config=c,
                active_snp_row_idx=snp_row_idx,
                sample_outpath=sample_outpath,
            )

            label_line_base = _set_up_label_line_dict(
                sample_name=sample_outpath.name, fieldnames=fieldnames
            )

            label_line_dict = _get_current_test_label_values(
                values_dict=label_line_base,
                num_active_snps=num_active_snps_in_sample,
                cur_class=cls,
            )
            label_file_writer.writerow(label_line_dict)

    label_file_handle.close()

    write_test_data_snp_file(c.scoped_tmp_path, c.n_snps)

    if c.request_params.get("split_to_test", False):
        split_test_array_folder(c.scoped_tmp_path)

    return c


@dataclass
class TestDataConfig:
    request_params: Dict
    task_type: str
    scoped_tmp_path: Path
    target_classes: Dict[str, int]
    n_per_class: int
    n_snps: int


def _create_test_data_config(
    create_test_data_fixture_request: SubRequest, tmp_path_factory, parsed_test_cl_args
):

    request_params = create_test_data_fixture_request.param
    task_type = request_params["task_type"]
    scoped_tmp_path = tmp_path_factory.mktemp(task_type)

    target_classes = {"Asia": 0, "Europe": 1}
    if task_type != "binary":
        target_classes["Africa"] = 2

    config = TestDataConfig(
        request_params=request_params,
        task_type=task_type,
        scoped_tmp_path=scoped_tmp_path,
        target_classes=target_classes,
        n_per_class=parsed_test_cl_args["n_per_class"],
        n_snps=parsed_test_cl_args["n_snps"],
    )

    return config


def _set_up_label_file_writing(path: Path, fieldnames: List[str]):
    label_file = str(path / "labels.csv")

    label_file_handle = open(str(label_file), "w")

    writer = csv.DictWriter(f=label_file_handle, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()

    return label_file_handle, writer


def _set_up_label_line_dict(sample_name: str, fieldnames: List[str]):
    label_line_dict = {k: None for k in fieldnames}
    assert "ID" in label_line_dict.keys()
    label_line_dict["ID"] = sample_name
    return label_line_dict


def _get_current_test_label_values(values_dict, num_active_snps: List, cur_class: str):
    class_base_heights = {"Asia": 120, "Europe": 140, "Africa": 160}
    cur_base_height = class_base_heights[cur_class]

    added_height = 5 * len(num_active_snps)
    noise = np.random.randn()

    height_value = cur_base_height + added_height + noise
    values_dict["Height"] = height_value
    values_dict["ExtraTarget"] = height_value - 50

    values_dict["Origin"] = cur_class
    values_dict["OriginExtraCol"] = cur_class

    return values_dict


def _save_test_array_to_disk(
    test_data_config: TestDataConfig, active_snp_row_idx, sample_outpath: Path
):
    c = test_data_config

    base_array, snp_idxs_candidates = _set_up_base_test_array(c.n_snps)

    cur_test_array, snps_this_sample = _create_test_array(
        base_array=base_array,
        snp_idxs_candidates=snp_idxs_candidates,
        snp_row_idx=active_snp_row_idx,
    )

    np.save(str(sample_outpath), cur_test_array)

    return snps_this_sample


def _set_up_base_test_array(n_snps: int) -> Tuple[np.ndarray, np.ndarray]:
    # create random one hot array
    base_array = np.eye(4)[np.random.choice(4, n_snps)].T
    # set up 10 candidates
    step_size = n_snps // 10
    snp_idxs_candidates = np.array(range(50, n_snps, step_size))

    return base_array, snp_idxs_candidates


def _create_test_array(
    base_array: np.ndarray, snp_idxs_candidates: np.ndarray, snp_row_idx: int,
) -> Tuple[np.ndarray, List[int]]:
    # make samples have missing for chosen, otherwise might have alleles chosen
    # below by random, without having the phenotype
    base_array[:, snp_idxs_candidates] = 0
    base_array[3, snp_idxs_candidates] = 1

    lower_bound, upper_bound = 4, 11  # between 4 and 10 snps

    np.random.shuffle(snp_idxs_candidates)
    num_snps_this_sample = np.random.randint(lower_bound, upper_bound)
    snp_idxs = sorted(snp_idxs_candidates[:num_snps_this_sample])

    base_array[:, snp_idxs] = 0
    base_array[snp_row_idx, snp_idxs] = 1

    base_array = base_array.astype(np.uint8)
    return base_array, snp_idxs


def _set_up_test_data_array_outpath(base_folder: Path) -> Path:
    array_folder = base_folder / "test_arrays"
    if not array_folder.exists():
        array_folder.mkdir()

    return array_folder


def write_test_data_snp_file(base_folder: Path, n_snps: int) -> None:
    """
    BIM specs:
        0. Chromosome code
        1. Variant ID
        2. Position in centi-morgans
        3. Base-pair coordinate (1-based)
        4. ALT allele cod
        5. REF allele code
    """
    snp_file = base_folder / "test_snps.bim"
    base_snp_string_list = ["1", "REPLACE_W_IDX", "0.1", "REPLACE_W_IDX", "A", "T"]

    with open(str(snp_file), "w") as snpfile:
        for snp_idx in range(n_snps):
            cur_snp_list = base_snp_string_list[:]
            cur_snp_list[1] = str(snp_idx)
            cur_snp_list[3] = str(snp_idx)

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
def create_test_cl_args(request, args_config, create_test_data):
    c = create_test_data
    test_path = c.scoped_tmp_path

    n_snps = request.config.getoption("--num_snps")

    args_config.data_source = str(test_path / "test_arrays")
    args_config.snp_file = str(test_path / "test_snps.bim")
    args_config.model_task = c.task_type
    args_config.label_file = str(test_path / "labels.csv")

    args_config.target_width = n_snps

    # If tests need to have their own config different from the base defined above,
    # only supporting custom_cl_args hardcoded for now
    if hasattr(request, "param"):
        assert "custom_cl_args" in request.param.keys()
        custom_cl_args = request.param["custom_cl_args"]
        for k, v in custom_cl_args.items():
            setattr(args_config, k, v)

    # This is done after in case tests modify run_name
    args_config.run_name += (
        "_" + args_config.model_type + "_" + c.request_params["task_type"]
    )

    configure_root_logger(run_name=args_config.run_name)

    return args_config


@pytest.fixture()
def create_test_model(create_test_cl_args, create_test_datasets):
    cl_args = create_test_cl_args
    train_dataset, _ = create_test_datasets

    run_folder = get_run_folder(cl_args.run_name)

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

    return model


def cleanup(run_path):
    rmtree(run_path)


@pytest.fixture()
def create_test_datasets(create_test_data, create_test_cl_args):
    c = create_test_data

    cl_args = create_test_cl_args
    cl_args.data_source = str(c.scoped_tmp_path / "test_arrays")

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
        train_dataset, batch_size=cl_args.batch_size, shuffle=True, drop_last=True
    )

    valid_dloader = DataLoader(
        valid_dataset, batch_size=cl_args.batch_size, shuffle=False, drop_last=True
    )

    return train_dloader, valid_dloader, train_dataset, valid_dataset


def create_test_optimizer(
    cl_args: Namespace,
    model: nn.Module,
    target_columns: Dict[str, List[str]],
    criterions,
):

    """
    TODO: Refactor loss module construction out of this function.
    """

    loss_module = train._get_loss_callable(
        target_columns=target_columns, criterions=criterions, device=cl_args.device
    )

    optimizer = optimizers.get_optimizer(
        model=model, loss_callable=loss_module, cl_args=cl_args
    )

    return optimizer, loss_module


@dataclass
class ModelTestConfig:
    iteration: int
    run_path: Path
    last_sample_folders: Dict[str, Path]
    activations_path: Dict[str, Path]
    masked_activations_path: Dict[str, Path]


@pytest.fixture()
def prep_modelling_test_configs(
    create_test_data,
    create_test_cl_args,
    create_test_dloaders,
    create_test_model,
    create_test_datasets,
) -> Tuple[Config, ModelTestConfig]:
    """
    Note that the fixtures used in this fixture get indirectly parametrized by
    test_classification and test_regression.
    """
    cl_args = create_test_cl_args
    train_loader, valid_loader, train_dataset, valid_dataset = create_test_dloaders
    model = create_test_model

    criterions = train._get_criterions(
        target_columns=train_dataset.target_columns, model_type=cl_args.model_type
    )
    test_metrics = metrics.get_default_metrics(
        target_transformers=train_dataset.target_transformers
    )
    test_metrics = _patch_metrics(metrics=test_metrics)

    optimizer, loss_module = create_test_optimizer(
        cl_args=cl_args,
        model=model,
        target_columns=train_dataset.target_columns,
        criterions=criterions,
    )

    train_dataset, valid_dataset = create_test_datasets

    train._log_model(model=model, l1_weight=cl_args.l1)

    hooks = train._get_hooks(cl_args_=cl_args)
    config = Config(
        cl_args=cl_args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        loss_function=loss_module,
        metrics=test_metrics,
        labels_dict=train_dataset.labels_dict,
        target_transformers=train_dataset.target_transformers,
        target_columns=train_dataset.target_columns,
        data_width=train_dataset.data_width,
        writer=train.get_summary_writer(run_folder=Path("runs", cl_args.run_name)),
        hooks=hooks,
    )

    test_config = _get_cur_modelling_test_config(
        train_loader=train_loader, cl_args=cl_args
    )

    return config, test_config


def _patch_metrics(metrics):
    warnings.warn(
        "This function will soon be deprecated as conftest will need to "
        "create its own metrics when train.py default metrics will be "
        "minimal.",
        category=DeprecationWarning,
    )
    for type_ in ("cat", "con"):
        for metric_record in metrics[type_]:
            metric_record.only_val = False
    return metrics


def _get_cur_modelling_test_config(
    train_loader: DataLoader, cl_args: Namespace
) -> ModelTestConfig:

    last_iter = len(train_loader) * cl_args.n_epochs
    run_path = Path(f"runs/{cl_args.run_name}/")

    target_columns = cl_args.target_cat_columns + cl_args.target_con_columns
    last_sample_folders = _get_all_last_sample_folders(
        target_columns=target_columns, run_path=run_path, iteration=last_iter
    )

    gen = last_sample_folders.items
    activations_path = {k: folder / "activations/top_acts.npy" for k, folder in gen()}
    masked_activations_path = {
        k: folder / "activations/top_acts_masked.npy" for k, folder in gen()
    }

    test_config = ModelTestConfig(
        iteration=last_iter,
        run_path=run_path,
        last_sample_folders=last_sample_folders,
        activations_path=activations_path,
        masked_activations_path=masked_activations_path,
    )

    return test_config


def _get_all_last_sample_folders(
    target_columns: List[str], run_path: Path, iteration: int
) -> Dict[str, Path]:
    sample_folders = {}
    for col in target_columns:
        sample_folders[col] = _get_test_sample_folder(
            run_path=run_path, iteration=iteration, column_name=col
        )

    return sample_folders


def _get_test_sample_folder(run_path: Path, iteration: int, column_name: str) -> Path:
    sample_folder = run_path / f"results/{column_name}/samples/{iteration}"

    return sample_folder
