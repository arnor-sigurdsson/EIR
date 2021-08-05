import csv
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from shutil import rmtree
from typing import List, Tuple, Dict, Sequence, Mapping, Union, Literal

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from aislib.misc_utils import ensure_path_exists
from torch import nn
from torch.utils.data import DataLoader

import eir.models.omics.omics_models
import eir.setup.input_setup
import eir.train
from eir import train
from eir.data_load import datasets
from eir.setup import schemas, config
from eir.setup.config import recursive_dict_replace
from eir.setup.input_setup import serialize_all_input_transformers
from eir.train import (
    Experiment,
    get_model,
    set_up_num_outputs_per_target,
)
from eir.train_utils import optimizers, metrics
from eir.train_utils.utils import configure_root_logger, get_run_folder, seed_everything

al_prep_modelling_test_configs = Tuple[Experiment, "ModelTestConfig"]

seed_everything()


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


@dataclass
class TestConfigInits:
    global_configs: Sequence[dict]
    input_configs: Sequence[dict]
    predictor_configs: Sequence[dict]
    target_configs: Sequence[dict]


@pytest.fixture
def create_test_config_init_base(
    request, create_test_data
) -> Tuple[TestConfigInits, "TestDataConfig"]:

    injections = {}
    if hasattr(request, "param"):
        assert "injections" in request.param.keys()
        injections = request.param["injections"]

        injections_keys = set(injections.keys())
        expected_keys = set(TestConfigInits.__dataclass_fields__.keys())
        assert injections_keys.issubset(expected_keys)

    test_global_init = get_test_base_global_init()
    test_global_init = general_sequence_inject(
        sequence=test_global_init, inject_dict=injections.get("global_configs", {})
    )

    test_input_init = get_test_inputs_inits(
        test_path=create_test_data.scoped_tmp_path,
        input_config_dicts=injections.get("input_configs", {}),
    )

    model_type = injections.get("predictor_configs", {}).get("model_type", "nn")
    test_predictor_init = get_test_base_predictor_init(model_type=model_type)

    test_predictor_init = general_sequence_inject(
        sequence=test_predictor_init,
        inject_dict=injections.get("predictor_configs", {}),
    )

    test_target_inits = get_test_base_target_inits(test_data_config=create_test_data)
    test_target_inits = general_sequence_inject(
        sequence=test_target_inits,
        inject_dict=injections.get("target_configs", {}),
    )

    test_config = TestConfigInits(
        global_configs=test_global_init,
        input_configs=test_input_init,
        predictor_configs=test_predictor_init,
        target_configs=test_target_inits,
    )

    return test_config, create_test_data


def general_sequence_inject(
    sequence: Sequence[dict], inject_dict: dict
) -> Sequence[dict]:

    injected = []

    for dict_ in sequence:
        dict_injected = recursive_dict_replace(dict_=dict_, dict_to_inject=inject_dict)
        injected.append(dict_injected)

    return injected


def get_test_base_global_init() -> Sequence[dict]:
    global_inits = [
        {
            "run_name": "test_run",
            "plot_skip_steps": 0,
            "get_acts": True,
            "act_every_sample_factor": 0,
            "act_background_samples": 256,
            "n_epochs": 10,
            "warmup_steps": 100,
            "lr": 1e-02,
            "lr_lb": 1e-03,
            "batch_size": 32,
            "valid_size": 0.05,
            "wd": 1e-03,
        }
    ]
    return global_inits


def get_test_inputs_inits(
    test_path: Path, input_config_dicts: Sequence[dict]
) -> Sequence[dict]:

    inits = []

    base_func_map = get_input_test_init_base_func_map()
    for init_dict in input_config_dicts:
        cur_name = init_dict["input_info"]["input_name"]
        cur_base_func = base_func_map.get(cur_name)
        cur_init_base = cur_base_func(test_path=test_path)

        cur_init_injected = recursive_dict_replace(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_input_test_init_base_func_map():
    mapping = {
        "test_genotype": get_test_omics_input_init,
        "test_tabular": get_test_tabular_input_init,
    }

    return mapping


def get_test_omics_input_init(
    test_path: Path,
) -> dict:

    input_init_kwargs = {
        "input_info": {
            "input_source": str(test_path / "test_arrays"),
            "input_name": "test_genotype",
            "input_type": "omics",
        },
        "input_type_info": {
            "model_type": "genome-local-net",
            "na_augment_perc": 0.10,
            "na_augment_prob": 0.10,
            "snp_file": str(test_path / "test_snps.bim"),
        },
        "model_config": {},
    }

    return input_init_kwargs


def get_test_tabular_input_init(
    test_path: Path,
) -> dict:

    input_init_kwargs = {
        "input_info": {
            "input_source": str(test_path / "labels.csv"),
            "input_name": "test_tabular",
            "input_type": "tabular",
        },
        "input_type_info": {"model_type": "tabular"},
        "model_config": {},
    }

    return input_init_kwargs


def get_test_base_predictor_init(model_type: Literal["nn", "linear"]) -> Sequence[dict]:
    if model_type == "linear":
        return [{}]
    return [{"model_config": {"rb_do": 0.25, "fc_do": 0.25}}]


def get_test_base_target_inits(test_data_config: "TestDataConfig") -> Sequence[dict]:
    test_path = test_data_config.scoped_tmp_path

    test_target_init_kwargs = {
        "label_file": str(test_path / "labels.csv"),
        "target_cat_columns": ["Origin"],
    }

    test_target_init_kwargs_sequence = [test_target_init_kwargs]

    return test_target_init_kwargs_sequence


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

    test_data_config = TestDataConfig(
        request_params=request_params,
        task_type=task_type,
        scoped_tmp_path=scoped_tmp_path,
        target_classes=target_classes,
        n_per_class=parsed_test_cl_args["n_per_class"],
        n_snps=parsed_test_cl_args["n_snps"],
    )

    return test_data_config


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
    base_array: np.ndarray,
    snp_idxs_candidates: np.ndarray,
    snp_row_idx: int,
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
def create_test_config(
    create_test_config_init_base,
) -> config.Configs:

    test_init, test_data_config = copy(create_test_config_init_base)

    test_global_config = config.get_global_config(
        global_configs=test_init.global_configs
    )
    test_input_configs = config.get_input_configs(input_configs=test_init.input_configs)
    test_predictor_configs = config.load_predictor_config(
        predictor_configs=test_init.predictor_configs
    )
    test_target_configs = config.load_configs_general(
        config_dict_iterable=test_init.target_configs, cls=schemas.TargetConfig
    )

    test_configs = config.Configs(
        global_config=test_global_config,
        input_configs=test_input_configs,
        predictor_config=test_predictor_configs,
        target_configs=test_target_configs,
    )

    # This is done after in case tests modify run_name
    run_name = (
        test_configs.global_config.run_name
        + "_"
        + "_".join(i.input_type_info.model_type for i in test_configs.input_configs)
        + "_"
        + test_data_config.request_params["task_type"]
    )
    test_configs.global_config.run_name = run_name

    configure_root_logger(run_name=run_name)

    return test_configs


def modify_test_configs(
    configs: TestConfigInits, modifications: Union[Sequence[dict], dict]
) -> TestConfigInits:
    if isinstance(modifications, Mapping):
        assert set(modifications.keys()).issubset(set(configs.__dict__.keys()))

    configs_copy = copy(configs)

    for key, inner_modifications in modifications.items():
        # E.g. List of GlobalConfig objects
        cur_inner_configs = getattr(configs_copy, key)
        assert isinstance(cur_inner_configs, Sequence)

        for idx, inner_config_object in enumerate(cur_inner_configs):
            assert isinstance(inner_config_object, Mapping)

            if isinstance(modifications, Mapping):
                current_inner_mod = inner_modifications
            else:
                current_inner_mod = inner_modifications[idx]

            recursive_dict_replace(
                dict_=inner_config_object, dict_to_inject=current_inner_mod
            )

    return configs_copy


@pytest.fixture()
def get_test_data_dimensions(create_test_config: config.Configs, create_test_data):

    test_config = create_test_config

    for input_config in test_config.input_configs:
        if input_config.input_info.input_type == "omics":
            cur_source = input_config.input_info.input_source
            cur_dimensions = eir.setup.input_setup.get_data_dimension_from_data_source(
                data_source=Path(cur_source)
            )
            cur_name = input_config.input_info.input_name
            yield cur_name, cur_dimensions


@pytest.fixture()
def create_test_model(
    create_test_config: config.Configs, create_test_labels
) -> nn.Module:
    gc = create_test_config.global_config
    target_labels = create_test_labels

    num_outputs_per_class = set_up_num_outputs_per_target(
        target_transformers=target_labels.label_transformers
    )

    inputs_as_dict = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=tuple(create_test_labels.train_labels.keys()),
        valid_ids=tuple(create_test_labels.valid_labels.keys()),
        hooks=None,
    )

    model = get_model(
        inputs_as_dict=inputs_as_dict,
        global_config=gc,
        predictor_config=create_test_config.predictor_config,
        num_outputs_per_target=num_outputs_per_class,
    )

    return model


def set_up_inputs_as_dict(input_configs: Sequence[schemas.InputConfig]):
    input_name_config_iter = eir.setup.input_setup.get_input_name_config_iterator(
        input_configs=input_configs
    )
    inputs_as_dict = {k: v for k, v in input_name_config_iter}
    return inputs_as_dict


def cleanup(run_path):
    rmtree(run_path)


@pytest.fixture()
def create_test_labels(
    create_test_data, create_test_config: config.Configs
) -> train.Labels:

    c = create_test_config
    gc = c.global_config

    run_folder = get_run_folder(run_name=gc.run_name)

    # TODO: Use better logic here, to do the cleanup. Should not be in this fixture.
    if run_folder.exists():
        cleanup(run_folder)

    ensure_path_exists(run_folder, is_folder=True)

    all_array_ids = train.gather_all_ids_from_target_configs(
        target_configs=c.target_configs
    )
    train_ids, valid_ids = train.split_ids(ids=all_array_ids, valid_size=gc.valid_size)

    target_labels_info = train.get_tabular_target_file_infos(
        target_configs=c.target_configs
    )
    target_labels = train.set_up_target_labels_wrapper(
        tabular_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    train.save_transformer_set(
        transformers=target_labels.label_transformers, run_folder=run_folder
    )

    return target_labels


@pytest.fixture()
def create_test_datasets(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
):

    configs = create_test_config
    target_labels = create_test_labels

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
    )

    return train_dataset, valid_dataset


@pytest.fixture()
def create_test_dloaders(create_test_config: config.Configs, create_test_datasets):
    c = create_test_config
    gc = c.global_config
    train_dataset, valid_dataset = create_test_datasets

    train_dloader = DataLoader(
        train_dataset, batch_size=gc.batch_size, shuffle=True, drop_last=True
    )

    valid_dloader = DataLoader(
        valid_dataset, batch_size=gc.batch_size, shuffle=False, drop_last=True
    )

    return train_dloader, valid_dloader, train_dataset, valid_dataset


def create_test_optimizer(
    global_config: schemas.GlobalConfig,
    model: nn.Module,
    criterions,
):

    """
    TODO: Refactor loss module construction out of this function.
    """

    loss_module = train._get_loss_callable(criterions=criterions)

    optimizer = optimizers.get_optimizer(
        model=model, loss_callable=loss_module, global_config=global_config
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
    create_test_labels,
    create_test_config: config.Configs,
    create_test_dloaders,
    create_test_model,
    create_test_datasets,
    get_test_data_dimensions,
) -> Tuple[Experiment, ModelTestConfig]:
    """
    Note that the fixtures used in this fixture get indirectly parametrized by
    test_classification and test_regression.
    """
    c = create_test_config
    gc = c.global_config
    train_loader, valid_loader, train_dataset, valid_dataset = create_test_dloaders
    target_labels = create_test_labels

    model = create_test_model

    num_outputs_per_target = set_up_num_outputs_per_target(
        target_transformers=target_labels.label_transformers
    )

    criterions = train._get_criterions(target_columns=train_dataset.target_columns)
    test_metrics = metrics.get_default_metrics(
        target_transformers=target_labels.label_transformers,
    )
    test_metrics = _patch_metrics(metrics_=test_metrics)

    optimizer, loss_module = create_test_optimizer(
        global_config=gc,
        model=model,
        criterions=criterions,
    )

    train_dataset, valid_dataset = create_test_datasets

    train._log_model(model=model)

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=c.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )
    serialize_all_input_transformers(
        inputs_dict=inputs, run_folder=get_run_folder(gc.run_name)
    )

    hooks = train.get_default_hooks(configs=c)
    experiment = Experiment(
        configs=c,
        inputs=inputs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criterions=criterions,
        loss_function=loss_module,
        metrics=test_metrics,
        target_transformers=target_labels.label_transformers,
        num_outputs_per_target=num_outputs_per_target,
        target_columns=train_dataset.target_columns,
        writer=train.get_summary_writer(run_folder=Path("runs", gc.run_name)),
        hooks=hooks,
    )

    keys_to_serialize = train.get_default_experiment_keys_to_serialize()
    train.serialize_experiment(
        experiment=experiment,
        run_folder=get_run_folder(gc.run_name),
        keys_to_serialize=keys_to_serialize,
    )

    targets = config.get_all_targets(targets_configs=c.target_configs)
    test_config = _get_cur_modelling_test_config(
        train_loader=train_loader, global_config=gc, targets=targets
    )

    return experiment, test_config


def _patch_metrics(metrics_):
    warnings.warn(
        "This function will soon be deprecated as conftest will need to "
        "create its own metrics when train.py default metrics will be "
        "minimal.",
        category=DeprecationWarning,
    )
    for type_ in ("cat", "con"):
        for metric_record in metrics_[type_]:
            metric_record.only_val = False
    return metrics_


def _get_cur_modelling_test_config(
    train_loader: DataLoader,
    global_config: schemas.GlobalConfig,
    targets: config.Targets,
) -> ModelTestConfig:

    last_iter = len(train_loader) * global_config.n_epochs
    run_path = Path(f"runs/{global_config.run_name}/")

    last_sample_folders = _get_all_last_sample_folders(
        target_columns=targets.all_targets, run_path=run_path, iteration=last_iter
    )

    gen = last_sample_folders.items
    activations_path = {
        k: folder / "activations/omics_test_genotype/top_acts.npy"
        for k, folder in gen()
    }
    masked_activations_path = {
        k: folder / "activations/omics_test_genotype/top_acts_masked.npy"
        for k, folder in gen()
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


@pytest.fixture()
def get_transformer_test_data():
    test_labels_dict = {
        "1": {"Origin": "Asia", "Height": 150},
        "2": {"Origin": "Africa", "Height": 190},
        "3": {"Origin": "Europe", "Height": 170},
    }
    test_labels_df = pd.DataFrame(test_labels_dict).T

    test_target_columns_dict = {"con": ["Height"], "cat": ["Origin"]}

    return test_labels_df, test_target_columns_dict
