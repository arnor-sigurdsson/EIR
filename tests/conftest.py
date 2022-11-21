import json
import random
import warnings
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree
from typing import (
    Callable,
    Tuple,
    Dict,
    Sequence,
    Mapping,
    Union,
    Literal,
    Optional,
    Iterable,
    Any,
)

import deeplake
import numpy as np
import pandas as pd
import pytest
import torch.utils.data
from _pytest.fixtures import SubRequest
from aislib.misc_utils import ensure_path_exists
from torch import nn
from torch.utils.data import DataLoader

import eir.experiment_io.experiment_io
import eir.models.omics.omics_models
import eir.setup.input_setup
import eir.train
from eir import train
from eir.data_load import datasets
from eir.experiment_io.experiment_io import (
    serialize_all_input_transformers,
    serialize_chosen_input_objects,
)
from eir.models.model_setup import get_model, get_default_model_registry_per_input_type
from eir.setup import schemas, config
from eir.setup.config import recursive_dict_replace
from eir.setup.output_setup import (
    set_up_outputs_for_training,
)
from eir.train import (
    Experiment,
)
from eir.train_utils import optimizers, metrics
from eir.train_utils.utils import configure_root_logger, get_run_folder, seed_everything
from tests.test_modelling.setup_modelling_test_data.setup_image_test_data import (
    create_test_image_data,
)
from tests.test_modelling.setup_modelling_test_data.setup_omics_test_data import (
    create_test_omics_data_and_labels,
)
from tests.test_modelling.setup_modelling_test_data.setup_sequence_test_data import (
    create_test_sequence_data,
)
from tests.test_modelling.setup_modelling_test_data.setup_test_data_utils import (
    set_up_test_data_root_outpath,
    common_split_test_data_wrapper,
)

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


def get_system_info() -> Tuple[bool, str]:
    import os
    import platform

    in_gh_actions = os.environ.get("GITHUB_ACTIONS", False)
    if in_gh_actions:
        in_gh_actions = True

    system = platform.system()

    return in_gh_actions, system


def should_skip_in_gha():

    in_gha, _ = get_system_info()
    if in_gha:
        return True

    return False


def should_skip_in_gha_macos():
    """
    We use this to skip some modelling tests as the GHA MacOS runner can be very slow.
    """
    in_gha, platform = get_system_info()
    if in_gha and platform == "Darwin":
        return True

    return False


@pytest.fixture(scope="session")
def parse_test_cl_args(request) -> Dict[str, Any]:
    n_per_class = request.config.getoption("--num_samples_per_class")
    num_snps = request.config.getoption("--num_snps")

    parsed_args = {"n_per_class": n_per_class, "n_snps": num_snps}

    return parsed_args


@dataclass
class TestConfigInits:
    global_configs: Sequence[dict]
    input_configs: Sequence[dict]
    fusion_configs: Sequence[dict]
    output_configs: Sequence[dict]


@pytest.fixture
def create_test_config_init_base(
    request, create_test_data: "TestDataConfig"
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
        split_to_test=create_test_data.request_params.get("split_to_test", False),
        source=create_test_data.source,
        extra_kwargs=create_test_data.extras,
    )

    model_type = injections.get("fusion_configs", {}).get("model_type", "default")
    test_fusion_init = get_test_base_fusion_init(model_type=model_type)

    test_fusion_init = general_sequence_inject(
        sequence=test_fusion_init,
        inject_dict=injections.get("fusion_configs", {}),
    )

    test_output_init = get_test_outputs_inits(
        test_path=create_test_data.scoped_tmp_path,
        output_configs_dicts=injections.get("output_configs", {}),
        split_to_test=create_test_data.request_params.get("split_to_test", False),
    )

    test_config = TestConfigInits(
        global_configs=test_global_init,
        input_configs=test_input_init,
        fusion_configs=test_fusion_init,
        output_configs=test_output_init,
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
            "output_folder": "runs/test_run",
            "plot_skip_steps": 0,
            "get_acts": True,
            "act_every_sample_factor": 0,
            "act_background_samples": 256,
            "n_epochs": 12,
            "warmup_steps": 100,
            "lr": 2e-03,
            "optimizer": "adamw",
            "lr_lb": 1e-05,
            "batch_size": 32,
            "valid_size": 0.05,
            "wd": 1e-03,
        }
    ]
    return global_inits


def get_test_inputs_inits(
    test_path: Path,
    input_config_dicts: Sequence[dict],
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    extra_kwargs: Optional[dict] = None,
) -> Sequence[dict]:

    if extra_kwargs is None:
        extra_kwargs = {}

    inits = []

    base_func_map = get_input_test_init_base_func_map()
    for init_dict in input_config_dicts:
        cur_name = init_dict["input_info"]["input_name"]

        cur_base_func_keys = [i for i in base_func_map.keys() if cur_name.startswith(i)]
        assert len(cur_base_func_keys) == 1
        cur_base_func_key = cur_base_func_keys[0]

        cur_base_func = base_func_map.get(cur_base_func_key)
        cur_init_base = cur_base_func(
            init_dict=init_dict,
            test_path=test_path,
            split_to_test=split_to_test,
            source=source,
            extra_kwargs=extra_kwargs,
        )

        cur_init_injected = recursive_dict_replace(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_test_outputs_inits(
    test_path: Path, output_configs_dicts: Sequence[dict], split_to_test: bool
) -> Sequence[dict]:

    inits = []

    base_func_map = get_output_test_init_base_func_map()

    for init_dict in output_configs_dicts:
        cur_name = init_dict["output_info"]["output_name"]

        cur_base_func_keys = [i for i in base_func_map.keys() if cur_name == i]
        assert len(cur_base_func_keys) == 1
        cur_base_func_key = cur_base_func_keys[0]

        cur_base_func = base_func_map.get(cur_base_func_key)
        cur_init_base = cur_base_func(test_path=test_path, split_to_test=split_to_test)

        cur_init_injected = recursive_dict_replace(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_output_test_init_base_func_map() -> Dict[str, Callable]:
    mapping = {
        "test_output": get_test_base_output_inits,
        "test_output_copy": get_test_base_output_inits,
    }

    return mapping


def get_input_test_init_base_func_map() -> Dict[str, Callable]:
    mapping = {
        "test_genotype": get_test_omics_input_init,
        "test_tabular": get_test_tabular_input_init,
        "test_sequence": get_test_sequence_input_init,
        "test_bytes": get_test_bytes_input_init,
        "test_image": get_test_image_input_init,
    }

    return mapping


def _inject_train_source_path(
    test_path: Path,
    source: Literal["local", "deeplake"],
    local_name: Literal["omics", "sequence", "image"],
    split_to_test: bool,
) -> Path:

    if source == "local":
        input_source = test_path / local_name

        if split_to_test:
            input_source = input_source / "train_set"

    elif source == "deeplake":

        input_source = test_path / "deeplake"
        if split_to_test:
            input_source = test_path / "deeplake_train_set"

    else:
        raise ValueError(f"Source {source} not supported.")

    return input_source


def get_test_omics_input_init(
    test_path: Path,
    split_to_test: bool,
    init_dict: Dict,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> dict:

    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="omics",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_genotype",
            "input_type": "omics",
            "input_inner_key": "test_genotype",
        },
        "input_type_info": {
            "na_augment_perc": 0.10,
            "na_augment_prob": 0.10,
            "snp_file": str(test_path / "test_snps.bim"),
        },
        "model_config": {"model_type": "genome-local-net"},
    }

    if init_dict.get("input_type_info", {}).get("subset_snps_file", None) == "auto":
        subset_path = str(test_path / "test_subset_snps.txt")
        init_dict["input_type_info"]["subset_snps_file"] = subset_path

    return input_init_kwargs


def get_test_tabular_input_init(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> dict:

    input_source = test_path / "labels.csv"
    if split_to_test:
        input_source = test_path / "labels_train.csv"

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_tabular",
            "input_type": "tabular",
        },
        "input_type_info": {},
        "model_config": {"model_type": "tabular"},
    }

    return input_init_kwargs


def get_test_sequence_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    extra_kwargs: dict,
    *args,
    **kwargs,
) -> dict:

    if extra_kwargs.get("sequence_csv_source", False):
        assert source == "local"
        name = "sequence.csv"
        if split_to_test:
            name = "sequence_train.csv"
        input_source = test_path / name
    else:
        input_source = _inject_train_source_path(
            test_path=test_path,
            source=source,
            local_name="sequence",
            split_to_test=split_to_test,
        )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_sequence",
            "input_type": "sequence",
            "input_inner_key": "test_sequence",
        },
        "input_type_info": {
            "max_length": "max",
            "tokenizer_language": "en",
        },
        "model_config": {
            "model_type": "sequence-default",
            "embedding_dim": 64,
            "model_init_config": {
                "num_heads": 2,
                "num_layers": 1,
                "dropout": 0.10,
            },
        },
    }

    return input_init_kwargs


def get_test_bytes_input_init(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> Dict:
    input_source = test_path / "sequence"
    if split_to_test:
        input_source = input_source / "train_set"

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_sequence",
            "input_type": "bytes",
        },
        "input_type_info": {
            "max_length": 128,
        },
        "model_config": {
            "model_type": "sequence-default",
            "embedding_dim": 8,
            "window_size": 64,
        },
    }

    return input_init_kwargs


def get_test_image_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> Dict:

    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="image",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_image",
            "input_type": "image",
            "input_inner_key": "test_image",
        },
        "input_type_info": {
            "auto_augment": False,
            "size": (16,),
        },
        "model_config": {
            "model_type": "ResNet",
            "pretrained_model": False,
            "num_output_features": 128,
            "freeze_pretrained_model": False,
            "model_init_config": {
                "layers": [1, 1, 1, 1],
                "block": "BasicBlock",
            },
        },
    }

    return input_init_kwargs


def get_test_base_fusion_init(model_type: str) -> Sequence[dict]:
    if model_type == "identity":
        return [{}]
    elif model_type in ("default", "mgmoe"):
        return [
            {
                "model_config": {
                    "rb_do": 0.1,
                    "fc_do": 0.1,
                    "layers": [1],
                    "fc_task_dim": 128,
                }
            }
        ]
    else:
        raise ValueError()


def get_test_base_output_inits(test_path: Path, split_to_test: bool) -> Dict:

    label_file = test_path / "labels.csv"
    if split_to_test:
        label_file = test_path / "labels_train.csv"

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output",
            "output_type": "tabular",
            "output_source": str(label_file),
        },
        "output_type_info": {
            "target_cat_columns": ["Origin"],
        },
        "model_config": {
            "model_init_config": {
                "layers": [1],
                "fc_task_dim": 128,
            }
        },
    }

    return test_target_init_kwargs


@pytest.fixture()
def create_test_data(request, tmp_path_factory, parse_test_cl_args) -> "TestDataConfig":
    test_data_config = _create_test_data_config(
        create_test_data_fixture_request=request,
        tmp_path_factory=tmp_path_factory,
        parsed_test_cl_args=parse_test_cl_args,
    )

    base_outfolder = set_up_test_data_root_outpath(
        base_folder=test_data_config.scoped_tmp_path
    )

    drop_random_samples = test_data_config.random_samples_dropped_from_modalities

    omics_path = base_outfolder / "omics"
    if "omics" in test_data_config.modalities and not omics_path.exists():
        omics_sample_path = create_test_omics_data_and_labels(
            test_data_config=test_data_config,
            array_outfolder=omics_path,
        )

        if drop_random_samples:
            _delete_random_files_from_folder(folder=omics_sample_path, n_to_drop=50)

    image_path = base_outfolder / "image"
    if "image" in test_data_config.modalities and not image_path.exists():
        image_sample_folder = create_test_image_data(
            test_data_config=test_data_config,
            image_output_folder=image_path,
        )
        if drop_random_samples:
            _delete_random_files_from_folder(folder=image_sample_folder, n_to_drop=50)

    sequence_path = base_outfolder / "sequence"
    if "sequence" in test_data_config.modalities and not sequence_path.exists():
        sequence_sample_folder = create_test_sequence_data(
            test_data_config=test_data_config,
            sequence_outfolder=sequence_path,
        )
        if drop_random_samples:
            _delete_random_files_from_folder(
                folder=sequence_sample_folder, n_to_drop=50
            )

    _merge_labels_from_modalities(base_path=base_outfolder)

    if drop_random_samples:
        label_file = test_data_config.scoped_tmp_path / "labels.csv"
        _delete_random_rows_from_csv(csv_file=label_file, n_to_drop=50)

    if test_data_config.request_params.get("split_to_test", False):
        post_split_callables = _get_test_post_split_callables()
        for modality in test_data_config.modalities:
            common_split_test_data_wrapper(
                test_folder=test_data_config.scoped_tmp_path,
                name=modality,
                post_split_callables=post_split_callables,
            )

    if test_data_config.request_params.get("split_to_test", False):
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name="train_set"
        )
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name="test_set"
        )
    else:
        _make_deeplake_test_dataset(
            base_output_folder=base_outfolder, sub_folder_name=None
        )

    return test_data_config


def _get_test_post_split_callables() -> Dict[str, Callable]:
    def _sequence_post_split(
        test_root_folder: Path,
        train_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> None:
        df_sequence = pd.read_csv(test_root_folder / "sequence.csv")

        df_sequence_train = df_sequence[df_sequence["ID"].isin(train_ids)]
        df_sequence_test = df_sequence[df_sequence["ID"].isin(test_ids)]

        df_sequence_train.to_csv(test_root_folder / "sequence_train.csv", index=False)
        df_sequence_test.to_csv(test_root_folder / "sequence_test.csv", index=False)

        (test_root_folder / "sequence.csv").unlink()

    callables = {"sequence": _sequence_post_split}

    return callables


def _merge_labels_from_modalities(base_path: Path) -> None:
    dfs = []

    for file in base_path.iterdir():
        # if we have already merged the labels
        if file.name in ("labels.csv", "labels_train.csv", "labels_test.csv"):
            return

        elif file.name.startswith("labels_"):
            assert file.suffix == ".csv"
            dfs.append(pd.read_csv(file, index_col="ID"))

    df_final = dfs[0].copy()

    if len(dfs) == 1:
        df_final.to_csv(base_path / "labels.csv")
        return

    for df in dfs[1:]:
        assert df["Origin"].equals(df_final["Origin"])
        assert df["OriginExtraCol"].equals(df_final["Origin"])
        assert df.index.equals(df_final.index)

        df_final["Height"] += df["Height"]
        df_final["ExtraTarget"] += df["ExtraTarget"]

    df_final["Height"] /= len(dfs)
    df_final["ExtraTarget"] /= len(dfs)

    df_final.to_csv(base_path / "labels.csv")


def _make_deeplake_test_dataset(
    base_output_folder: Path,
    sub_folder_name: Union[None, Literal["train_set", "test_set"]],
) -> None:

    if sub_folder_name is None:
        suffix = ""
    else:
        suffix = f"_{sub_folder_name}"

    if (base_output_folder / f"deeplake{suffix}").exists():
        return

    samples = {}
    for f in base_output_folder.iterdir():
        if not f.is_dir() or "deeplake" in f.name:
            continue

        file_iterator = f.iterdir()
        if sub_folder_name is not None:
            file_iterator = (f / sub_folder_name).iterdir()

        for sample_file in file_iterator:

            sample_id = sample_file.stem
            if sample_id not in samples:
                samples[sample_id] = {"ID": sample_id}

            if f.name == "omics":
                cur_name = "test_genotype"
                sample_data = np.load(str(sample_file))
            elif f.name == "image":
                cur_name = "test_image"
                sample_data = datasets.default_loader(str(sample_file))
                sample_data = np.array(sample_data)
            elif f.name == "sequence":
                cur_name = "test_sequence"
                sample_data = sample_file.read_text().strip()
            else:
                raise ValueError()

            samples[sample_id][cur_name] = sample_data

    name = "deeplake"
    if sub_folder_name is not None:
        name = f"{name}_{sub_folder_name}"
    ds = deeplake.empty(base_output_folder / name, overwrite=True)

    ds.create_tensor("ID", htype="text")
    ds.create_tensor("test_genotype")
    ds.create_tensor("test_image", htype="image", sample_compression="jpg")
    ds.create_tensor("test_sequence", htype="text")
    with ds:
        for sample_id, sample in samples.items():
            ds.append(sample, append_empty=True)


def _delete_random_rows_from_csv(csv_file: Path, n_to_drop: int):
    df = pd.read_csv(filepath_or_buffer=csv_file, index_col=0)
    drop_indices = np.random.choice(df.index, n_to_drop, replace=False)
    df_subset = df.drop(drop_indices)
    df_subset.to_csv(path_or_buf=csv_file)


def _delete_random_files_from_folder(folder: Path, n_to_drop: int):
    all_files = tuple(folder.iterdir())
    to_drop = random.sample(population=all_files, k=n_to_drop)

    for f in to_drop:
        f.unlink()


@dataclass
class TestDataConfig:
    request_params: Dict
    task_type: str
    scoped_tmp_path: Path
    target_classes: Dict[str, int]
    n_per_class: int
    n_snps: int
    modalities: Sequence[Union[Literal["omics"], Literal["sequence"]]] = ("omics",)
    random_samples_dropped_from_modalities: bool = False
    source: Literal["local", "deeplake"] = "local"
    extras: Dict[str, Any] = field(default_factory=dict)


def _create_test_data_config(
    create_test_data_fixture_request: SubRequest, tmp_path_factory, parsed_test_cl_args
) -> TestDataConfig:

    request_params = create_test_data_fixture_request.param
    task_type = request_params["task_type"]

    if request_params.get("manual_test_data_creator", None):
        manual_name_creator_callable = request_params["manual_test_data_creator"]
        basename = str(manual_name_creator_callable())
    else:
        basename = "test_data_" + str(
            _hash_dict(dict_to_hash={**request_params, **parsed_test_cl_args})
        )

    scoped_tmp_path = tmp_path_factory.getbasetemp().joinpath(basename)

    if not scoped_tmp_path.exists():
        scoped_tmp_path.mkdir(mode=0o700)

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
        modalities=request_params.get("modalities", ("omics",)),
        random_samples_dropped_from_modalities=request_params.get(
            "random_samples_dropped_from_modalities", False
        ),
        source=request_params.get("source", "local"),
        extras=request_params.get("extras", {}),
    )

    return test_data_config


def _hash_dict(dict_to_hash: dict) -> int:
    dict_hash = hash(json.dumps(dict_to_hash, sort_keys=True))
    return dict_hash


@pytest.fixture()
def create_test_config(
    create_test_config_init_base, keep_outputs: bool
) -> config.Configs:

    test_init, test_data_config = copy(create_test_config_init_base)

    test_global_config = config.get_global_config(
        global_configs=test_init.global_configs
    )

    test_input_configs = config.get_input_configs(input_configs=test_init.input_configs)
    test_fusion_configs = config.load_fusion_configs(
        fusion_configs=test_init.fusion_configs
    )

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    test_output_configs = config.load_output_configs(
        output_configs=test_init.output_configs,
        dynamic_output_setup=tabular_output_setup,
    )

    test_configs = config.Configs(
        global_config=test_global_config,
        input_configs=test_input_configs,
        fusion_config=test_fusion_configs,
        output_configs=test_output_configs,
    )

    # This is done after in case tests modify output_folder
    output_folder = (
        test_configs.global_config.output_folder
        + "_"
        + "_".join(i.model_config.model_type for i in test_configs.input_configs)
        + "_"
        + f"{test_configs.output_configs[0].model_config.model_type}"
        + "_"
        + test_data_config.request_params["task_type"]
    )

    if not output_folder.startswith("runs/"):
        output_folder = "runs/" + output_folder

    test_configs.global_config.output_folder = output_folder

    run_folder = get_run_folder(output_folder=output_folder)

    # If another test had side-effect leftovers, TODO: Enforce unique names
    if run_folder.exists():
        cleanup(run_path=run_folder)

    ensure_path_exists(path=run_folder, is_folder=True)

    configure_root_logger(output_folder=output_folder)

    yield test_configs

    if not keep_outputs:
        cleanup(run_path=run_folder)


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
def create_test_model(
    create_test_config: config.Configs, create_test_labels
) -> nn.Module:
    gc = create_test_config.global_config
    target_labels = create_test_labels

    inputs_as_dict = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=create_test_config.input_configs,
        train_ids=tuple(create_test_labels.train_labels.keys()),
        valid_ids=tuple(create_test_labels.valid_labels.keys()),
        hooks=None,
    )

    input_model_registry = get_default_model_registry_per_input_type()

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    model = get_model(
        inputs_as_dict=inputs_as_dict,
        model_registry_per_input_type=input_model_registry,
        model_registry_per_output_type={},
        fusion_config=create_test_config.fusion_config,
        outputs_as_dict=outputs_as_dict,
        global_config=gc,
    )

    return model


def set_up_inputs_as_dict(input_configs: Sequence[schemas.InputConfig]):
    input_name_config_iter = eir.setup.input_setup.get_input_name_config_iterator(
        input_configs=input_configs
    )
    inputs_as_dict = {k: v for k, v in input_name_config_iter}
    return inputs_as_dict


def cleanup(run_path: Union[Path, str]) -> None:
    rmtree(path=run_path)


@pytest.fixture()
def create_test_labels(
    create_test_data, create_test_config: config.Configs
) -> train.MergedTargetLabels:

    c = create_test_config
    gc = c.global_config

    run_folder = get_run_folder(output_folder=gc.output_folder)

    all_array_ids = train.gather_all_ids_from_output_configs(
        output_configs=c.output_configs
    )
    train_ids, valid_ids = train.split_ids(ids=all_array_ids, valid_size=gc.valid_size)

    target_labels_info = train.get_tabular_target_file_infos(
        output_configs=c.output_configs
    )
    target_labels = train.set_up_tabular_target_labels_wrapper(
        tabular_target_file_infos=target_labels_info,
        custom_label_ops=None,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )

    train.save_transformer_set(
        transformers_per_source=target_labels.label_transformers, run_folder=run_folder
    )

    return target_labels


@pytest.fixture()
def create_test_datasets(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    configs = create_test_config
    target_labels = create_test_labels

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=configs.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    train_dataset, valid_dataset = datasets.set_up_datasets_from_configs(
        configs=configs,
        target_labels=target_labels,
        inputs_as_dict=inputs,
        outputs_as_dict=outputs_as_dict,
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
        valid_dataset, batch_size=gc.batch_size, shuffle=False, drop_last=False
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

    loss_module = train._get_loss_callable(criteria=criterions)

    optimizer = optimizers.get_optimizer(
        model=model, loss_callable=loss_module, global_config=global_config
    )

    return optimizer, loss_module


@dataclass
class ModelTestConfig:
    iteration: int
    run_path: Path
    last_sample_folders: Dict[str, Dict[str, Path]]
    activations_paths: Dict[str, Dict[str, Dict[str, Path]]]


@pytest.fixture()
def prep_modelling_test_configs(
    create_test_data,
    create_test_labels,
    create_test_config: config.Configs,
    create_test_dloaders,
    create_test_model,
    create_test_datasets,
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

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        target_transformers=target_labels.label_transformers,
    )

    criteria = train._get_criteria(outputs_as_dict=outputs_as_dict)
    test_metrics = metrics.get_default_metrics(
        target_transformers=target_labels.label_transformers,
    )
    test_metrics = _patch_metrics(metrics_=test_metrics)

    optimizer, loss_module = create_test_optimizer(
        global_config=gc,
        model=model,
        criterions=criteria,
    )

    train_dataset, valid_dataset = create_test_datasets

    train._log_model(model=model)

    inputs = eir.setup.input_setup.set_up_inputs_for_training(
        inputs_configs=c.input_configs,
        train_ids=tuple(target_labels.train_labels.keys()),
        valid_ids=tuple(target_labels.valid_labels.keys()),
        hooks=None,
    )
    run_folder = get_run_folder(output_folder=gc.output_folder)
    serialize_all_input_transformers(inputs_dict=inputs, run_folder=run_folder)
    serialize_chosen_input_objects(inputs_dict=inputs, run_folder=run_folder)

    hooks = train.get_default_hooks(configs=c)
    experiment = Experiment(
        configs=c,
        inputs=inputs,
        outputs=outputs_as_dict,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=valid_dataset,
        model=model,
        optimizer=optimizer,
        criteria=criteria,
        loss_function=loss_module,
        metrics=test_metrics,
        writer=train.get_summary_writer(run_folder=Path(gc.output_folder)),
        hooks=hooks,
    )

    keys_to_serialize = (
        eir.experiment_io.experiment_io.get_default_experiment_keys_to_serialize()
    )
    eir.experiment_io.experiment_io.serialize_experiment(
        experiment=experiment,
        run_folder=get_run_folder(gc.output_folder),
        keys_to_serialize=keys_to_serialize,
    )

    targets = config.get_all_tabular_targets(output_configs=c.output_configs)
    test_config = _get_cur_modelling_test_config(
        train_loader=train_loader,
        global_config=gc,
        targets=targets,
        input_names=inputs.keys(),
    )

    return experiment, test_config


def _patch_metrics(
    metrics_: metrics.al_metric_record_dict,
) -> metrics.al_metric_record_dict:
    warnings.warn(
        "This function will soon be deprecated as conftest will need to "
        "create its own metrics when train.py default metrics will be "
        "minimal.",
        category=DeprecationWarning,
    )
    for type_ in ("con",):
        for metric_record in metrics_[type_]:
            if metric_record.name == "r2":
                metric_record.only_val = False
    return metrics_


def _get_cur_modelling_test_config(
    train_loader: DataLoader,
    global_config: schemas.GlobalConfig,
    targets: config.TabularTargets,
    input_names: Iterable[str],
) -> ModelTestConfig:

    last_iter = len(train_loader) * global_config.n_epochs
    run_path = Path(f"{global_config.output_folder}/")

    last_sample_folders = _get_all_last_sample_folders(
        targets=targets, run_path=run_path, iteration=last_iter
    )

    all_activation_paths = _get_all_activation_paths(
        last_sample_folder_per_target_in_each_output=last_sample_folders,
        input_names=input_names,
    )

    test_config = ModelTestConfig(
        iteration=last_iter,
        run_path=run_path,
        last_sample_folders=last_sample_folders,
        activations_paths=all_activation_paths,
    )

    return test_config


def _get_all_activation_paths(
    last_sample_folder_per_target_in_each_output: Dict[str, Dict[str, Path]],
    input_names: Iterable[str],
) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    output_name -> target_name -> input_name: path
    """

    all_activation_paths = {}

    dict_to_iter = last_sample_folder_per_target_in_each_output
    for (output_name, file_per_target_dict) in dict_to_iter.items():
        if output_name not in all_activation_paths:
            all_activation_paths[output_name] = {}

        for target_name, last_sample_folder in file_per_target_dict.items():
            all_activation_paths[output_name][target_name] = {}

            for input_name in input_names:
                path = last_sample_folder / "activations" / input_name
                all_activation_paths[output_name][target_name][input_name] = path

    return all_activation_paths


def _get_all_last_sample_folders(
    targets: config.TabularTargets, run_path: Path, iteration: int
) -> Dict[str, Dict[str, Path]]:
    """
    output_name -> target_name: path
    """
    sample_folders = {}

    for output_name in targets.con_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_con_columns = targets.con_targets[output_name]
        for con_column_name in cur_con_columns:
            sample_folders[output_name][con_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                column_name=con_column_name,
            )

    for output_name in targets.cat_targets:
        if output_name not in sample_folders:
            sample_folders[output_name] = {}

        cur_cat_columns = targets.cat_targets[output_name]
        for cat_column_name in cur_cat_columns:
            sample_folders[output_name][cat_column_name] = _get_test_sample_folder(
                run_path=run_path,
                iteration=iteration,
                output_name=output_name,
                column_name=cat_column_name,
            )

    return sample_folders


def _get_test_sample_folder(
    run_path: Path, iteration: int, output_name: str, column_name: str
) -> Path:
    sample_folder = (
        run_path / f"results/{output_name}/{column_name}/samples/{iteration}"
    )

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
