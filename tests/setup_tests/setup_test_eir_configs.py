from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch.backends
import torch.cuda

from eir.setup.config_setup_modules.config_setup_utils import recursive_dict_inject
from tests.conftest import get_system_info


def get_test_base_global_init(
    allow_cuda: bool = True,
    allow_mps: bool = False,
) -> Sequence[dict]:
    device = "cpu"
    in_gha, _ = get_system_info()
    if allow_cuda:
        device = "cuda" if torch.cuda.is_available() else device
    if allow_mps and not in_gha:
        device = "mps" if torch.backends.mps.is_available() else device

    global_inits = [
        {
            "basic_experiment": {
                "output_folder": "runs/test_run",
                "device": device,
                "n_epochs": 12,
                "batch_size": 32,
                "valid_size": 0.05,
            },
            "visualization_logging": {
                "plot_skip_steps": 0,
            },
            "attribution_analysis": {
                "compute_attributions": True,
                "attributions_every_sample_factor": 0,
                "attribution_background_samples": 64,
            },
            "training_control": {
                "early_stopping_patience": 0,
            },
            "optimization": {
                "lr": 2e-03,
                "optimizer": "adabelief",
                "lr_lb": 1e-05,
                "wd": 1e-03,
            },
            "lr_schedule": {
                "warmup_steps": 100,
            },
            "accelerator": {
                "hardware": "cpu",
            },
        }
    ]
    return global_inits


def get_test_inputs_inits(
    test_path: Path,
    input_config_dicts: Sequence[dict],
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    extra_kwargs: dict | None = None,
) -> Sequence[dict]:
    if extra_kwargs is None:
        extra_kwargs = {}

    inits = []

    base_func_map = get_input_test_init_base_func_map()
    for init_dict in input_config_dicts:
        cur_name = init_dict["input_info"]["input_name"]

        cur_base_func_keys = [i for i in base_func_map if cur_name.startswith(i)]
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

        cur_init_injected = recursive_dict_inject(
            dict_=cur_init_base, dict_to_inject=init_dict
        )
        inits.append(cur_init_injected)

    return inits


def get_test_outputs_inits(
    test_path: Path,
    output_configs_dicts: Sequence[dict],
    split_to_test: bool,
    source: Literal["local", "deeplake"],
) -> Sequence[dict]:
    inits = []

    base_func_map = get_output_test_init_base_func_map()

    for init_dict in output_configs_dicts:
        cur_name = init_dict["output_info"]["output_name"]

        cur_base_func_keys = [i for i in base_func_map if cur_name.startswith(i)]
        assert len(cur_base_func_keys) == 1
        cur_base_func_key = cur_base_func_keys[0]

        cur_base_func = base_func_map.get(cur_base_func_key)
        cur_init_base = cur_base_func(
            test_path=test_path,
            split_to_test=split_to_test,
            source=source,
        )

        cur_init_injected = recursive_dict_inject(
            dict_=cur_init_base, dict_to_inject=init_dict
        )

        cur_model_type = cur_init_injected["model_config"].get(
            "model_type", "mlp_residual"
        )
        cur_model_config = _get_test_output_model_config(
            output_model_type=cur_model_type,
            cur_settings=cur_init_injected["model_config"],
        )
        cur_init_injected["model_config"] = cur_model_config

        inits.append(cur_init_injected)

    return inits


def get_output_test_init_base_func_map() -> dict[str, Callable]:
    mapping = {
        "test_output_tabular": get_test_tabular_base_output_inits,
        "test_output_copy": get_test_tabular_base_output_inits,
        "test_output_sequence": get_test_sequence_base_output_inits,
        "test_output_array": get_test_array_base_output_inits,
        "test_output_image": get_test_image_base_output_inits,
        "test_output_survival": get_test_survival_base_output_inits,
        # For diffusion compatibility
        "test_array": get_test_array_base_output_inits,
        "test_image": get_test_image_base_output_inits,
    }

    return mapping


def get_input_test_init_base_func_map() -> dict[str, Callable]:
    mapping = {
        "test_genotype": get_test_omics_input_init,
        "test_tabular": get_test_tabular_input_init,
        "test_sequence": get_test_sequence_input_init,
        "test_bytes": get_test_bytes_input_init,
        "test_image": get_test_image_input_init,
        "copy_test_image": get_test_image_input_init,
        "test_array": get_test_array_input_init,
        "copy_test_array": get_test_array_input_init,
    }

    return mapping


def _inject_train_source_path(
    test_path: Path,
    source: Literal["local", "deeplake"],
    local_name: Literal["omics", "sequence", "image", "array"],
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
    init_dict: dict,
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
            "na_augment_alpha": 1.0,
            "na_augment_beta": 5.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 20.0,
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
) -> dict:
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
) -> dict:
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
            "model_type": "cnn",
            "pretrained_model": False,
            "num_output_features": 0,
            "freeze_pretrained_model": False,
            "model_init_config": {
                "layers": [2],
            },
        },
    }

    return input_init_kwargs


def get_test_array_input_init(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
    *args,
    **kwargs,
) -> dict:
    input_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="array",
        split_to_test=split_to_test,
    )

    input_init_kwargs = {
        "input_info": {
            "input_source": str(input_source),
            "input_name": "test_array",
            "input_type": "array",
            "input_inner_key": "test_array",
        },
        "model_config": {"model_type": "cnn"},
    }

    return input_init_kwargs


def get_test_base_fusion_init(model_type: str) -> Sequence[dict]:
    if model_type in ("identity", "pass-through"):
        return [{}]
    elif model_type == "attention":
        return [
            {
                "model_config": {
                    "n_layers": 1,
                    "common_embedding_dim": 128,
                    "n_heads": 4,
                    "dim_feedforward": "auto",
                    "dropout": 0.1,
                }
            }
        ]
    elif model_type in ("mlp-residual", "mgmoe"):
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
    raise ValueError(f"Unknown fusion model type: '{model_type}'")


def get_test_tabular_base_output_inits(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> dict:
    label_file = test_path / "labels.csv"
    if split_to_test:
        label_file = test_path / "labels_train.csv"

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output_tabular",
            "output_type": "tabular",
            "output_source": str(label_file),
        },
        "output_type_info": {
            "target_cat_columns": ["Origin"],
        },
        "model_config": {},
    }

    return test_target_init_kwargs


def get_test_sequence_base_output_inits(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
) -> dict:
    output_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="sequence",
        split_to_test=split_to_test,
    )

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output_sequence",
            "output_type": "sequence",
            "output_source": str(output_source),
            "output_inner_key": "test_sequence",
        },
        "output_type_info": {
            "max_length": 32,
            "split_on": "",
            "sampling_strategy_if_longer": "uniform",
            "min_freq": 1,
        },
        "model_config": {"model_type": "sequence"},
    }

    return test_target_init_kwargs


def get_test_array_base_output_inits(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
) -> dict:
    output_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="array",
        split_to_test=split_to_test,
    )

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output_array",
            "output_type": "array",
            "output_source": str(output_source),
            "output_inner_key": "test_array",
        },
        "model_config": {"model_type": "lcl"},
        "sampling_config": {"diffusion_inference_steps": 50},
    }

    return test_target_init_kwargs


def get_test_image_base_output_inits(
    test_path: Path,
    split_to_test: bool,
    source: Literal["local", "deeplake"],
) -> dict:
    output_source = _inject_train_source_path(
        test_path=test_path,
        source=source,
        local_name="image",
        split_to_test=split_to_test,
    )

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output_image",
            "output_type": "image",
            "output_source": str(output_source),
            "output_inner_key": "test_image",
        },
        "model_config": {
            "model_type": "cnn",
        },
        "sampling_config": {"diffusion_inference_steps": 50},
    }

    return test_target_init_kwargs


def get_test_survival_base_output_inits(
    test_path: Path, split_to_test: bool, *args, **kwargs
) -> dict:
    label_file = test_path / "labels.csv"
    if split_to_test:
        label_file = test_path / "labels_train.csv"

    test_target_init_kwargs = {
        "output_info": {
            "output_name": "test_output_tabular",
            "output_type": "survival",
            "output_source": str(label_file),
        },
        "output_type_info": {
            "event_column": "Origin",
            "time_column": "Height",
        },
        "model_config": {},
    }

    return test_target_init_kwargs


def _get_test_output_model_config(
    output_model_type: Literal["mlp_residual", "linear"], cur_settings: dict
) -> dict:
    """
    Done just to have a default model config for the output model
    if nothing is specified.
    """
    base = {
        "model_type": output_model_type,
    }

    match output_model_type:
        case "mlp_residual":
            base["model_init_config"] = {
                "layers": [1],
                "fc_task_dim": 128,
            }

        case _:
            return cur_settings

    return base
