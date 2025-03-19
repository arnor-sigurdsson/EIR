from typing import TYPE_CHECKING

import pytest

from eir import train
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_experiment import ModelTestConfig


def get_parametrization(memory_dataset):
    params = [get_base_parametrization(compiled=False, memory_dataset=memory_dataset)]

    return params


def get_base_parametrization(
    memory_dataset: bool,
    compiled: bool = False,
) -> dict:
    params = {
        "injections": {
            "global_configs": {
                "basic_experiment": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 12,
                    "memory_dataset": memory_dataset,
                },
                "model": {
                    "compile_model": compiled,
                },
                "optimization": {
                    "gradient_clipping": 1.0,
                    "lr": 0.002,
                },
            },
            "input_configs": [
                {
                    "input_info": {"input_name": "test_genotype"},
                    "model_config": {
                        "model_type": "genome-local-net",
                        "model_init_config": {"l1": 1e-06},
                    },
                },
                {
                    "input_info": {"input_name": "test_sequence"},
                },
                {
                    "input_info": {"input_name": "test_bytes"},
                },
                {
                    "input_info": {"input_name": "test_array"},
                    "input_type_info": {
                        "normalization": None,
                    },
                    "model_config": {
                        "model_type": "lcl",
                    },
                },
                {
                    "input_info": {"input_name": "test_image"},
                    "model_config": {
                        "model_init_config": {
                            "layers": [2],
                            "kernel_width": 2,
                            "kernel_height": 2,
                            "down_stride_width": 2,
                            "down_stride_height": 2,
                        },
                    },
                },
                {
                    "input_info": {"input_name": "test_tabular"},
                    "input_type_info": {
                        "input_cat_columns": ["OriginExtraCol"],
                        "input_con_columns": ["ExtraTarget"],
                    },
                    "model_config": {
                        "model_type": "tabular",
                        "model_init_config": {"l1": 1e-04},
                    },
                },
            ],
            "fusion_configs": {
                "model_type": "attention",
            },
            "output_configs": [
                {
                    "output_info": {"output_name": "test_output_copy"},
                    "output_type_info": {
                        "target_cat_columns": [],
                        "target_con_columns": ["Height"],
                    },
                },
                {
                    "output_info": {"output_name": "test_output_tabular"},
                    "output_type_info": {
                        "target_cat_columns": ["Origin"],
                        "target_con_columns": ["Height"],
                    },
                },
                {
                    "output_info": {
                        "output_name": "test_output_sequence",
                    },
                    "model_config": {
                        "model_init_config": {
                            "dropout": 0.0,
                        }
                    },
                },
                {
                    "output_info": {
                        "output_name": "test_output_array_cnn",
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 3,
                            "allow_pooling": False,
                        },
                    },
                },
                {
                    "output_info": {
                        "output_name": "test_output_image",
                    },
                    "output_type_info": {
                        "loss": "mse",
                        "size": [16, 16],
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 4,
                            "allow_pooling": False,
                        },
                    },
                },
                {
                    "output_info": {"output_name": "test_output_survival"},
                    "output_type_info": {
                        "event_column": "BinaryOrigin",
                        "time_column": "Time",
                    },
                },
            ],
        },
    }

    return params


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "random_samples_dropped_from_modalities": True,
            "source": "local",
        },
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "random_samples_dropped_from_modalities": False,
            "source": "deeplake",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    get_parametrization(memory_dataset=False),
    indirect=True,
)
def test_multi_modal_multi_task_disk(
    prep_modelling_test_configs: tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.80, 0.80),
        min_thresholds=(2.0, 2.0),
    )


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "random_samples_dropped_from_modalities": True,
            "source": "local",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    get_parametrization(memory_dataset=True),
    indirect=True,
)
def test_multi_modal_multi_task_memory(
    prep_modelling_test_configs: tuple[train.Experiment, "ModelTestConfig"],
):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.80, 0.80),
        min_thresholds=(2.0, 2.0),
    )
