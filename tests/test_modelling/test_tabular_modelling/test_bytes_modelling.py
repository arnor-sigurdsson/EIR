import pytest

from eir import train
from eir.train_utils.utils import seed_everything
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper

seed_everything(seed=0)


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("sequence",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        # Case 1: Classification - Basic Transformer
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_classification_vanilla_"
                        "transformer_bytes",
                        "n_epochs": 12,
                        "memory_dataset": True,
                    },
                    "training_control": {
                        "early_stopping_patience": 5,
                        "mixing_alpha": 0.1,
                    },
                    "attribution_analysis": {
                        "attribution_background_samples": 8,
                    },
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_bytes"},
                        "input_type_info": {
                            "max_length": 128,
                        },
                        "model_config": {
                            "position": "embed",
                            "window_size": 64,
                            "model_type": "sequence-default",
                            "model_init_config": {
                                "num_heads": 2,
                                "num_layers": 2,
                                "dropout": 0.10,
                            },
                        },
                    }
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    }
                ],
            },
        },
    ],
    indirect=True,
)
def test_bytes_modelling(prep_modelling_test_configs):
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    thresholds = get_bytes_test_args(
        mixing=experiment.configs.gc.training_control.mixing_alpha
    )

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=thresholds,
    )


def get_bytes_test_args(mixing: float) -> tuple[float, float]:
    thresholds = (0.7, 0.6)
    if mixing:
        thresholds = (0.0, 0.6)

    return thresholds
