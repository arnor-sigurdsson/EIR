import pytest

from eir.train_utils import train_handlers


def test_unflatten_engine_metrics_dict():
    test_step_base = {
        "Origin": {"Origin_mcc": 0.9, "Origin_loss": 0.1},
        "Height": {"Height_pcc": 0.9, "Height_rmse": 0.1},
    }
    test_flat_metrics_dict = {
        "Origin_mcc": 0.99,
        "Origin_loss": 0.11,
        "Height_pcc": 0.99,
        "Height_rmse": 0.11,
    }

    test_output = train_handlers._unflatten_engine_metrics_dict(
        step_base=test_step_base, engine_metrics_dict=test_flat_metrics_dict
    )

    # we want to make sure the original values are present
    assert test_output["Origin"]["Origin_mcc"] == 0.99
    assert test_output["Height"]["Height_pcc"] == 0.99

    assert test_output["Origin"]["Origin_loss"] == 0.11
    assert test_output["Height"]["Height_rmse"] == 0.11


@pytest.mark.parametrize("create_test_data", [{"task_type": "binary"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {"lr_schedule": "plateau", "lr": 1e-03},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "input_type_info": {"model_type": "identity"},
                    },
                ],
            }
        }
    ],
    indirect=True,
)
def test_generate_h_param_dict(create_test_config):

    test_configs = create_test_config
    gc = test_configs.global_config

    test_h_params = [
        "lr",
    ]
    test_h_dict = train_handlers._generate_h_param_dict(
        global_config=gc, h_params=test_h_params
    )

    assert test_h_dict["lr"] == gc.lr
