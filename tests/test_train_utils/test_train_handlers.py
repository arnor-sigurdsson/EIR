from human_origins_supervised.train_utils import train_handlers


def test_unflatten_engine_metrics_dict():
    test_step_base = {
        "Origin": {"t_Origin_mcc": 0.9, "t_Origin_loss": 0.1},
        "Height": {"t_Height_pcc": 0.9, "t_Height_rmse": 0.1},
    }
    test_flat_metrics_dict = {
        "t_Origin_mcc": 0.99,
        "t_Origin_loss": 0.11,
        "t_Height_pcc": 0.99,
        "t_Height_rmse": 0.11,
    }

    test_output = train_handlers._unflatten_engine_metrics_dict(
        step_base=test_step_base, engine_metrics_dict=test_flat_metrics_dict
    )

    # we want to make sure the original values are present
    assert test_output["Origin"]["t_Origin_mcc"] == 0.99
    assert test_output["Height"]["t_Height_pcc"] == 0.99

    assert test_output["Origin"]["t_Origin_loss"] == 0.11
    assert test_output["Height"]["t_Height_rmse"] == 0.11


def test_generate_h_param_dict(args_config):
    test_h_params = ["lr", "na_augment_perc", "channel_exp_base"]
    test_h_dict = train_handlers._generate_h_param_dict(
        cl_args=args_config, h_params=test_h_params
    )

    assert test_h_dict["lr"] == 0.01
    assert test_h_dict["na_augment_perc"] == 0.0
    assert test_h_dict["channel_exp_base"] == 5
