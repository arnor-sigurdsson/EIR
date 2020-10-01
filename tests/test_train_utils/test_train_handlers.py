from snp_pred.train_utils import train_handlers


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


def test_generate_h_param_dict(args_config):
    args_config.layers = [2, 2, 4]
    test_h_params = ["lr", "na_augment_perc", "channel_exp_base", "layers"]
    test_h_dict = train_handlers._generate_h_param_dict(
        cl_args=args_config, h_params=test_h_params
    )

    assert test_h_dict["lr"] == args_config.lr
    assert test_h_dict["na_augment_perc"] == args_config.na_augment_perc
    assert test_h_dict["channel_exp_base"] == args_config.channel_exp_base
    assert test_h_dict["layers"] == "2_2_4"
