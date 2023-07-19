from argparse import Namespace
from typing import TYPE_CHECKING

from eir.setup.config_validation import validate_input_configs, validate_output_configs

if TYPE_CHECKING:
    from eir.setup.config import Configs


def validate_predict_configs_and_args(
    predict_configs: "Configs", predict_cl_args: Namespace
) -> None:
    validate_input_configs(input_configs=predict_configs.input_configs)
    validate_output_configs(output_configs=predict_configs.output_configs)
    _validate_predict_cl_args_and_config_synergy(
        predict_configs=predict_configs, predict_cl_args=predict_cl_args
    )


def _validate_predict_cl_args_and_config_synergy(
    predict_cl_args: Namespace, predict_configs: "Configs"
) -> None:
    gc = predict_configs.global_config

    if gc.compute_attributions and not predict_cl_args.evaluate:
        raise ValueError(
            "When doing prediction, if compute_attributions is True, "
            "--evaluate in the eirpredict CL arguments must be used. "
            "This is because the currently the true labels are used "
            "to compute the attributions. "
        )
