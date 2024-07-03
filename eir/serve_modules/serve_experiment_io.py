from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

from eir.experiment_io.experiment_io import (
    LoadedTrainExperiment,
    load_serialized_train_experiment,
)
from eir.models import model_setup
from eir.models.meta.meta import MetaModel
from eir.serve_modules.serve_input_setup import set_up_inputs_for_serve
from eir.setup.config import Configs
from eir.setup.input_setup import al_input_objects_as_dict
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train import Hooks, al_output_objects_as_dict
    from eir.train_utils.metrics import al_metric_record_dict


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass(frozen=True)
class ServeExperiment:
    configs: Configs
    hooks: Union["Hooks", None]
    metrics: "al_metric_record_dict"
    inputs: al_input_objects_as_dict
    outputs: "al_output_objects_as_dict"
    model: MetaModel


def load_experiment_for_serve(
    model_path: str,
    device: str,
) -> ServeExperiment:
    model_path_object = Path(model_path)
    run_folder = model_path_object.parent.parent

    loaded_train_experiment = load_serialized_train_experiment(run_folder=run_folder)

    default_train_hooks = loaded_train_experiment.hooks
    train_configs = loaded_train_experiment.configs

    loaded_train_experiment.configs.global_config.device = device

    inputs = set_up_inputs_for_serve(
        test_inputs_configs=train_configs.input_configs,
        hooks=default_train_hooks,
        output_folder=str(run_folder),
    )

    logger.info("Loading EIR PyTorch model from '%s'.", model_path)

    model = load_pytorch_eir_model_for_serve(
        model_pt_path=str(model_path),
        loaded_train_experiment=loaded_train_experiment,
        inputs=inputs,
        device=device,
    )

    assert not model.training

    loaded_train_experiment_as_dict = loaded_train_experiment.__dict__
    serve_experiment_kwargs = {
        **{"model": model, "inputs": inputs},
        **loaded_train_experiment_as_dict,
    }

    serve_experiment = ServeExperiment(**serve_experiment_kwargs)

    return serve_experiment


def load_pytorch_eir_model_for_serve(
    model_pt_path: str,
    loaded_train_experiment: LoadedTrainExperiment,
    inputs: al_input_objects_as_dict,
    device: str,
) -> model_setup.al_meta_model:
    func = model_setup.get_meta_model_class_and_kwargs_from_configs
    meta_model_class, meta_model_kwargs = func(
        global_config=loaded_train_experiment.configs.global_config,
        fusion_config=loaded_train_experiment.configs.fusion_config,
        inputs_as_dict=inputs,
        outputs_as_dict=loaded_train_experiment.outputs,
    )

    model = model_setup.load_model(
        model_path=Path(model_pt_path),
        model_class=meta_model_class,
        model_init_kwargs=meta_model_kwargs,
        device=device,
        test_mode=True,
        strict_shapes=True,
    )
    assert not model.training

    for name, _ in model.named_modules():
        if "class" in name.split("."):
            raise NameError(
                f"'class' is a reserved keyword and cannot be part of the "
                f"modules in model. Please ensure no module is named 'class'. "
                f"Found in '{name}'."
            )

    return model
