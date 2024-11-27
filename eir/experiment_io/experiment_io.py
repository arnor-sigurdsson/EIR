from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from eir.experiment_io.configs_io import load_configs
from eir.experiment_io.io_utils import check_version
from eir.experiment_io.label_transformer_io import load_transformers
from eir.experiment_io.output_object_io import load_all_serialized_output_objects
from eir.setup.config import Configs
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.metrics import get_default_metrics
from eir.train_utils.step_logic import get_default_hooks
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train_utils.metrics import al_metric_record_dict
    from eir.train_utils.step_logic import Hooks


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class LoadedTrainExperiment:
    configs: Configs
    hooks: "Hooks"
    metrics: "al_metric_record_dict"
    outputs: al_output_objects_as_dict


def load_serialized_train_experiment(
    run_folder: Path,
    device: str,
    source_folder: Literal["configs", "configs_stripped"] = "configs_stripped",
) -> LoadedTrainExperiment:
    check_version(run_folder=run_folder)

    if source_folder == "configs":
        configs_folder = run_folder / "configs"
    elif source_folder == "configs_stripped":
        configs_folder = run_folder / "serializations" / "configs_stripped"
    else:
        raise ValueError(
            f"source_folder must be one of 'configs' or 'configs_stripped', "
            f"not {source_folder}"
        )

    configs_loaded = load_configs(configs_root_folder=configs_folder)

    configs_loaded.global_config.be.device = device

    hooks = get_default_hooks(configs=configs_loaded)

    transformers = load_transformers(run_folder=run_folder)
    gc = configs_loaded.global_config
    metrics = get_default_metrics(
        target_transformers=transformers,
        cat_metrics=gc.metrics.cat_metrics,
        con_metrics=gc.metrics.con_metrics,
        cat_averaging_metrics=gc.metrics.cat_averaging_metrics,
        con_averaging_metrics=gc.metrics.con_averaging_metrics,
        output_configs=configs_loaded.output_configs,
    )

    output_objects_as_dict = load_all_serialized_output_objects(
        output_configs=configs_loaded.output_configs,
        run_folder=run_folder,
    )

    loaded_experiment = LoadedTrainExperiment(
        configs=configs_loaded,
        hooks=hooks,
        metrics=metrics,
        outputs=output_objects_as_dict,
    )

    return loaded_experiment


def get_version_file(run_folder: Path) -> Path:
    return run_folder / "meta/eir_version.txt"
