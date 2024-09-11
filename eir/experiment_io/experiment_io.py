from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable, Union

import dill
from aislib.misc_utils import ensure_path_exists

from eir.experiment_io.configs_io import load_configs
from eir.experiment_io.input_object_io import check_version
from eir.experiment_io.label_transformer_io import load_transformers
from eir.setup.config import Configs
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.metrics import get_default_metrics
from eir.train_utils.step_logic import get_default_hooks
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.train_utils.metrics import al_metric_record_dict
    from eir.train_utils.step_logic import Hooks


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class LoadedTrainExperiment:
    configs: Configs
    hooks: "Hooks"
    metrics: "al_metric_record_dict"
    outputs: al_output_objects_as_dict


def load_serialized_train_experiment(run_folder: Path) -> LoadedTrainExperiment:
    check_version(run_folder=run_folder)

    train_experiment_path = get_train_experiment_serialization_path(
        run_folder=run_folder
    )
    with open(train_experiment_path, "rb") as infile:
        train_experiment_object = dill.load(file=infile)

    expected_keys = get_default_experiment_keys_to_serialize()
    train_experiment_as_dict = train_experiment_object.__dict__
    assert set(train_experiment_as_dict.keys()) == set(expected_keys)

    configs_folder = run_folder / "configs"
    configs_loaded = load_configs(configs_root_folder=configs_folder)
    train_experiment_as_dict["configs"] = configs_loaded

    hooks = get_default_hooks(configs=configs_loaded)
    train_experiment_as_dict["hooks"] = hooks

    transformers = load_transformers(run_folder=run_folder)
    gc = configs_loaded.global_config
    metrics = get_default_metrics(
        target_transformers=transformers,
        cat_metrics=gc.metrics.cat_metrics,
        con_metrics=gc.metrics.con_metrics,
        cat_averaging_metrics=gc.metrics.cat_averaging_metrics,
        con_averaging_metrics=gc.metrics.con_averaging_metrics,
    )
    train_experiment_as_dict["metrics"] = metrics

    loaded_experiment = LoadedTrainExperiment(**train_experiment_as_dict)

    return loaded_experiment


def serialize_experiment(
    experiment: "Experiment",
    run_folder: Path,
    keys_to_serialize: Union[Iterable[str], None],
) -> None:
    serialization_path = get_train_experiment_serialization_path(run_folder=run_folder)
    ensure_path_exists(path=serialization_path)

    filtered_experiment = filter_experiment_by_keys(
        experiment=experiment,
        keys=keys_to_serialize,
    )
    serialize_namespace(namespace=filtered_experiment, output_path=serialization_path)


def get_train_experiment_serialization_path(run_folder: Path) -> Path:
    train_experiment_path = run_folder / "serializations" / "filtered_experiment.dill"

    return train_experiment_path


def filter_experiment_by_keys(
    experiment: "Experiment", keys: Union[None, Iterable[str]] = None
) -> SimpleNamespace:
    filtered = {}

    config_fields = (f.name for f in fields(experiment))
    iterable = keys if keys is not None else config_fields

    for k in iterable:
        filtered[k] = getattr(experiment, k)

    namespace = SimpleNamespace(**filtered)

    return namespace


def serialize_namespace(namespace: SimpleNamespace, output_path: Path) -> None:
    with open(output_path, "wb") as outfile:
        dill.dump(namespace, outfile)


def get_default_experiment_keys_to_serialize() -> Iterable[str]:
    return ("outputs",)


def get_version_file(run_folder: Path) -> Path:
    return run_folder / "meta/eir_version.txt"
