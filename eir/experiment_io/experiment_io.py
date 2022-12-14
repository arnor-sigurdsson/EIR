from copy import copy
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Union, TYPE_CHECKING, Iterable, Sequence, Dict

import dill
import joblib
from aislib.misc_utils import get_logger, ensure_path_exists

from eir.data_load import label_setup
from eir.data_load.label_setup import (
    al_label_transformers,
    save_transformer_set,
)
from eir.setup import schemas
from eir.setup.config import Configs
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.utils import get_run_folder

if TYPE_CHECKING:
    from eir.train import Hooks, Experiment
    from eir.setup.input_setup import (
        al_input_objects_as_dict,
        al_serializable_input_objects,
        al_serializable_input_classes,
    )
    from eir.train_utils.metrics import al_metric_record_dict


logger = get_logger(name=__name__, tqdm_compatible=True)


def get_run_folder_from_model_path(model_path: str) -> Path:
    model_path_object = Path(model_path)
    assert model_path_object.exists()

    run_folder = model_path_object.parents[1]
    assert run_folder.exists()

    return run_folder


@dataclass
class LoadedTrainExperiment:
    configs: Configs
    hooks: Union["Hooks", None]
    metrics: "al_metric_record_dict"
    outputs: al_output_objects_as_dict


def load_serialized_train_experiment(run_folder: Path) -> LoadedTrainExperiment:
    train_config_path = get_train_experiment_serialization_path(run_folder=run_folder)
    with open(train_config_path, "rb") as infile:
        train_config = dill.load(file=infile)

    expected_keys = get_default_experiment_keys_to_serialize()
    train_config_as_dict = train_config.__dict__
    assert set(train_config_as_dict.keys()) == set(expected_keys)

    loaded_experiment = LoadedTrainExperiment(**train_config_as_dict)

    return loaded_experiment


def serialize_experiment(
    experiment: "Experiment",
    run_folder: Path,
    keys_to_serialize: Union[Iterable[str], None],
) -> None:
    serialization_path = get_train_experiment_serialization_path(run_folder=run_folder)
    ensure_path_exists(path=serialization_path)

    filtered_experiment = filter_experiment_by_keys(
        experiment=experiment, keys=keys_to_serialize
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
    return (
        "configs",
        "outputs",
        "metrics",
        "hooks",
    )


def load_serialized_input_object(
    input_config: schemas.InputConfig,
    input_class: "al_serializable_input_classes",
    *args,
    output_folder: Union[None, str] = None,
    run_folder: Union[None, Path] = None,
    custom_input_name: Union[str, None] = None,
    **kwargs,
) -> "al_serializable_input_objects":

    assert output_folder or run_folder
    if not run_folder:
        run_folder = get_run_folder(output_folder=output_folder)

    input_name = input_config.input_info.input_name
    input_type = input_config.input_info.input_type

    if custom_input_name:
        input_name = custom_input_name

    serialized_input_config_path = get_input_serialization_path(
        run_folder=run_folder,
        input_name=input_name,
        input_type=input_type,
    )

    assert serialized_input_config_path.exists(), serialized_input_config_path
    with open(serialized_input_config_path, "rb") as infile:
        serialized_input_config_object: "al_serializable_input_objects" = dill.load(
            file=infile
        )

    assert isinstance(serialized_input_config_object, input_class)

    train_input_info_kwargs = serialized_input_config_object.__dict__
    assert "input_config" in train_input_info_kwargs.keys()

    loaded_input_info_kwargs = copy(train_input_info_kwargs)

    _check_current_and_loaded_input_config_compatibility(
        current_input_config=input_config,
        loaded_input_config=serialized_input_config_object.input_config,
        serialized_input_config_path=serialized_input_config_path,
    )
    loaded_input_info_kwargs["input_config"] = input_config

    loaded_input_object = input_class(**loaded_input_info_kwargs)

    return loaded_input_object


def get_input_serialization_path(
    run_folder: Path, input_type: str, input_name: str
) -> Path:
    path = (
        run_folder
        / "serializations"
        / f"{input_type}_input_serializations/{input_name}.dill"
    )

    return path


def _check_current_and_loaded_input_config_compatibility(
    current_input_config: schemas.InputConfig,
    loaded_input_config: schemas.InputConfig,
    serialized_input_config_path: Path,
) -> None:

    fieldnames = current_input_config.__dict__.keys()
    assert set(fieldnames) == set(loaded_input_config.__dict__.keys())

    should_be_same = ("model_config",)

    for key in should_be_same:

        current_value = getattr(current_input_config, key)
        loaded_value = getattr(loaded_input_config, key)

        if current_value != loaded_value:

            logger.warning(
                "Expected '%s' to be the same in current input configuration '%s' and "
                "loaded input configuration '%s' (loaded from '%s'). If you are loading"
                " a pretrained EIR module, this can be expected if you are changing "
                "parameters that are expected to be agnostic across runs (e.g. dropout)"
                ", but in many cases this will cause (a) the model you are trying to "
                "load and (b) the model you are setting up for the current experiment "
                "to diverge, which will most likely lead to a RuntimeError. The "
                "resolution is likely to ensure that the input configurations of "
                "(a) and (b) are exactly the same when it comes to model "
                "configurations, which should ensure that the model architectures "
                "match.",
                key,
                current_value,
                loaded_value,
                serialized_input_config_path,
            )


def serialize_all_input_transformers(
    inputs_dict: "al_input_objects_as_dict", run_folder: Path
):
    for input_name, input_ in inputs_dict.items():
        input_type = input_.input_config.input_info.input_type
        if input_type == "tabular":
            save_transformer_set(
                transformers_per_source={input_name: input_.labels.label_transformers},
                run_folder=run_folder,
            )


def serialize_chosen_input_objects(
    inputs_dict: "al_input_objects_as_dict", run_folder: Path
):
    targets_to_serialize = {"sequence", "bytes", "image"}
    for input_name, input_ in inputs_dict.items():
        input_type = input_.input_config.input_info.input_type

        any_match = any(i for i in targets_to_serialize if input_type == i)

        if any_match:
            outpath = get_input_serialization_path(
                run_folder=run_folder,
                input_type=input_type,
                input_name=input_name,
            )
            ensure_path_exists(path=outpath, is_folder=False)
            with open(outpath, "wb") as outfile:
                dill.dump(obj=input_, file=outfile)


def load_transformers(
    transformers_to_load: Union[Dict[str, Sequence[str]], None],
    output_folder: Union[str, None] = None,
    run_folder: Union[None, Path] = None,
) -> Dict[str, al_label_transformers]:

    assert run_folder or output_folder

    if not run_folder:
        run_folder = get_run_folder(output_folder=output_folder)

    if not transformers_to_load:
        transformers_to_load = {}
        transformer_sources = run_folder / "serializations/transformers"
        for transformer_source in transformer_sources.iterdir():
            transformers_to_load[transformer_source.stem] = [
                i.stem for i in transformer_source.iterdir()
            ]

    loaded_transformers = {}

    for source_name, source_transformers_to_load in transformers_to_load.items():
        loaded_transformers[source_name] = {}

        for transformer_name in source_transformers_to_load:
            target_transformer_path = label_setup.get_transformer_path(
                run_path=run_folder,
                transformer_name=transformer_name,
                source_name=source_name,
            )
            target_transformer_object = joblib.load(filename=target_transformer_path)
            loaded_transformers[source_name][
                transformer_name
            ] = target_transformer_object

    return loaded_transformers
