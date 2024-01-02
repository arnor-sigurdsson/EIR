from copy import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

import dill
import joblib
from aislib.misc_utils import ensure_path_exists

from eir import __version__
from eir.data_load import label_setup
from eir.data_load.label_setup import (
    al_label_transformers,
    al_label_transformers_object,
)
from eir.setup import schemas
from eir.setup.config import Configs
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train_utils.utils import get_run_folder
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.setup.input_setup import (
        al_input_objects_as_dict,
        al_serializable_input_classes,
        al_serializable_input_objects,
    )
    from eir.train import Experiment
    from eir.train_utils.metrics import al_metric_record_dict
    from eir.train_utils.step_logic import Hooks


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
    hooks: "Hooks"
    metrics: "al_metric_record_dict"
    outputs: al_output_objects_as_dict


def load_serialized_train_experiment(run_folder: Path) -> LoadedTrainExperiment:
    check_version(run_folder=run_folder)

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
        assert output_folder is not None
        run_folder = get_run_folder(output_folder=output_folder)

    check_version(run_folder=run_folder)
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

        if _should_skip_warning(
            key=key,
            current_value=current_value,
            loaded_value=loaded_value,
        ):
            continue

        if current_value != loaded_value:
            logger.warning(
                "Expected '%s' to be the same in current input configuration"
                "\n'%s'\n and "
                "loaded input configuration \n'%s'\n "
                "(loaded from '%s'). If you are loading"
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


def _should_skip_warning(key: str, current_value: Any, loaded_value: Any) -> bool:
    if key != "model_config":
        return False

    current_dict = asdict(current_value)
    loaded_dict = asdict(loaded_value)

    differing_keys = [k for k, v in current_dict.items() if loaded_dict.get(k) != v]

    return (
        len(differing_keys) == 1
        and differing_keys[0] == "model_type"
        and "linked" in loaded_dict["model_type"]
    )


def serialize_chosen_input_objects(
    inputs_dict: "al_input_objects_as_dict", run_folder: Path
) -> None:
    targets_to_serialize = {"sequence", "bytes", "image", "array"}
    for input_name, input_ in inputs_dict.items():
        input_type = input_.input_config.input_info.input_type

        any_match = any(i for i in targets_to_serialize if input_type == i)

        if any_match:
            output_path = get_input_serialization_path(
                run_folder=run_folder,
                input_type=input_type,
                input_name=input_name,
            )
            ensure_path_exists(path=output_path, is_folder=False)
            with open(output_path, "wb") as outfile:
                dill.dump(obj=input_, file=outfile)


def get_transformer_sources(run_folder: Path) -> Dict[str, list[str]]:
    transformers_to_load = {}
    transformer_sources = run_folder / "serializations/transformers"
    for transformer_source in transformer_sources.iterdir():
        names = sorted([i.stem for i in transformer_source.iterdir()])
        transformers_to_load[transformer_source.stem] = names
    return transformers_to_load


def load_transformer(
    run_folder: Path,
    source_name: str,
    transformer_name: str,
) -> al_label_transformers_object:
    target_transformer_path = label_setup.get_transformer_path(
        run_path=run_folder,
        transformer_name=transformer_name,
        source_name=source_name,
    )
    return joblib.load(filename=target_transformer_path)


def load_transformers(
    transformers_to_load: Optional[Dict[str, list[str]]] = None,
    output_folder: Optional[str] = None,
    run_folder: Optional[Path] = None,
) -> Dict[str, al_label_transformers]:
    if not run_folder and not output_folder:
        raise ValueError("Either 'run_folder' or 'output_folder' must be provided.")

    if not run_folder:
        assert output_folder is not None
        run_folder = get_run_folder(output_folder=output_folder)

    if not transformers_to_load:
        transformers_to_load = get_transformer_sources(run_folder=run_folder)

    loaded_transformers: Dict[str, al_label_transformers] = {}
    for source_name, source_transformers_to_load in transformers_to_load.items():
        loaded_transformers[source_name] = {}
        for transformer_name in source_transformers_to_load:
            loaded_transformers[source_name][transformer_name] = load_transformer(
                run_folder=run_folder,
                source_name=source_name,
                transformer_name=transformer_name,
            )

    return loaded_transformers


def check_version(run_folder: Path) -> None:
    version_file = run_folder / "meta/eir_version.txt"
    if not version_file.exists():
        return

    cur_version = __version__
    with open(version_file, "r") as f:
        loaded_version = f.read().strip()

    if cur_version != loaded_version:
        logger.warning(
            f"The version of EIR used to train this model is {loaded_version}, "
            f"while the current version is {cur_version}. "
            f"This may cause unexpected behavior and subtle bugs."
        )


def get_version_file(run_folder: Path) -> Path:
    return run_folder / "meta/eir_version.txt"
