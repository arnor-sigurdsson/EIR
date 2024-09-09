import json
from copy import copy, deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import dill
from aislib.misc_utils import ensure_path_exists

from eir import __version__
from eir.experiment_io.input_object_io_modules.input_io_utils import (
    load_input_config_from_yaml,
)
from eir.experiment_io.input_object_io_modules.sequence_input_io import (
    load_sequence_input_object,
)
from eir.experiment_io.io_utils import (
    dump_config_to_yaml,
    load_dataclass,
    save_dataclass,
)
from eir.setup import schemas
from eir.setup.input_setup_modules.setup_array import ComputedArrayInputInfo
from eir.setup.input_setup_modules.setup_bytes import ComputedBytesInputInfo
from eir.setup.input_setup_modules.setup_image import (
    ComputedImageInputInfo,
    ImageNormalizationStats,
    set_up_computed_image_input_object,
)
from eir.setup.input_setup_modules.setup_sequence import ComputedSequenceInputInfo
from eir.train_utils.utils import get_logger, get_run_folder

if TYPE_CHECKING:
    from eir.setup.input_setup import (
        al_input_objects_as_dict,
        al_serializable_input_classes,
        al_serializable_input_objects,
    )


logger = get_logger(name=__name__)


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
    serialized_input_config_object = _read_serialized_input_object(
        input_class=input_class,
        serialized_input_config_path=serialized_input_config_path,
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
    run_folder: Path,
    input_type: str,
    input_name: str,
) -> Path:

    base_path = run_folder / "serializations" / f"{input_type}_input_serializations"
    match input_type:
        case "image" | "sequence":
            path = base_path / f"{input_name}/"
        case "bytes" | "array":
            path = base_path / f"{input_name}.dill"
        case _:
            raise ValueError(f"Invalid input type: {input_type}")

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

            assert isinstance(
                input_,
                (
                    ComputedImageInputInfo,
                    ComputedSequenceInputInfo,
                    ComputedBytesInputInfo,
                    ComputedArrayInputInfo,
                ),
            )

            ensure_path_exists(path=output_path, is_folder=output_path.is_dir())
            _serialize_input_object(
                input_object=input_,
                output_folder=output_path,
            )


def _serialize_input_object(
    input_object: "al_serializable_input_objects",
    output_folder: Path,
) -> None:

    input_config = input_object.input_config
    config_path = output_folder / "input_config.yaml"

    match input_object:
        case ComputedImageInputInfo():
            dump_config_to_yaml(config=input_config, output_path=config_path)

            save_dataclass(
                obj=input_object.normalization_stats,
                file_path=output_folder / "normalization_stats.json",
            )

            num_channels_obj = {"num_channels": input_object.num_channels}
            with open(output_folder / "num_channels.json", "w") as f:
                json.dump(num_channels_obj, f)

        case ComputedSequenceInputInfo():
            dump_config_to_yaml(config=input_config, output_path=config_path)

            index_to_string = input_object.vocab.itos
            with open(output_folder / "vocab_ordered.txt", "w") as f:
                for token in index_to_string:
                    f.write(f"{token}\n")

            string_to_index = input_object.vocab.stoi
            with open(output_folder / "vocab.json", "w") as f:
                json.dump(string_to_index, f)

            computed_max_length = input_object.computed_max_length
            with open(output_folder / "computed_max_length.json", "w") as f:
                json.dump(computed_max_length, f)

        case ComputedBytesInputInfo() | ComputedArrayInputInfo():
            with open(output_folder, "wb") as outfile:
                dill.dump(obj=input_object, file=outfile)


def _read_serialized_input_object(
    input_class: "al_serializable_input_classes",
    serialized_input_config_path: Path,
) -> "al_serializable_input_objects":
    loaded_object: "al_serializable_input_objects"
    if input_class is ComputedImageInputInfo:
        base_path = serialized_input_config_path
        config_path = base_path / "input_config.yaml"
        normalization_stats_path = base_path / "normalization_stats.json"
        num_channels_path = base_path / "num_channels.json"

        input_config = load_input_config_from_yaml(input_config_path=config_path)
        input_type_info_modified = deepcopy(input_config.input_type_info)
        assert isinstance(input_type_info_modified, schemas.ImageInputDataConfig)

        normalization_stats = load_dataclass(
            cls=ImageNormalizationStats, file_path=normalization_stats_path
        )
        num_channels = json.loads(num_channels_path.read_text())["num_channels"]

        input_config_modified = deepcopy(input_config)
        input_type_info_modified.num_channels = num_channels
        input_config.input_type_info = input_type_info_modified

        loaded_object = set_up_computed_image_input_object(
            input_config=input_config_modified,
            normalization_stats=normalization_stats,
        )

    elif input_class is ComputedSequenceInputInfo:
        loaded_object = load_sequence_input_object(
            serialized_input_folder=serialized_input_config_path
        )

    elif input_class in (
        ComputedSequenceInputInfo,
        ComputedBytesInputInfo,
        ComputedArrayInputInfo,
    ):

        with open(serialized_input_config_path, "rb") as infile:
            loaded_object = dill.load(file=infile)
    else:
        raise ValueError(f"Invalid input class: {input_class}")

    return loaded_object


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
