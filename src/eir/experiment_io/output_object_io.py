import json
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

from aislib.misc_utils import ensure_path_exists

from eir.experiment_io.io_utils import (
    check_version,
    dump_config_to_yaml,
    save_dataclass,
)
from eir.experiment_io.output_object_io_modules.array_output_io import (
    load_array_output_object,
)
from eir.experiment_io.output_object_io_modules.image_output_io import (
    load_image_output_object,
)
from eir.experiment_io.output_object_io_modules.sequence_output_io import (
    load_sequence_output_object,
)
from eir.experiment_io.output_object_io_modules.survival_output_io import (
    load_survival_output_object,
)
from eir.experiment_io.output_object_io_modules.tabular_output_io import (
    load_tabular_output_object,
)
from eir.setup import schemas
from eir.setup.input_setup_modules.setup_sequence import (
    extract_tokenizer_object_from_function,
)
from eir.setup.output_setup_modules.array_output_setup import ComputedArrayOutputInfo
from eir.setup.output_setup_modules.image_output_setup import ComputedImageOutputInfo
from eir.setup.output_setup_modules.sequence_output_setup import (
    ComputedSequenceOutputInfo,
)
from eir.setup.output_setup_modules.survival_output_setup import (
    ComputedSurvivalOutputInfo,
)
from eir.setup.output_setup_modules.tabular_output_setup import (
    ComputedTabularOutputInfo,
)

if TYPE_CHECKING:
    from eir.setup.output_setup import (
        al_output_classes,
        al_output_objects,
        al_output_objects_as_dict,
    )


def load_all_serialized_output_objects(
    output_configs: Sequence[schemas.OutputConfig],
    run_folder: Path,
) -> "al_output_objects_as_dict":
    output_objects_as_dict = {}

    output_class: al_output_classes
    for output_config in output_configs:
        output_type = output_config.output_info.output_type
        match output_type:
            case "tabular":
                output_class = ComputedTabularOutputInfo
            case "sequence":
                output_class = ComputedSequenceOutputInfo
            case "image":
                output_class = ComputedImageOutputInfo
            case "array":
                output_class = ComputedArrayOutputInfo
            case "survival":
                output_class = ComputedSurvivalOutputInfo
            case _:
                raise ValueError(f"Invalid output type: {output_type}")

        output_object = load_serialized_output_object(
            output_config=output_config,
            output_class=output_class,
            run_folder=run_folder,
        )
        output_objects_as_dict[output_config.output_info.output_name] = output_object

    return output_objects_as_dict


def load_serialized_output_object(
    output_config: schemas.OutputConfig,
    output_class: "al_output_classes",
    run_folder: Path,
) -> "al_output_objects":
    check_version(run_folder=run_folder)

    output_name = output_config.output_info.output_name
    output_type = output_config.output_info.output_type

    serialized_output_folder = get_output_serialization_path(
        run_folder=run_folder,
        output_type=output_type,
        output_name=output_name,
    )

    assert serialized_output_folder.exists(), serialized_output_folder
    serialized_output_config_object = _read_serialized_output_object(
        output_class=output_class,
        serialized_output_folder=serialized_output_folder,
        run_folder=run_folder,
    )
    assert isinstance(serialized_output_config_object, output_class)

    output_info_kwargs = serialized_output_config_object.__dict__
    assert "output_config" in output_info_kwargs

    loaded_kwargs = copy(output_info_kwargs)
    loaded_kwargs["output_config"] = output_config

    loaded_output_object = output_class(**loaded_kwargs)

    return loaded_output_object


def get_output_serialization_path(
    run_folder: Path,
    output_type: str,
    output_name: str,
) -> Path:
    base_path = run_folder / "serializations" / f"{output_type}_output_serializations"
    match output_type:
        case "image" | "sequence" | "array" | "tabular" | "survival":
            path = base_path / f"{output_name}/"
        case _:
            raise ValueError(f"Invalid output type: {output_type}")

    return path


def serialize_output_objects(
    output_objects: "al_output_objects_as_dict",
    run_folder: Path,
) -> None:
    for output_name, output_object in output_objects.items():
        output_type = output_object.output_config.output_info.output_type
        output_path = get_output_serialization_path(
            run_folder=run_folder,
            output_type=output_type,
            output_name=output_name,
        )
        ensure_path_exists(path=output_path, is_folder=output_path.is_dir())

        serialize_output_object(
            output_object=output_object,
            output_folder=output_path,
        )


def serialize_output_object(
    output_object: "al_output_objects",
    output_folder: Path,
) -> None:
    output_config = output_object.output_config
    config_path = output_folder / "output_config.yaml"

    match output_object:
        case ComputedTabularOutputInfo() | ComputedSurvivalOutputInfo():
            dump_config_to_yaml(config=output_config, output_path=config_path)
        case ComputedSequenceOutputInfo():
            dump_config_to_yaml(config=output_config, output_path=config_path)

            index_to_string = output_object.vocab.itos
            with open(output_folder / "vocab_ordered.txt", "w") as f:
                for token in index_to_string:
                    f.write(f"{token}\n")

            string_to_index = output_object.vocab.stoi
            with open(output_folder / "vocab.json", "w") as f:
                json.dump(string_to_index, f)

            output_type_info = output_config.output_type_info
            assert isinstance(output_type_info, schemas.SequenceOutputTypeConfig)
            tokenizer = output_type_info.tokenizer
            if tokenizer == "bpe":
                tokenizer_callable = output_object.tokenizer
                assert tokenizer_callable is not None
                tokenizer_object = extract_tokenizer_object_from_function(
                    tokenizer_callable=tokenizer_callable
                )
                tokenizer_object.save(str(output_folder / "bpe_tokenizer.json"))

            computed_max_length = output_object.computed_max_length
            with open(output_folder / "computed_max_length.json", "w") as f:
                json.dump(computed_max_length, f)

            embedding_dim = output_object.embedding_dim
            with open(output_folder / "embedding_dim.json", "w") as f:
                json.dump(embedding_dim, f)

        case ComputedImageOutputInfo():
            dump_config_to_yaml(config=output_config, output_path=config_path)

            save_dataclass(
                obj=output_object.normalization_stats,
                file_path=output_folder / "normalization_stats.json",
            )

            num_channels_obj = {"num_channels": output_object.num_channels}
            with open(output_folder / "num_channels.json", "w") as f:
                json.dump(num_channels_obj, f)

            if output_object.diffusion_config is not None:
                save_dataclass(
                    obj=output_object.diffusion_config,
                    file_path=output_folder / "diffusion_config.json",
                )

        case ComputedArrayOutputInfo():
            dump_config_to_yaml(config=output_config, output_path=config_path)

            if output_object.normalization_stats is not None:
                save_dataclass(
                    obj=output_object.normalization_stats,
                    file_path=output_folder / "normalization_stats.json",
                )

            dtype_str = str(output_object.dtype)
            with open(output_folder / "dtype.json", "w") as f:
                json.dump(dtype_str, f)

            save_dataclass(
                obj=output_object.data_dimensions,
                file_path=output_folder / "data_dimensions.json",
            )

            if output_object.diffusion_config is not None:
                save_dataclass(
                    obj=output_object.diffusion_config,
                    file_path=output_folder / "diffusion_config.json",
                )

        case _:
            raise ValueError(f"Invalid output object: {output_object}")


def _read_serialized_output_object(
    output_class: "al_output_classes",
    serialized_output_folder: Path,
    run_folder: Path,
) -> "al_output_objects":
    loaded_object: al_output_objects
    if output_class is ComputedTabularOutputInfo:
        loaded_object = load_tabular_output_object(
            serialized_output_folder=serialized_output_folder,
            run_folder=run_folder,
        )
    elif output_class is ComputedSurvivalOutputInfo:
        loaded_object = load_survival_output_object(
            serialized_output_folder=serialized_output_folder,
            run_folder=run_folder,
        )
    elif output_class is ComputedSequenceOutputInfo:
        loaded_object = load_sequence_output_object(
            serialized_output_folder=serialized_output_folder,
            run_folder=run_folder,
        )
    elif output_class is ComputedImageOutputInfo:
        loaded_object = load_image_output_object(
            serialized_output_folder=serialized_output_folder,
        )
    elif output_class is ComputedArrayOutputInfo:
        loaded_object = load_array_output_object(
            serialized_output_folder=serialized_output_folder,
        )
    else:
        raise ValueError(f"Invalid output class: {output_class}")

    return loaded_object
