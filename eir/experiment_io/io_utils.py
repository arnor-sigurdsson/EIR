import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import numpy as np
import torch
import yaml
from aislib.misc_utils import ensure_path_exists

from eir import __version__
from eir.setup.config_setup_modules.config_setup_utils import object_to_primitives
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


def encode_numpy(obj: np.ndarray) -> dict[str, Any]:
    return {"__np_array__": obj.tolist(), "dtype": str(obj.dtype)}


def encode_torch(obj: torch.Tensor) -> dict[str, Any]:
    return {
        "__torch_tensor__": obj.cpu().numpy().tolist(),
        "dtype": str(obj.dtype),
        "device": str(obj.device),
    }


def encode_tuple(obj: tuple) -> list:
    return list(obj)


def encode_primitive(obj: Any) -> Any:
    return obj


class EncoderProtocol(Protocol):
    def __call__(self, obj: Any) -> Any: ...


def get_encoder(obj: Any) -> EncoderProtocol:
    encoders = {
        np.ndarray: encode_numpy,
        torch.Tensor: encode_torch,
        tuple: encode_tuple,
        int: encode_primitive,
        float: encode_primitive,
        str: encode_primitive,
        bool: encode_primitive,
        type(None): encode_primitive,
    }

    encoder = encoders.get(type(obj), encode_primitive)

    return cast(EncoderProtocol, encoder)


def custom_encoder(obj: Any) -> Any:
    return get_encoder(obj=obj)(obj=obj)


def decode_numpy(dct: dict[str, Any]) -> np.ndarray:
    return np.array(dct["__np_array__"], dtype=np.dtype(dct["dtype"]))


def decode_torch(dct: dict[str, Any]) -> torch.Tensor:
    tensor = torch.tensor(
        dct["__torch_tensor__"], dtype=getattr(torch, str(dct["dtype"]).split(".")[-1])
    )
    return tensor.to(dct["device"])


def decode_primitive(dct: Any) -> Any:
    return dct


def get_decoder(dct: Any) -> Callable:
    if isinstance(dct, dict):
        decoders = {
            "__np_array__": decode_numpy,
            "__torch_tensor__": decode_torch,
        }
        for key in decoders:
            if key in dct:
                return decoders[key]
    return decode_primitive


def custom_decoder(dct: Any) -> Any:
    return get_decoder(dct=dct)(dct=dct)


def serialize_dataclass(obj: Any) -> str:
    return json.dumps(obj=asdict(obj), default=custom_encoder, indent=2)


def deserialize_dataclass(cls: type, json_str: str) -> Any:
    data = json.loads(s=json_str, object_hook=custom_decoder)
    return cls(**data)


def save_dataclass(obj: Any, file_path: Path) -> None:
    file_path.write_text(data=serialize_dataclass(obj=obj))


def load_dataclass(cls: type, file_path: Path) -> Any:
    return deserialize_dataclass(cls=cls, json_str=file_path.read_text())


def get_run_folder_from_model_path(model_path: str) -> Path:
    model_path_object = Path(model_path)
    assert model_path_object.exists()

    run_folder = model_path_object.parents[1]
    assert run_folder.exists()

    return run_folder


def dump_config_to_yaml(
    config: Any,
    output_path: Path,
    stripped: bool = True,
) -> None:
    object_primitive = object_to_primitives(obj=config)

    if stripped:
        object_primitive = strip_config(config=object_primitive)

    ensure_path_exists(path=output_path, is_folder=False)
    with open(output_path, "w") as outfile:
        yaml.dump(data=object_primitive, stream=outfile)


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


def strip_config(config: dict | list[dict]) -> dict | list[dict]:
    keys_to_strip = [
        "basic_experiment.manual_valid_ids_file",
        "basic_experiment.output_folder",
        "input_info.input_source",
        "output_info.output_source",
        "input_type_info.vocab_file",
        "input_type_info.snp_file",
        "input_type_info.subset_snps_file",
        "output_type_info.vocab_file",
    ]
    replacements = {k: None for k in keys_to_strip}

    if isinstance(config, list):
        stripped_configs = []

        for cur_config in config:
            cur_config_stripped = replace_dict_values(
                target=cur_config,
                replacements=replacements,
            )
            stripped_configs.append(cur_config_stripped)

        return stripped_configs

    config_stripped = replace_dict_values(
        target=config,
        replacements=replacements,
    )

    return config_stripped


def replace_dict_values(
    target: dict[str, Any],
    replacements: dict[str, Any],
) -> dict[str, Any]:

    def recursive_replace(
        d: dict[str, Any],
        path: str = "",
    ) -> None:
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if current_path in replacements:
                d[key] = replacements[current_path]
            elif isinstance(value, dict):
                recursive_replace(d=value, path=current_path)

    result = target.copy()
    recursive_replace(d=result)
    return result
