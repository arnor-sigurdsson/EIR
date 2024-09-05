import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from aislib.misc_utils import ensure_path_exists
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.train_utils.utils import get_run_folder

if TYPE_CHECKING:
    from eir.data_load.label_setup import (
        al_label_transformers,
        al_label_transformers_object,
    )


def get_transformer_sources(run_folder: Path) -> dict[str, list[str]]:
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
) -> "al_label_transformers_object":
    target_transformer_path = get_transformer_path(
        run_path=run_folder,
        transformer_name=transformer_name,
        source_name=source_name,
    )
    serialized_data = read_json(target_transformer_path)
    return deserialize_transformer(data=serialized_data)


def load_transformers(
    transformers_to_load: Optional[dict[str, list[str]]] = None,
    output_folder: Optional[str] = None,
    run_folder: Optional[Path] = None,
) -> dict[str, "al_label_transformers"]:
    if not run_folder and not output_folder:
        raise ValueError("Either 'run_folder' or 'output_folder' must be provided.")

    if not run_folder:
        assert output_folder is not None
        run_folder = get_run_folder(output_folder=output_folder)

    if not transformers_to_load:
        transformers_to_load = get_transformer_sources(run_folder=run_folder)

    loaded_transformers: dict[str, "al_label_transformers"] = {}
    for source_name, source_transformers_to_load in transformers_to_load.items():
        loaded_transformers[source_name] = {}
        for transformer_name in source_transformers_to_load:
            loaded_transformers[source_name][transformer_name] = load_transformer(
                run_folder=run_folder,
                source_name=source_name,
                transformer_name=transformer_name,
            )

    return loaded_transformers


def get_transformer_path(
    run_path: Path, source_name: str, transformer_name: str
) -> Path:
    if not transformer_name.endswith(".json"):
        transformer_name = f"{transformer_name}.json"

    transformer_path = (
        run_path / "serializations/transformers" / source_name / f"{transformer_name}"
    )

    return transformer_path


def save_label_transformer(
    run_folder: Path,
    output_name: str,
    transformer_name: str,
    target_transformer_object: "al_label_transformers_object",
) -> Path:
    target_transformer_outpath = get_transformer_path(
        run_path=run_folder,
        source_name=output_name,
        transformer_name=transformer_name,
    )
    ensure_path_exists(target_transformer_outpath)
    serialized_data = serialize_transformer(transformer=target_transformer_object)
    write_json(data=serialized_data, path=target_transformer_outpath)
    return target_transformer_outpath


def save_transformer_set(
    transformers_per_source: dict[str, "al_label_transformers"], run_folder: Path
) -> None:
    for output_name, transformers in transformers_per_source.items():
        for transformer_name, transformer_object in transformers.items():
            save_label_transformer(
                run_folder=run_folder,
                output_name=output_name,
                transformer_name=transformer_name,
                target_transformer_object=transformer_object,
            )


def serialize_standard_scaler(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "type": "StandardScaler",
        "mean": scaler.mean_.tolist() if scaler.mean_ is not None else None,
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
        "var": scaler.var_.tolist() if scaler.var_ is not None else None,
    }


def serialize_label_encoder(encoder: LabelEncoder) -> dict[str, Any]:
    return {
        "type": "LabelEncoder",
        "classes": encoder.classes_.tolist(),
    }


def serialize_transformer(
    transformer: "al_label_transformers_object",
) -> dict[str, Any]:
    serializers: dict[type, Callable] = {
        StandardScaler: serialize_standard_scaler,
        LabelEncoder: serialize_label_encoder,
    }
    serializer = serializers.get(type(transformer))
    if serializer is None:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")
    return serializer(transformer)


def deserialize_standard_scaler(data: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"]) if data["mean"] is not None else None
    scaler.scale_ = np.array(data["scale"]) if data["scale"] is not None else None
    scaler.var_ = np.array(data["var"]) if data["var"] is not None else None
    return scaler


def deserialize_label_encoder(data: dict[str, Any]) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.array(data["classes"])
    return encoder


def deserialize_transformer(data: dict[str, Any]) -> "al_label_transformers_object":
    deserializers: dict[str, Callable] = {
        "StandardScaler": deserialize_standard_scaler,
        "LabelEncoder": deserialize_label_encoder,
    }
    deserializer = deserializers.get(data["type"])
    if deserializer is None:
        raise ValueError(f"Unsupported transformer type: {data['type']}")
    return deserializer(data)


def write_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)
