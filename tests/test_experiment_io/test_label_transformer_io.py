import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.experiment_io.label_transformer_io import (
    deserialize_transformer,
    get_transformer_path,
    get_transformer_sources,
    load_transformer,
    load_transformers,
    save_label_transformer,
    serialize_transformer,
)


@pytest.fixture
def tmp_run_folder(tmp_path: Path) -> Path:
    return tmp_path / "run_folder"


@pytest.fixture
def setup_transformer_files(tmp_run_folder: Path) -> Path:
    transformer_path = tmp_run_folder / "serializations" / "transformers" / "source1"
    transformer_path.mkdir(parents=True)

    scaler = StandardScaler()
    scaler.fit([[1, 2, 3, 4, 5]])
    scaler_data = serialize_transformer(scaler)

    encoder = LabelEncoder()
    encoder.fit(["a", "b", "c"])
    encoder_data = serialize_transformer(encoder)

    with open(transformer_path / "transformer1.json", "w") as f:
        json.dump(scaler_data, f)

    with open(transformer_path / "transformer2.json", "w") as f:
        json.dump(encoder_data, f)

    return tmp_run_folder


def test_get_transformer_sources(setup_transformer_files: Path) -> None:
    transformer_sources = get_transformer_sources(run_folder=setup_transformer_files)
    assert transformer_sources == {"source1": ["transformer1", "transformer2"]}


def test_serialize_standard_scaler() -> None:
    scaler = StandardScaler()
    scaler.fit([[1, 2, 3, 4, 5]])
    serialized = serialize_transformer(scaler)
    assert serialized["type"] == "StandardScaler"
    assert "mean" in serialized
    assert "scale" in serialized
    assert "var" in serialized


def test_serialize_label_encoder() -> None:
    encoder = LabelEncoder()
    encoder.fit(["a", "b", "c"])
    serialized = serialize_transformer(encoder)
    assert serialized["type"] == "LabelEncoder"
    assert "classes" in serialized
    assert serialized["classes"] == ["a", "b", "c"]


def test_deserialize_standard_scaler() -> None:
    original_scaler = StandardScaler()
    original_scaler.fit([[1, 2, 3, 4, 5]])
    serialized = serialize_transformer(original_scaler)
    deserialized = deserialize_transformer(serialized)
    assert isinstance(deserialized, StandardScaler)
    np.testing.assert_array_almost_equal(original_scaler.mean_, deserialized.mean_)
    np.testing.assert_array_almost_equal(original_scaler.scale_, deserialized.scale_)


def test_deserialize_label_encoder() -> None:
    original_encoder = LabelEncoder()
    original_encoder.fit(["a", "b", "c"])
    serialized = serialize_transformer(original_encoder)
    deserialized = deserialize_transformer(serialized)
    assert isinstance(deserialized, LabelEncoder)
    np.testing.assert_array_equal(original_encoder.classes_, deserialized.classes_)


def test_save_label_transformer(tmp_run_folder: Path) -> None:
    scaler = StandardScaler()
    scaler.fit([[1, 2, 3, 4, 5]])

    saved_path = save_label_transformer(
        run_folder=tmp_run_folder,
        output_name="test_output",
        transformer_name="test_scaler",
        target_transformer_object=scaler,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    with open(saved_path) as f:
        loaded_data = json.load(f)

    assert loaded_data["type"] == "StandardScaler"
    assert "mean" in loaded_data
    assert "scale" in loaded_data
    assert "var" in loaded_data


def test_load_transformer(setup_transformer_files: Path) -> None:
    transformer = load_transformer(
        run_folder=setup_transformer_files,
        source_name="source1",
        transformer_name="transformer1",
    )
    assert isinstance(transformer, StandardScaler)

    transformer = load_transformer(
        run_folder=setup_transformer_files,
        source_name="source1",
        transformer_name="transformer2",
    )
    assert isinstance(transformer, LabelEncoder)


def test_load_transformers(setup_transformer_files: Path) -> None:
    transformers = load_transformers(run_folder=setup_transformer_files)
    assert set(transformers.keys()) == {"source1"}
    assert set(transformers["source1"].keys()) == {"transformer1", "transformer2"}
    assert isinstance(transformers["source1"]["transformer1"], StandardScaler)
    assert isinstance(transformers["source1"]["transformer2"], LabelEncoder)


def test_get_transformer_path(tmp_run_folder: Path) -> None:
    path = get_transformer_path(
        run_path=tmp_run_folder,
        source_name="test_source",
        transformer_name="test_transformer",
    )
    expected_path = (
        tmp_run_folder
        / "serializations"
        / "transformers"
        / "test_source"
        / "test_transformer.json"
    )
    assert path == expected_path


@pytest.mark.parametrize("transformer_name", ["test.json", "test"])
def test_get_transformer_path_with_json_extension(
    tmp_run_folder: Path, transformer_name: str
) -> None:
    path = get_transformer_path(
        run_path=tmp_run_folder,
        source_name="test_source",
        transformer_name=transformer_name,
    )
    expected_path = (
        tmp_run_folder / "serializations" / "transformers" / "test_source" / "test.json"
    )
    assert path == expected_path


def test_load_transformers_with_output_folder(
    tmp_run_folder: Path, monkeypatch, setup_transformer_files
) -> None:
    def mock_get_run_folder(output_folder: str) -> Path:
        return Path(output_folder) / "run_folder"

    monkeypatch.setattr(
        "eir.experiment_io.label_transformer_io.get_run_folder", mock_get_run_folder
    )

    transformers = load_transformers(output_folder=str(tmp_run_folder.parent))
    assert set(transformers.keys()) == {"source1"}
    assert set(transformers["source1"].keys()) == {"transformer1", "transformer2"}
    assert isinstance(transformers["source1"]["transformer1"], StandardScaler)
    assert isinstance(transformers["source1"]["transformer2"], LabelEncoder)


def test_load_transformers_invalid_input() -> None:
    with pytest.raises(
        ValueError, match="Either 'run_folder' or 'output_folder' must be provided."
    ):
        load_transformers()
