from pathlib import Path

import pytest
from joblib import dump
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.experiment_io.experiment_io import (
    get_transformer_sources,
    load_transformer,
    load_transformers,
)


@pytest.fixture
def setup_transformer_files(tmp_path: Path) -> Path:
    transformer_path = tmp_path / "serializations" / "transformers" / "source1"
    transformer_path.mkdir(parents=True)
    dump(StandardScaler(), transformer_path / "transformer1.save")
    dump(LabelEncoder(), transformer_path / "transformer2.save")

    return tmp_path


def test_get_transformer_sources(setup_transformer_files: Path) -> None:
    run_folder = setup_transformer_files
    transformer_sources = get_transformer_sources(run_folder=run_folder)
    assert transformer_sources == {"source1": ["transformer1", "transformer2"]}


def test_load_transformer(setup_transformer_files: Path) -> None:
    run_folder = setup_transformer_files
    transformer = load_transformer(
        run_folder=run_folder,
        source_name="source1",
        transformer_name="transformer1",
    )
    assert isinstance(transformer, StandardScaler)

    transformer = load_transformer(
        run_folder=run_folder,
        source_name="source1",
        transformer_name="transformer2",
    )
    assert isinstance(transformer, LabelEncoder)


def test_load_transformers(setup_transformer_files: Path) -> None:
    run_folder = setup_transformer_files
    transformers = load_transformers(run_folder=run_folder)
    assert set(transformers.keys()) == {"source1"}
    assert set(transformers["source1"].keys()) == {"transformer1", "transformer2"}
    assert isinstance(transformers["source1"]["transformer1"], StandardScaler)
    assert isinstance(transformers["source1"]["transformer2"], LabelEncoder)
