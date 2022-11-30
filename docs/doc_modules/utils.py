from pathlib import Path


def get_saved_model_path(run_folder: Path) -> str:
    model_path = next((run_folder / "saved_models").iterdir())

    assert model_path.suffix == ".pt"

    return str(model_path)
