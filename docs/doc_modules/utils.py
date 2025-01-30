import shutil
from pathlib import Path


def get_saved_model_path(run_folder: Path) -> str:
    model_path = next((run_folder / "saved_models").iterdir())

    assert model_path.suffix == ".pt"

    return str(model_path)


def zip_folder(target_folder: str, destination_zip: str | None = None) -> None:
    if destination_zip is None:
        destination_zip = f"{target_folder}.zip"

    base_name_no_suffix = destination_zip[:-4]

    shutil.make_archive(
        base_name=base_name_no_suffix,
        format="zip",
        root_dir=target_folder,
    )


def add_model_path_to_command(
    command: list[str], run_path: Path | str, replace_key: str = "FILL_MODEL"
) -> list[str]:
    model_path = get_saved_model_path(run_folder=Path(run_path))
    command = [x.replace(replace_key, model_path) for x in command]
    return command
