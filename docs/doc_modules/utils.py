import shutil
from pathlib import Path
from typing import Union


def get_saved_model_path(run_folder: Path) -> str:
    model_path = next((run_folder / "saved_models").iterdir())

    assert model_path.suffix == ".pt"

    return str(model_path)


def zip_folder(target_folder: str, destination_zip: Union[str, None] = None) -> None:
    if destination_zip is None:
        destination_zip = f"{target_folder}.zip"

    base_name_no_suffix = destination_zip[:-4]

    shutil.make_archive(
        base_name=base_name_no_suffix,
        format="zip",
        root_dir=target_folder,
    )
