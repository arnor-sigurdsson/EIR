from pathlib import Path
import zipfile

import gdown
from aislib.misc_utils import ensure_path_exists


def download_google_drive_file(
    url: str, output_path: Path, overwrite: bool = False
) -> None:

    if output_path.exists() and not overwrite:
        return

    if "/d/" in url:
        url = _parse_google_url(url_to_parse=url)

    gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True)


def _parse_google_url(url_to_parse: str):
    id_part = url_to_parse.split("/d/")[-1].split("/")[0]
    parsed_url = f"https://drive.google.com/uc?id={id_part}"

    return parsed_url


def unzip_file(file: Path):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(file.parent)

    return file.parent


def get_data(url: str, output_path: Path):
    ensure_path_exists(path=output_path, is_folder=False)

    download_google_drive_file(url=url, output_path=output_path, overwrite=False)

    unzip_file(file=output_path)

    return output_path.with_suffix("")
