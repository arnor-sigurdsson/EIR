import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Callable, Dict, List, Sequence, Tuple

from aislib.misc_utils import ensure_path_exists
from pdf2image import convert_from_path
from PIL.Image import Image

from docs.doc_modules.data import get_data
from eir.setup.config_setup_modules.config_setup_utils import load_yaml_config
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)


@dataclass
class AutoDocExperimentInfo:
    name: str
    data_url: str
    data_output_path: Path
    conf_output_path: Path
    base_path: Path
    command: List[str]
    files_to_copy_mapping: Sequence[Tuple[str, str]]
    pre_run_command_modifications: Sequence[Callable[[List[str]], List[str]]] = ()
    post_run_functions: Sequence[Tuple[Callable, Dict]] = ()
    force_run_command: bool = False


def make_tutorial_data(auto_doc_experiment_info: AutoDocExperimentInfo) -> None:
    ade = auto_doc_experiment_info

    get_data(url=ade.data_url, output_path=ade.data_output_path)

    set_up_conf_files(base_path=ade.base_path, conf_output_path=ade.conf_output_path)

    command = ade.command
    for command_modification in ade.pre_run_command_modifications:
        command = command_modification(ade.command)

    run_folder = run_experiment_from_command(
        command=command, force_run=ade.force_run_command
    )

    save_command_as_text(
        command=command,
        output_path=(ade.base_path / "commands" / ade.name).with_suffix(".txt"),
    )

    find_and_copy_files(
        run_folder=run_folder,
        output_folder=ade.base_path,
        patterns=ade.files_to_copy_mapping,
    )

    for func, kwargs in ade.post_run_functions:
        func(**kwargs)


def save_command_as_text(command: List[str], output_path: Path) -> None:
    ensure_path_exists(path=output_path)

    command_as_str = command[0] + " \\\n"

    cur_str = command[1]

    for part in command[2:]:
        if part.startswith("--"):
            command_as_str += cur_str + " \\\n"
            cur_str = part
        else:
            cur_str += " " + part

    command_as_str += cur_str

    output_path.write_text(command_as_str)


def set_up_conf_files(base_path: Path, conf_output_path: Path):
    ensure_path_exists(path=conf_output_path, is_folder=True)
    for path in base_path.rglob("*"):
        if path.suffix == ".yaml":
            copy2(path, conf_output_path)


def find_and_copy_files(
    run_folder: Path,
    output_folder: Path,
    patterns: Sequence[Tuple[str, str]],
    strict: bool = True,
):
    matched_patterns = {}

    for path in run_folder.rglob("*"):
        if path.name == ".DS_Store":
            continue

        for pattern, target in patterns:
            if re.match(pattern=pattern, string=str(path)) or pattern in str(path):
                output_destination = output_folder / target
                ensure_path_exists(path=output_destination, is_folder=False)
                copy2(path, output_destination)

                if output_destination.suffix == ".pdf":
                    files = convert_from_path(
                        pdf_path=output_destination,
                        fmt="png",
                    )

                    assert len(files) == 1
                    pil_image: Image = files[0]

                    pil_image.save(output_destination.with_suffix(".png"))
                    output_destination.unlink()

                matched_patterns[pattern] = True

    for pattern, _ in patterns:
        if pattern not in matched_patterns:
            if strict:
                raise FileNotFoundError(
                    f"No files found for pattern {pattern} in {run_folder}."
                )
            else:
                logger.warning(f"No files found for pattern {pattern} in {run_folder}.")


def run_experiment_from_command(command: List[str], force_run: bool = False):
    globals_file = next(i for i in command if "globals" in i)
    globals_dict = load_yaml_config(config_path=globals_file)
    run_folder = Path(globals_dict["output_folder"])

    output_folder_injected = tuple(i for i in command if ".output_folder=" in i)
    if output_folder_injected:
        assert len(output_folder_injected) == 1
        output_folder_inject_string = output_folder_injected[0]
        run_folder = Path(output_folder_inject_string.split(".output_folder=")[-1])

    if not force_run and run_folder.exists():
        return run_folder

    subprocess.run(args=command)

    return run_folder


def run_capture_and_save(command: List[str], output_path: Path, *args, **kwargs):
    result_text = run_subprocess_and_capture_output(command=command)

    output_path.write_text(result_text)


def run_subprocess_and_capture_output(command: List[str]):
    result = subprocess.run(args=command, capture_output=True, text=True).stdout

    return result
