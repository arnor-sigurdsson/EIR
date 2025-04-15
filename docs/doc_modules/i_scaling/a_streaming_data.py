import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from eir.setup.config_setup_modules.config_setup_utils import load_yaml_config
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

CONTENT_ROOT = "i_scaling"
TUTORIAL_NAME = "01_streaming_data"


def run_with_server(command: list[str]) -> Path:
    server_path = f"docs/doc_modules/{CONTENT_ROOT}/text_streamer.py"

    globals_file = next(i for i in command if "globals" in i)
    globals_dict = load_yaml_config(config_path=globals_file)
    run_folder = Path(globals_dict["basic_experiment"]["output_folder"])

    output_folder_injected = tuple(i for i in command if ".output_folder=" in i)
    if output_folder_injected:
        assert len(output_folder_injected) == 1
        output_folder_inject_string = output_folder_injected[0]
        run_folder = Path(output_folder_inject_string.split(".output_folder=")[-1])

    if run_folder.exists():
        return run_folder

    cur_env = os.environ.copy()
    cur_env["SEQUENCE_LENGTH"] = "64"

    server_process = subprocess.Popen(
        ["python", server_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info("Server process starting...")

    try:
        time.sleep(60)
        subprocess.run(args=command, check=True)
    finally:
        server_process.terminate()
        server_process.wait(timeout=5)

    return run_folder


def get_streaming_generation_experiment() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CONTENT_ROOT}/{TUTORIAL_NAME}"
    conf_output_path = f"eir_tutorials/{CONTENT_ROOT}/{TUTORIAL_NAME}"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS.pdf",
        ),
        (
            "samples/500/auto/0_generated.txt",
            "figures/auto_generated_iter_500.txt",
        ),
        (
            "samples/500/manual/1_generated.txt",
            "figures/manual_generated_iter_500.txt",
        ),
        (
            "samples/2500/auto/0_generated.txt",
            "figures/auto_generated_iter_2500.txt",
        ),
        (
            "samples/2500/manual/1_generated.txt",
            "figures/manual_generated_iter_2500.txt",
        ),
    ]

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CONTENT_ROOT}/{TUTORIAL_NAME}",
                "-L",
                "3",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="STREAMING_SEQUENCE_GENERATION",
        data_url=None,
        data_output_path=None,
        base_path=Path(base_path),
        conf_output_path=Path(conf_output_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
        run_command_wrapper=run_with_server,
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_streaming_generation_experiment()
    return [exp_1]
