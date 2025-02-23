import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from eir.setup.config_setup_modules.config_setup_utils import load_yaml_config

CONTENT_ROOT = "i_scaling"
TUTORIAL_NAME = "02_scaling_compute"


def run_with_server(command: list[str]) -> Path:
    server_path = f"docs/doc_modules/{CONTENT_ROOT}/openwebtext_streamer.py"

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

    env = os.environ.copy()
    env.update(
        {
            "MAX_SEQUENCES": "8000000",
            "SEQUENCE_LENGTH": "256",
            "DATASET_NAME": "Skylion007/openwebtext",
            "DATASET_SPLIT": "train",
        }
    )

    server_args = [
        "python",
        server_path,
    ]

    server_process = subprocess.Popen(
        server_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )

    try:
        time.sleep(5)
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
            "samples/45000/auto/0_generated.txt",
            "figures/auto_generated_iter_45000.txt",
        ),
        (
            "samples/45000/manual/1_generated.txt",
            "figures/manual_generated_iter_45000.txt",
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
