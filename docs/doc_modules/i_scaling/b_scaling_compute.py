import os
import subprocess
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command, get_saved_model_path
from eir.setup.config_setup_modules.config_setup_utils import load_yaml_config
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

CONTENT_ROOT = "i_scaling"
TUTORIAL_NAME = "02_scaling_compute"


def run_with_server(
    command: list[str],
    dataset_name: str = "HuggingFaceFW/fineweb",
) -> Path:
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

    env = os.environ.copy()
    env.update(
        {
            "SEQUENCE_LENGTH": "512",
            "DATASET_NAME": dataset_name,
            "DATASET_SPLIT": "train",
        }
    )

    server_args = [
        "python",
        server_path,
    ]

    server_process = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
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
            "samples/1000/auto/0_generated.txt",
            "figures/auto_generated_iter_500.txt",
        ),
        (
            "samples/1000/manual/1_generated.txt",
            "figures/manual_generated_iter_500.txt",
        ),
        (
            "samples/52000/auto/0_generated.txt",
            "figures/auto_generated_iter_52000.txt",
        ),
        (
            "samples/52000/manual/1_generated.txt",
            "figures/manual_generated_iter_52000.txt",
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


def get_sft_from_pretrained_experiment() -> AutoDocExperimentInfo:
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
        f"--globals.basic_experiment.output_folder="
        f"eir_tutorials/tutorial_runs/{CONTENT_ROOT}/{TUTORIAL_NAME}_sft_pretrained/",
        "--globals.model.pretrained_checkpoint=FILL_MODEL",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/sft_pretrained_training_curve_LOSS.pdf",
        ),
        (
            "samples/1000/auto/0_generated.txt",
            "figures/sft_pretrained_auto_generated_iter_500.txt",
        ),
        (
            "samples/12000/auto/0_generated.txt",
            "figures/sft_pretrained_auto_generated_iter_12000.txt",
        ),
    ]

    ade = AutoDocExperimentInfo(
        name="SFT_FROM_PRETRAINED_EXPERIMENT",
        data_url=None,
        data_output_path=None,
        base_path=Path(base_path),
        conf_output_path=Path(conf_output_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        run_command_wrapper=partial(run_with_server, dataset_name="tatsu-lab/alpaca"),
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CONTENT_ROOT}/{TUTORIAL_NAME}/"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_sft_generation_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CONTENT_ROOT}/{TUTORIAL_NAME}"

    server_command = [
        "eirserve",
        "--device",
        "cuda",
        "--model-path",
        "FILL_MODEL",
    ]

    example_requests = [
        [
            {
                "text_output": "### Instruction: Generate a list of five healthy"
                " breakfast ideas. ### Response:"
            },
            {
                "text_output": "### Instruction: Explain quantum computing in simple"
                " terms. ### Response:"
            },
            {
                "text_output": "### Instruction: Write a short poem about artificial "
                "intelligence. ### Response:"
            },
            {
                "text_output": "### Instruction: Give me three tips for improving"
                " time management. ### Response:"
            },
            {
                "text_output": "### Instruction: Describe the process of "
                "photosynthesis. ### Response:"
            },
        ]
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CONTENT_ROOT}/{TUTORIAL_NAME}_sft_pretrained",
    )

    example_request_module_python = build_request_example_module_from_function(
        function=sft_example_request_function_python,
        name="python",
        language="python",
    )

    bash_args = _get_sft_example_request_bash_args()
    example_request_module_bash = build_request_example_module_from_function(
        **bash_args
    )

    ade = AutoDocServingInfo(
        name="SFT_GENERATION_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
            example_request_module_bash,
        ],
    )

    return ade


def sft_example_request_function_python():
    import requests

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    payload = [
        {
            "text_output": "### Instruction: Write a short story about a robot "
            "learning to feel emotions. ### Response:"
        },
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def _get_sft_example_request_bash_args():
    command = """curl -X POST \\
        "http://localhost:8000/predict" \\
        -H "accept: application/json" \\
        -H "Content-Type: application/json" \\
        -d '[{"text_output": "### Instruction: Explain three ways to reduce carbon
         emissions. ### Response:"}]
           '"""

    def _function_to_run_example() -> dict:
        import json
        import subprocess

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result_as_dict = json.loads(result.stdout)
        return result_as_dict

    command_as_text = command
    return {
        "function": _function_to_run_example,
        "custom_body": command_as_text,
        "name": "bash",
        "language": "shell",
    }


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_streaming_generation_experiment()
    exp_2 = get_sft_from_pretrained_experiment()
    exp_3 = get_sft_generation_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
