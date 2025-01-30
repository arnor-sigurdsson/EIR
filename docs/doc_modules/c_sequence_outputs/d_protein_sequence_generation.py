from collections.abc import Sequence
from functools import partial
from pathlib import Path

from aislib.misc_utils import ensure_path_exists

from docs.doc_modules.c_sequence_outputs.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command, get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "04_protein_sequence_generation"


def get_protein_sequence_generation_sequence_only() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
        "--globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/c_sequence_output/04_protein_sequence_generation_sequence_only",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_text.pdf",
        ),
        (
            "latent_outputs/5000/"
            "output_modules.protein_sequence.output_transformer.layers.1/"
            "tsne.png",
            "figures/1_only_sequence/tsne_5000.png",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/protein_generation.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}",
                "-L",
                "3",
                "-I",
                "*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="01_PROTEIN_GENERATION_ONLY_TEXT",
        data_url="https://drive.google.com/file/d/16FMSCOdPxGcCx8oJD5GU1AYIjacbJ2yZ",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_protein_sequence_generation_tabular() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_tabular.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_conditioned.yaml",
        "--globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/c_sequence_output/04_protein_sequence_generation_tabular",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_2_conditioned.pdf",
        ),
        (
            "latent_outputs/5000/"
            "output_modules.protein_sequence.output_transformer.layers.1/"
            "tsne.png",
            "figures/1_only_conditioned/tsne_5000.png",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/protein_generation.zip")

    ade = AutoDocExperimentInfo(
        name="02_PROTEIN_GENERATION_TABULAR",
        data_url="https://drive.google.com/file/d/16FMSCOdPxGcCx8oJD5GU1AYIjacbJ2yZ",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_protein_sequence_generation_tabular_predict() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_1_output_path = (
        "eir_tutorials/tutorial_runs/c_sequence_output/"
        "04_protein_sequence_generation_tabular/test_results"
    )
    ensure_path_exists(path=Path(run_1_output_path), is_folder=True)

    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_tabular_test.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_conditioned_test.yaml",
        "--model_path",
        "FILL_MODEL",
        "--output_folder",
        run_1_output_path,
        "--evaluate",
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/protein_generation.zip")

    ade = AutoDocExperimentInfo(
        name="03_PREDICT_GENERATION",
        data_url="https://drive.google.com/file/d/16FMSCOdPxGcCx8oJD5GU1AYIjacbJ2yZ",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=(),
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
        force_run_command=True,
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}_tabular/"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_protein_sequence_generation_tabular_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    model_path_placeholder = "FILL_MODEL"

    server_command = ["eirserve", "--model-path", model_path_placeholder]

    example_requests = [
        [
            {
                "proteins_tabular": {"classification": "HYDROLASE"},
                "protein_sequence": "",
            },
            {
                "proteins_tabular": {"classification": "TRANSFERASE"},
                "protein_sequence": "",
            },
            {
                "proteins_tabular": {"classification": "OXIDOREDUCTASE"},
                "protein_sequence": "AAA",
            },
        ],
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/c_sequence_output/"
        "04_protein_sequence_generation_tabular",
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    bash_args = _get_example_request_bash_args()
    example_request_module_bash = build_request_example_module_from_function(
        **bash_args
    )

    ade = AutoDocServingInfo(
        name="TABULAR_DEPLOY",
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


def example_request_function_python():
    import requests

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    payload = [
        {"proteins_tabular": {"classification": "HYDROLASE"}, "protein_sequence": ""},
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def _get_example_request_bash_args():
    command = """curl -X POST \\
        "http://localhost:8000/predict" \\
        -H "accept: application/json" \\
        -H "Content-Type: application/json" \\
        -d '[{"proteins_tabular": {"classification": "HYDROLASE"},
         "protein_sequence": ""}]'"""

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
    exp_1 = get_protein_sequence_generation_sequence_only()
    exp_2 = get_protein_sequence_generation_tabular()
    exp_3 = get_protein_sequence_generation_tabular_predict()
    exp_4 = get_protein_sequence_generation_tabular_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
        exp_4,
    ]
