from collections.abc import Sequence
from functools import partial
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command


def get_04_imdb_run_1_transformer_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/"
    )

    conf_output_path = "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/04_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/04_imdb_input.yaml",
        "--output_configs",
        f"{conf_output_path}/04_imdb_output.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/04_imdb_training_curve_ACC_transformer_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/data/imdb.zip"
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/",
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
        name="SEQUENCE_IMDB_1_TRANSFORMER",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_04_imdb_run_2_local_transformer_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/"
    )

    conf_output_path = "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/04_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/04_imdb_input_windowed.yaml",
        "--output_configs",
        f"{conf_output_path}/04_imdb_output.yaml",
        "--04_imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_04_imdb_run_local",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/04_imdb_training_curve_ACC_local_transformer_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/data/imdb.zip"
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_IMDB_2_LOCAL",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_04_imdb_run_3_longformer_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/"
    )

    conf_output_path = "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/04_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/04_imdb_input_longformer.yaml",
        "--output_configs",
        f"{conf_output_path}/04_imdb_output.yaml",
        "--04_imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_04_imdb_run_longformer",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/04_imdb_training_curve_ACC_longformer_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/data/imdb.zip"
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_IMDB_3_LONGFORMER",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_04_imdb_run_4_tiny_bert_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/"
    )

    conf_output_path = "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/04_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/04_imdb_input_tiny-bert.yaml",
        "--output_configs",
        f"{conf_output_path}/04_imdb_output.yaml",
        "--04_imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_04_imdb_run_tiny-bert",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/04_imdb_training_curve_ACC_tiny_bert_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/data/imdb.zip"
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_IMDB_4_TINY_BERT",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_04_imdb_run_5_combined_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/"
    )

    conf_output_path = "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/04_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/04_imdb_input_windowed.yaml",
        f"{conf_output_path}/04_imdb_input_longformer.yaml",
        f"{conf_output_path}/04_imdb_input_tiny-bert.yaml",
        "--output_configs",
        f"{conf_output_path}/04_imdb_output.yaml",
        "--04_imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_04_imdb_run_combined",
        "--04_imdb_globals.basic_experiment.device='mps'",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/04_imdb_training_curve_ACC_combined_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/04_pretrained_sequence_tutorial/data/imdb.zip"
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_IMDB_5_COMBINED",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_04_imdb_run_1_serve_info() -> AutoDocServingInfo:
    base_path = (
        "docs/tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial"
    )

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    base_inputs = [
        "This movie was great! I loved it!",
        "This movie was terrible! I hated it!",
        "You'll have to have your wits about "
        "you and your brain fully switched"
        " on watching Oppenheimer as it could easily get away from a "
        "nonattentive viewer. This is intelligent filmmaking which shows "
        "it's audience great respect. It fires dialogue packed with "
        "information at a relentless pace and jumps to very different "
        "times in Oppenheimer's life continuously through it's 3 hour"
        " runtime. There are visual clues to guide the viewer through these"
        " times but again you'll have to get to grips with these quite "
        "quickly. This relentlessness helps to express the urgency with "
        "which the US attacked it's chase for the atomic bomb before "
        "Germany could do the same. An absolute career best performance "
        "from (the consistenly brilliant) Cillian Murphy anchors the film. ",
    ]

    feature_extractor_per_input_names = [
        "imdb_reviews_windowed",
        "imdb_reviews_longformer",
        "imdb_reviews_tiny_bert",
    ]

    example_requests = [[]]

    for input_ in base_inputs:
        cur_input = {}
        for feature_extractor_name in feature_extractor_per_input_names:
            cur_input[feature_extractor_name] = input_
        example_requests[0].append(cur_input)

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/a_using_eir/"
        "tutorial_04_imdb_run_combined",
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
        name="COMBINED_SEQUENCE_DEPLOY",
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

    def send_request(url: str, payload: list[dict]) -> dict:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    payload = [
        {
            "imdb_reviews_windowed": "This movie was great! I loved it!",
            "imdb_reviews_longformer": "This movie was great! I loved it!",
            "imdb_reviews_tiny_bert": "This movie was great! I loved it!",
        },
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
        -d '[{"imdb_reviews_windowed": "This movie was great! I loved it!",
        "imdb_reviews_longformer": "This movie was great! I loved it!",
        "imdb_reviews_tiny_bert": "This movie was great! I loved it!"}]'
        """

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
    exp_1 = get_04_imdb_run_1_transformer_info()
    exp_2 = get_04_imdb_run_2_local_transformer_info()
    exp_3 = get_04_imdb_run_3_longformer_info()
    exp_4 = get_04_imdb_run_4_tiny_bert_info()
    exp_5 = get_04_imdb_run_5_combined_info()
    exp_6 = get_04_imdb_run_1_serve_info()

    return [exp_1, exp_2, exp_3, exp_4, exp_5, exp_6]
