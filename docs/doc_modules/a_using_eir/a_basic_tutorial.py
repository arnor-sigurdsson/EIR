import textwrap
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import get_saved_model_path


def get_tutorial_01_run_1_gln_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    conf_output_path = "eir_tutorials/a_using_eir/01_basic_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--output_configs",
        f"{conf_output_path}/tutorial_01_outputs.yaml",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_01_training_curve_ACC_gln_1.pdf"),
        ("600/confusion_matrix", "figures/tutorial_01_confusion_matrix_gln_1.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/01_basic_tutorial/data/processed_sample_data.zip"
    )

    get_data_folder = (
        run_capture_and_save,
        {
            "command": ["tree", str(data_output_path.parent), "-L", "2", "--noreport"],
            "output_path": Path(base_path) / "commands/input_folder.txt",
        },
    )
    get_eirtrain_help = (
        run_capture_and_save,
        {
            "command": [
                "eirtrain",
                "--help",
            ],
            "output_path": Path(base_path) / "commands/eirtrain_help.txt",
        },
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/a_using_eir/01_basic_tutorial/",
                "-L",
                "3",
                "-I",
                "*01b*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )
    get_run_1_experiment_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run/",
                "-I",
                "tensorboard_logs|serializations|transformers|*.yaml|*.pt|*.log",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/experiment_01_folder.txt",
        },
    )
    get_eirpredict_help = (
        run_capture_and_save,
        {
            "command": [
                "eirpredict",
                "--help",
            ],
            "output_path": Path(base_path) / "commands/eirpredict_help.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="GLN_1",
        data_url="https://drive.google.com/file/d/1MELauhv7zFwxM8nonnj3iu_SmS69MuNi",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_data_folder,
            get_eirtrain_help,
            get_tutorial_folder,
            get_run_1_experiment_folder,
            get_eirpredict_help,
        ),
    )

    return ade


def get_tutorial_01_run_2_gln_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    conf_output_path = "eir_tutorials/a_using_eir/01_basic_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--output_configs",
        f"{conf_output_path}/tutorial_01_outputs.yaml",
        "--tutorial_01_globals.basic_experiment."
        "output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_01_run_lr-0.002_epochs-20",
        "--tutorial_01_globals.optimization.lr=0.002",
        "--tutorial_01_globals.basic_experiment.n_epochs=20",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_01_training_curve_ACC_gln_2.pdf"),
        ("600/confusion_matrix", "figures/tutorial_01_confusion_matrix_gln_2.pdf"),
    ]

    ade = AutoDocExperimentInfo(
        name="GLN_2",
        data_url="https://drive.google.com/file/d/1MELauhv7zFwxM8nonnj3iu_SmS69MuNi",
        data_output_path=Path(
            "data/tutorial_data/01_basic_tutorial/processed_sample_data.zip"
        ),
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_tutorial_01_run_2_gln_predict_info() -> AutoDocExperimentInfo:
    """
    We are abusing the `make_tutorial_data` here a bit by switching to the predict
    code, but we'll allow it for now.
    """
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    conf_output_path = "eir_tutorials/a_using_eir/01_basic_tutorial/conf"

    run_1_output_path = (
        "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run/"
        "test_predictions/known_outputs"
    )
    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--output_configs",
        f"{conf_output_path}/tutorial_01_outputs.yaml",
        "--model_path",
        "FILL_MODEL",
        "--evaluate",
        "--output_folder",
        run_1_output_path,
    ]

    data_output_path = Path(
        "data/tutorial_data/01_basic_tutorial/processed_sample_data.zip"
    )

    mapping = [
        (
            "calculated_metrics",
            "tutorial_data/calculated_metrics_test.json",
        ),
        (
            "Origin/predictions.csv",
            "tutorial_data/predictions_test.csv",
        ),
    ]

    csv_preview_func = (
        csv_preview,
        {
            "run_path": Path(run_1_output_path),
            "output_path": Path(base_path, "csv_preview.html"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="GLN_1_PREDICT",
        data_url="https://drive.google.com/file/d/1MELauhv7zFwxM8nonnj3iu_SmS69MuNi",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(csv_preview_func,),
        force_run_command=True,
    )

    return ade


def get_tutorial_01_run_3_gln_predict_info() -> AutoDocExperimentInfo:
    """
    We are abusing the `make_tutorial_data` here a bit by switching to the predict
    code, but we'll allow it for now.
    """
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    conf_output_path = "eir_tutorials/a_using_eir/01_basic_tutorial/conf"

    run_1_output_path = (
        "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run/"
        "test_predictions/unknown_outputs"
    )

    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--output_configs",
        f"{conf_output_path}/tutorial_01_outputs_unknown.yaml",
        "--model_path",
        "FILL_MODEL",
        "--output_folder",
        run_1_output_path,
    ]

    data_output_path = Path(
        "data/tutorial_data/01_basic_tutorial/processed_sample_data.zip"
    )

    mapping = [
        (
            "Origin/predictions.csv",
            "tutorial_data/predictions_test_unknown.csv",
        ),
    ]

    csv_preview_func = (
        csv_preview,
        {
            "run_path": Path(run_1_output_path),
            "output_path": Path(base_path, "csv_preview_unknown.html"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="GLN_1_PREDICT_UNKNOWN",
        data_url="https://drive.google.com/file/d/1MELauhv7zFwxM8nonnj3iu_SmS69MuNi",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(csv_preview_func,),
        force_run_command=True,
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def csv_preview(run_path: Path, output_path: Path) -> None:
    csv_path = run_path / "ancestry_output/Origin/predictions.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    df = df.round(2)
    df.columns = ["<br>".join(textwrap.wrap(name, width=20)) for name in df.columns]

    preview = df.head(5).to_html(index=False, escape=False)

    with open(output_path, "w") as f:
        f.write(preview)
        f.write("<br>")


def get_tutorial_01_run_3_serve() -> AutoDocServingInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    model_path_placeholder = "FILL_MODEL"

    server_command = ["eirserve", "--model-path", model_path_placeholder]

    base = (
        "eir_tutorials/a_using_eir/01_basic_tutorial/data/processed_sample_data/arrays"
    )
    example_requests = [
        [
            {"genotype": f"{base}/A374.npy"},
            {"genotype": f"{base}/Ayodo_468C.npy"},
            {"genotype": f"{base}/NOR146.npy"},
        ]
    ]

    example_request_module = build_request_example_module_from_function(
        function=example_request_function,
        name="python",
        language="python",
    )

    ade = AutoDocServingInfo(
        name="GLN_1_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[example_request_module],
    )

    return ade


def example_request_function():
    import base64

    import numpy as np
    import requests

    def encode_numpy_array(file_path: str) -> str:
        array = np.load(file_path)
        encoded = base64.b64encode(array.tobytes()).decode("utf-8")
        return encoded

    def send_request(url: str, payload: list[dict]):
        response = requests.post(url, json=payload)
        return response.json()

    encoded_data = encode_numpy_array(
        file_path="eir_tutorials/a_using_eir/01_basic_tutorial/data/"
        "processed_sample_data/arrays/A_French-4.DG.npy"
    )
    response = send_request(
        url="http://localhost:8000/predict", payload=[{"genotype": encoded_data}]
    )
    print(response)

    # --skip-after
    return response


def get_experiments() -> Sequence[AutoDocExperimentInfo | AutoDocServingInfo]:
    exp_1 = get_tutorial_01_run_1_gln_info()
    exp_2 = get_tutorial_01_run_2_gln_info()
    exp_3 = get_tutorial_01_run_2_gln_predict_info()
    exp_4 = get_tutorial_01_run_3_gln_predict_info()
    exp_5 = get_tutorial_01_run_3_serve()

    return [exp_1, exp_2, exp_3, exp_4, exp_5]
