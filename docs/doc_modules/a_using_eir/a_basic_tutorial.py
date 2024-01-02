import textwrap
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from docs.doc_modules.deploy_experiments_utils import load_data_for_deploy
from docs.doc_modules.deployment_experiments import AutoDocDeploymentInfo
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
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
        "--tutorial_01_globals.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_01_run_lr-0.002_epochs-20",
        "--tutorial_01_globals.lr=0.002",
        "--tutorial_01_globals.n_epochs=20",
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

    run_1_output_path = "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run"

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


def _get_model_path_for_predict() -> str:
    run_1_output_path = "eir_tutorials/tutorial_runs/a_using_eir/tutorial_01_run"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: List[str]) -> List[str]:
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


def get_tutorial_01_run_3_deploy() -> AutoDocDeploymentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/01_basic_tutorial"

    model_path_placeholder = "FILL_MODEL"

    server_command = ["eirdeploy", "--model-path", model_path_placeholder]

    base = (
        "eir_tutorials/a_using_eir/01_basic_tutorial/data/processed_sample_data/arrays"
    )
    example_requests = [
        {"genotype": f"{base}/A374.npy"},
        {"genotype": f"{base}/Ayodo_468C.npy"},
        {"genotype": f"{base}/NOR146.npy"},
    ]

    ade = AutoDocDeploymentInfo(
        name="GLN_1_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
        example_requests=example_requests,
        data_loading_function=load_data_for_deploy,
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo | AutoDocDeploymentInfo]:
    exp_1 = get_tutorial_01_run_1_gln_info()
    exp_2 = get_tutorial_01_run_2_gln_info()
    exp_3 = get_tutorial_01_run_2_gln_predict_info()
    exp_4 = get_tutorial_01_run_3_deploy()

    return [exp_1, exp_2, exp_3, exp_4]
