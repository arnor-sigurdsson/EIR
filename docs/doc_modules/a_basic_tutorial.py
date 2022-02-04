from pathlib import Path
from typing import Sequence

from .experiments import AutoDocExperimentInfo, run_capture_and_save


def get_tutorial_01_run_1_gln_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/01_basic_tutorial"

    conf_output_path = "eir_tutorials/01_basic_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--target_configs",
        f"{conf_output_path}/tutorial_01_targets.yaml",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_01_training_curve_ACC_gln_1.pdf"),
        ("600/confusion_matrix", "figures/tutorial_01_confusion_matrix_gln_1.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/01_basic_tutorial/data/processed_sample_data.zip"
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
                "eir_tutorials/01_basic_tutorial/",
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
                "eir_tutorials/tutorial_runs/tutorial_01_run/",
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
        data_url="https://drive.google.com/file/d/1uzOR7-kZDHMsyhkzFdG9APYHVVf5fzMl",
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
    base_path = "docs/tutorials/tutorial_files/01_basic_tutorial"

    conf_output_path = "eir_tutorials/01_basic_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01_input.yaml",
        "--target_configs",
        f"{conf_output_path}/tutorial_01_targets.yaml",
        "--tutorial_01_globals.output_folder=eir_tutorials/tutorial_runs"
        "/tutorial_01_run_lr=0.002_epochs=20",
        "--tutorial_01_globals.lr=0.002",
        "--tutorial_01_globals.n_epochs=20",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_01_training_curve_ACC_gln_2.pdf"),
        ("600/confusion_matrix", "figures/tutorial_01_confusion_matrix_gln_2.pdf"),
    ]

    ade = AutoDocExperimentInfo(
        name="GLN_2",
        data_url="https://drive.google.com/file/d/1uzOR7-kZDHMsyhkzFdG9APYHVVf5fzMl",
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


def get_tutorial_01_run_3_linear_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/01_basic_tutorial"

    conf_output_path = "eir_tutorials/01_basic_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/tutorial_01_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/tutorial_01b_input_identity.yaml",
        "--target_configs",
        f"{conf_output_path}/tutorial_01_targets.yaml",
        "--predictor_configs",
        f"{conf_output_path}/tutorial_01b_predictor_linear.yaml",
        "--tutorial_01_globals.output_folder=eir_tutorials/tutorial_runs"
        "/tutorial_01_run_linear",
        "--tutorial_01_globals.n_epochs=20",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_01_training_curve_ACC_linear_1.pdf"),
    ]

    ade = AutoDocExperimentInfo(
        name="LINEAR_1",
        data_url="https://drive.google.com/file/d/1uzOR7-kZDHMsyhkzFdG9APYHVVf5fzMl",
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


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_tutorial_01_run_1_gln_info()
    exp_2 = get_tutorial_01_run_2_gln_info()
    exp_3 = get_tutorial_01_run_3_linear_info()

    return [exp_1, exp_2, exp_3]
