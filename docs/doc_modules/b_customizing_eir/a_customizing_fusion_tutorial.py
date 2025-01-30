from collections.abc import Sequence
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


def get_tutorial_01_run_1_fusion_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/b_customizing_eir/01_customizing_fusion_tutorial"
    )

    conf_output_path = "eir_tutorials/b_customizing_eir/01_customizing_fusion/conf"

    command = [
        "python",
        "docs/doc_modules/b_customizing_eir/a_customizing_fusion.py",
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
                "eir_tutorials/tutorial_runs/b_customizing_eir/tutorial_01_run",
                "-I",
                "tensorboard_logs|serializations|transformers|*.yaml|*.pt|*.log",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/experiment_01_folder.txt",
        },
    )
    get_model_info = (
        run_capture_and_save,
        {
            "command": [
                "cat",
                "eir_tutorials/tutorial_runs/b_customizing_eir/tutorial_01_run/"
                "model_info.txt",
            ],
            "output_path": Path(base_path) / "commands/model_info.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="CUSTOM_FUSION",
        data_url="https://drive.google.com/file/d/1MELauhv7zFwxM8nonnj3iu_SmS69MuNi",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            get_run_1_experiment_folder,
            get_model_info,
        ),
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_tutorial_01_run_1_fusion_info()

    return [exp_1]
