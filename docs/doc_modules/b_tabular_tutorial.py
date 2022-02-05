from pathlib import Path
from typing import Sequence

from .experiments import AutoDocExperimentInfo, run_capture_and_save


def get_02_poker_hands_run_1_tabular_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/02_tabular_tutorial"

    conf_output_path = "eir_tutorials/02_tabular_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/02_poker_hands_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/02_poker_hands_input.yaml",
        "--predictor_configs",
        f"{conf_output_path}/02_poker_hands_predictor.yaml",
        "--target_configs",
        f"{conf_output_path}/02_poker_hands_target.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/02_poker_hands_training_curve_ACC_tabular_1.pdf",
        ),
        (
            "training_curve_MCC",
            "figures/02_poker_hands_training_curve_MCC_tabular_1.pdf",
        ),
        (
            "15000/confusion_matrix",
            "figures/02_poker_hands_confusion_matrix_tabular_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/02_tabular_tutorial/data/poker_hands_data.zip"
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/02_tabular_tutorial/",
                "-I",
                "*test*.yaml|*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="TABULAR_1",
        data_url="https://drive.google.com/file/d/1Ck1F_iYT3WdoAHjtPwR1peOqhwjmCqHl",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_02_poker_hands_run_1_predict_info() -> AutoDocExperimentInfo:
    """
    We are abusing the `make_tutorial_data` here a bit by switching to the predict
    code, but we'll allow it for now.
    """
    base_path = "docs/tutorials/tutorial_files/02_tabular_tutorial"

    conf_output_path = "eir_tutorials/02_tabular_tutorial/conf"

    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/02_poker_hands_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/02_poker_hands_input_test.yaml",
        "--predictor_configs",
        f"{conf_output_path}/02_poker_hands_predictor.yaml",
        "--target_configs",
        f"{conf_output_path}/02_poker_hands_target_test.yaml",
        "--model_path",
        "eir_tutorials/tutorial_runs/tutorial_02_run/saved_models"
        "/tutorial_02_run_model_15000_perf-average=0.9815.pt",
        "--evaluate",
        "--output_folder",
        "eir_tutorials/tutorial_runs/tutorial_02_run",
    ]

    data_output_path = Path(
        "eir_tutorials/02_tabular_tutorial/data/poker_hands_data.zip"
    )

    mapping = [
        (
            "calculated_metrics",
            "tutorial_data/calculated_metrics_test.json",
        ),
    ]

    ade = AutoDocExperimentInfo(
        name="TABULAR_1_PREDICT",
        data_url="https://drive.google.com/file/d/1Ck1F_iYT3WdoAHjtPwR1peOqhwjmCqHl",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
        force_run_command=True,
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_02_poker_hands_run_1_tabular_info()
    exp_2 = get_02_poker_hands_run_1_predict_info()

    return [exp_1, exp_2]
