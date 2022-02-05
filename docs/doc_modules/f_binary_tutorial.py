from pathlib import Path
from typing import Sequence

from .experiments import AutoDocExperimentInfo, run_capture_and_save


def get_06_imdb_binary_run_1_transformer_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/06_raw_bytes_tutorial/"

    conf_output_path = "eir_tutorials/06_raw_bytes_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input.yaml",
        "--predictor_configs",
        f"{conf_output_path}/predictor.yaml",
        "--target_configs",
        f"{conf_output_path}/target.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/06_training_curve_ACC_transformer_1.pdf",
        ),
        (
            "training_curve_MCC",
            "figures/06_training_curve_MCC_transformer_1.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/06_raw_bytes_tutorial/data/imdb.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/06_raw_bytes_tutorial/",
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
        name="SEQUENCE_BINARY_IMDB_1",
        data_url="https://drive.google.com/file/d/1lYGVhynlhGSDHfLOnS7SqBVL_J3lvL64",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_06_imdb_binary_run_1_transformer_info()

    return [exp_1]

