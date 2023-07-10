from pathlib import Path
from typing import Sequence

from aislib.misc_utils import ensure_path_exists

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.c_sequence_outputs.utils import get_content_root
from docs.doc_modules.utils import get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "01_sequence_generation"


def get_sequence_gen_01_imdb_generation() -> AutoDocExperimentInfo:
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
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1.pdf",
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
            "samples/9500/auto/0_generated.txt",
            "figures/auto_generated_iter_9500.txt",
        ),
        (
            "samples/9500/manual/1_generated.txt",
            "figures/manual_generated_iter_9500.txt",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/imdb.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}",
                "-L",
                "3",
                "-I",
                "*.zip|*Anticancer*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_GENERATION_IMDB_1",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_sequence_gen_01_imdb_generation_predict() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}/test_results"
    ensure_path_exists(path=Path(run_1_output_path), is_folder=True)

    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_test.yaml",
        "--model_path",
        "FILL_MODEL",
        "--output_folder",
        run_1_output_path,
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/imdb.zip")

    mapping = [
        (
            "test_results/results/imdb_output/imdb_output/"
            "samples/0/auto/0_generated.txt",
            "tutorial_data/test_results/auto_0.txt",
        ),
        (
            "test_results/results/imdb_output/imdb_output/"
            "samples/0/manual/10_generated.txt",
            "tutorial_data/test_results/manual_0.txt",
        ),
        (
            "test_results/results/imdb_output/imdb_output/"
            "samples/0/manual/11_generated.txt",
            "tutorial_data/test_results/manual_1.txt",
        ),
    ]

    ade = AutoDocExperimentInfo(
        name="01_PREDICT_GENERATION",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
        force_run_command=True,
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_sequence_gen_01_imdb_generation()
    exp_2 = get_sequence_gen_01_imdb_generation_predict()

    return [
        exp_1,
        exp_2,
    ]
