from collections.abc import Sequence
from pathlib import Path

from docs.doc_modules.e_pretraining.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.utils import get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "01_checkpointing"


def train_01_imdb_from_scratch_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/imdb_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/imdb_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/imdb_output.yaml",
        f"--imdb_globals.basic_experiment.output_folder="
        f"eir_tutorials/tutorial_runs/{CR}/{TN}/",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_1_text_from_scratch.pdf",
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
                "*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="1_CHECKPOINT_PRETRAIN_IMDB_FROM_SCRATCH",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def train_02_imdb_from_pretrained_global_loading() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/imdb_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/imdb_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/imdb_output.yaml",
        "--imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/01_checkpointing_imdb_from_pretrained_global",
        "--imdb_globals.model.pretrained_checkpoint=FILL_MODEL",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_2_text_from_global_pretrained.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/imdb.zip")

    ade = AutoDocExperimentInfo(
        name="2_CHECKPOINTING_IMDB_FROM_PRETRAINED_GLOBAL",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
    )

    return ade


def train_03_imdb_from_pretrained_global_loading_non_strict() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/imdb_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/imdb_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/imdb_output.yaml",
        "--imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/01_checkpointing_imdb_from_pretrained_global_non_strict",
        "--imdb_fusion.model_config.fc_task_dim=64",
        "--imdb_globals.model.pretrained_checkpoint=FILL_MODEL",
        "--imdb_globals.model.strict_pretrained_loading=False",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_3_text_from_global_pretrained_non_strict.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/imdb.zip")

    ade = AutoDocExperimentInfo(
        name="3_CHECKPOINTING_IMDB_FROM_PRETRAINED_GLOBAL_NON_STRICT",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}/"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_01_imdb_from_scratch_model()
    exp_2 = train_02_imdb_from_pretrained_global_loading()
    exp_3 = train_03_imdb_from_pretrained_global_loading_non_strict()

    experiments = [
        exp_1,
        exp_2,
        exp_3,
    ]

    return experiments
