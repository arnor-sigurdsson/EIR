from collections.abc import Sequence
from pathlib import Path
from shutil import copytree

from docs.doc_modules.e_pretraining.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.utils import get_saved_model_path
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "02_mini_foundation"


def train_01_mini_foundation_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = "eir_tutorials/tutorial_runs/e_pretraining/02_mini_foundation"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_array_cnn.yaml",
        f"{conf_output_path}/inputs_sequence.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_sequence.yaml",
        f"--globals.basic_experiment.output_folder={run_output_folder}",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_0_pretrain.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/mini_foundation.zip")

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

    copy_run_folder = (
        _copy_run_folder_to_data_path,
        {
            "src": run_output_folder,
            "dst": str(data_output_path.parent / "data" / "02_mini_foundation"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="0_MINI_FOUNDATION_PRETRAIN",
        data_url="https://drive.google.com/file/d/1WyTNS5RZ4o26F9wN66ahfdVqO4GauAvv",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            copy_run_folder,
        ),
    )

    return ade


def _copy_run_folder_to_data_path(src: str, dst: str) -> None:
    if not Path(dst).exists():
        logger.info(f"Creating folder at {dst}")
        Path(dst).mkdir(parents=True, exist_ok=True)

    copytree(src=src, dst=dst, dirs_exist_ok=True)


def train_02_imdb_from_scratch_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/imdb/imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/imdb/imdb_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/imdb/imdb_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/imdb/imdb_output.yaml",
        "--imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/02_mini_foundation_imdb_from_scratch",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_1_text_from_scratch.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/mini_foundation.zip")

    ade = AutoDocExperimentInfo(
        name="1_MINI_FOUNDATION_PRETRAIN_IMDB_FROM_SCRATCH",
        data_url="https://drive.google.com/file/d/1WyTNS5RZ4o26F9wN66ahfdVqO4GauAvv",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def train_03_imdb_from_pretrained_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/imdb/imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/imdb/imdb_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/imdb/imdb_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/imdb/imdb_output.yaml",
        "--imdb_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/02_mini_foundation_imdb_from_pretrained",
        "--imdb_input.pretrained_config.model_path=FILL_MODEL",
        "--imdb_input.pretrained_config.load_module_name=text",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_2_text_from_pretrain.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/mini_foundation.zip")

    ade = AutoDocExperimentInfo(
        name="2_MINI_FOUNDATION_PRETRAIN_IMDB_FROM_PRETRAINED",
        data_url="https://drive.google.com/file/d/1WyTNS5RZ4o26F9wN66ahfdVqO4GauAvv",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
    )

    return ade


def train_04_cifar_from_scratch_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/cifar/cifar_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/cifar/cifar_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/cifar/cifar_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/cifar/cifar_output.yaml",
        "--cifar_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/02_mini_foundation_cifar_from_scratch",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_3_image_from_scratch.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/mini_foundation.zip")

    ade = AutoDocExperimentInfo(
        name="3_MINI_FOUNDATION_PRETRAIN_CIFAR_FROM_SCRATCH",
        data_url="https://drive.google.com/file/d/1WyTNS5RZ4o26F9wN66ahfdVqO4GauAvv",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def train_05_cifar_from_pretrained_model() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/cifar/cifar_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/cifar/cifar_input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/cifar/cifar_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/cifar/cifar_output.yaml",
        "--cifar_globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs/"
        "e_pretraining/02_mini_foundation_cifar_from_pretrained",
        "--cifar_input.pretrained_config.model_path=FILL_MODEL",
        "--cifar_input.pretrained_config.load_module_name=image_input",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_4_image_from_pretrain.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/mini_foundation.zip")

    ade = AutoDocExperimentInfo(
        name="4_MINI_FOUNDATION_PRETRAIN_CIFAR_FROM_PRETRAINED",
        data_url="https://drive.google.com/file/d/1WyTNS5RZ4o26F9wN66ahfdVqO4GauAvv",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(),
    )

    return ade


def get_downloaded_foundation_run_folder_path() -> str:
    return "eir_tutorials/e_pretraining/02_mini_foundation/data/02_mini_foundation"


def _get_model_path_for_predict() -> str:
    run_1_output_path = get_downloaded_foundation_run_folder_path()
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_01_mini_foundation_model()
    exp_2 = train_02_imdb_from_scratch_model()
    exp_3 = train_03_imdb_from_pretrained_model()
    exp_4 = train_04_cifar_from_scratch_model()
    exp_5 = train_05_cifar_from_pretrained_model()

    return [
        exp_1,
        exp_2,
        exp_3,
        exp_4,
        exp_5,
    ]
