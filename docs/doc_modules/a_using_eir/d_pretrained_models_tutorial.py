from pathlib import Path
from typing import Sequence

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


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
        "--04_imdb_globals.output_folder=eir_tutorials/tutorial_runs"
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
        "--04_imdb_globals.output_folder=eir_tutorials/tutorial_runs"
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
        "--04_imdb_globals.output_folder=eir_tutorials/tutorial_runs"
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
        "--04_imdb_globals.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_04_imdb_run_combined",
        "--04_imdb_globals.device='cpu'",
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


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_04_imdb_run_1_transformer_info()
    exp_2 = get_04_imdb_run_2_local_transformer_info()
    exp_3 = get_04_imdb_run_3_longformer_info()
    exp_4 = get_04_imdb_run_4_tiny_bert_info()
    exp_5 = get_04_imdb_run_5_combined_info()

    return [exp_1, exp_2, exp_3, exp_4, exp_5]
