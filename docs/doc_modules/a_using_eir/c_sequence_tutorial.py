from pathlib import Path
from typing import Sequence

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


def get_03_imdb_run_1_transformer_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/03_sequence_tutorial/a_IMDB"

    conf_output_path = "eir_tutorials/03_sequence_tutorial/a_IMDB/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/03a_imdb_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/03a_imdb_input.yaml",
        "--output_configs",
        f"{conf_output_path}/03a_imdb_output.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/03a_imdb_training_curve_ACC_transformer_1.pdf",
        ),
        (
            "4000/activations/imdb_reviews/Positive/token_influence_Positive.pdf",
            "figures/tutorial_03a_feature_importance_Positive.pdf",
        ),
        (
            "4000/activations/imdb_reviews/Negative/token_influence_Negative.pdf",
            "figures/tutorial_03a_feature_importance_Negative.pdf",
        ),
        (
            ".*4000/activations/imdb_reviews/single_samples.html",
            "figures/tutorial_03a_single_samples_example.html",
        ),
    ]

    data_output_path = Path("eir_tutorials/03_sequence_tutorial/data/imdb.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/03_sequence_tutorial/",
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
        name="SEQUENCE_IMDB_1",
        data_url="https://drive.google.com/file/d/1u6bkIr9sECkU9z3Veutjn8cx6Mu3GP3Z",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_03_peptides_run_1_transformer_info() -> AutoDocExperimentInfo:
    base_path = (
        "docs/tutorials/tutorial_files/03_sequence_tutorial/b_Anticancer_peptides"
    )

    conf_output_path = "eir_tutorials/03_sequence_tutorial/b_Anticancer_peptides/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/03b_peptides_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/03b_peptides_input.yaml",
        "--output_configs",
        f"{conf_output_path}/03b_peptides_output.yaml",
    ]

    mapping = [
        (
            "training_curve_MCC",
            "figures/03b_peptides_training_curve_MCC_transformer_1.pdf",
        ),
        (
            "2000/confusion_matrix",
            "figures/03b_peptides_confusion_matrix_1.pdf",
        ),
        (
            "2400/activations/peptide_sequences/mod. active/token_influence_mod. "
            "active.pdf",
            "figures/tutorial_03b_feature_importance_mod._active.pdf",
        ),
        (
            ".*2400/activations/peptide_sequences/single_samples.html",
            "figures/tutorial_03b_single_samples.html",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/03_sequence_tutorial/data/Anticancer_Peptides.zip"
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/03_sequence_tutorial/",
                "-L",
                "3",
                "-I",
                "*.zip|*IMDB*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_PEPTIDES_1",
        data_url="https://drive.google.com/file/d/12vHW1V8hhIuasih_gWPn7xHmZZTAd22Q",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_03_imdb_run_1_transformer_info()
    exp_2 = get_03_peptides_run_1_transformer_info()

    return [exp_1, exp_2]
