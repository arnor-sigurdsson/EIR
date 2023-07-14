from pathlib import Path
from typing import Sequence

import pandas as pd

from docs.doc_modules.c_sequence_outputs.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "02_sequence_to_sequence"


def get_sequence_to_sequence_01_english_only() -> AutoDocExperimentInfo:
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
        "--globals.output_folder=eir_tutorials/tutorial_runs"
        "/c_sequence_output/02_seq_to_seq_eng_only",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_only_english.pdf",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/13500/auto/{i}_generated.txt",
                f"figures/auto_generated_{i}_iter_13500_only_english.txt",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/english_spanish.zip")

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
        name="SEQUENCE_TO_SEQUENCE_ENGLISH",
        data_url="https://drive.google.com/file/d/1MIARnMmzYNPEDU_f7cuPwaHp8BsXNy59",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_sequence_to_sequence_02_spanish_to_english() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_spanish.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_spanish_to_english.pdf",
        ),
    ]

    for i in range(10):
        mapping.append(
            (
                f"samples/13500/auto/{i}_generated.txt",
                f"figures/auto_generated_{i}_iter_13500_spanish_to_english.txt",
            )
        )
        mapping.append(
            (
                f"samples/13500/auto/{i}_inputs/spanish.txt",
                f"figures/auto_input_{i}_iter_13500_spanish_to_english.txt",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/english_spanish.zip")

    save_translations_func = (
        save_translations,
        {
            "folder_path": Path(base_path, "figures"),
            "output_folder": Path(base_path, "figures"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="SEQUENCE_TO_SEQUENCE_SPANISH_ENGLISH",
        data_url="https://drive.google.com/file/d/1MIARnMmzYNPEDU_f7cuPwaHp8BsXNy59",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(save_translations_func,),
    )

    return ade


def save_translations(folder_path: Path, output_folder: Path) -> None:
    files = [f for f in folder_path.iterdir() if f.name.endswith(".txt")]

    input_files = [f for f in files if "input" in f.name]
    generated_files = [
        f for f in files if "generated" in f.name and "spanish_to_english" in f.name
    ]
    assert len(input_files) == len(generated_files)

    input_files.sort()
    generated_files.sort()

    data = pd.DataFrame(columns=["Spanish", "English Translation"])

    for inp, gen in zip(input_files, generated_files):
        with open(inp, "r") as file:
            sentences = file.readlines()
        with open(gen, "r") as file:
            translations = file.readlines()

        temp_df = pd.DataFrame(
            {"Spanish": sentences, "English Translation": translations}
        )

        data = pd.concat([data, temp_df])

    data.reset_index(drop=True, inplace=True)

    data.to_html(output_folder / "translations.html")


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_sequence_to_sequence_01_english_only()
    exp_2 = get_sequence_to_sequence_02_spanish_to_english()

    return [
        exp_1,
        exp_2,
    ]
