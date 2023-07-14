import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from PIL import Image

from docs.doc_modules.c_sequence_outputs.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "03_image_captioning"


def get_image_captioning_01_text_only() -> AutoDocExperimentInfo:
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
        "/c_sequence_output/03_image_captioning_text_only",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_text.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_captions.zip")

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
        name="IMAGE_CAPTIONING_ONLY_TEXT",
        data_url="https://drive.google.com/file/d/10zanaprFyX4RE0Mib1h5gYi7yO9DNTyy",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_image_captioning_02_image_and_text() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_resnet18.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_image_text.pdf",
        ),
    ]

    for i in range(10):
        mapping.append(
            (
                f"samples/11000/auto/{i}_generated.txt",
                f"figures/auto_generated_{i}_iter_11000_caption.txt",
            )
        )
        mapping.append(
            (
                f"samples/11000/auto/{i}_inputs/image_captioning.png",
                f"figures/auto_input_{i}_iter_11000_image.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_captions.zip")

    save_image_func = (
        generate_image_grid,
        {
            "input_folder": Path(base_path, "figures"),
            "output_folder": Path(base_path, "figures"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="IMAGE_CAPTIONING_IMAGE_TEXT",
        data_url="https://drive.google.com/file/d/10zanaprFyX4RE0Mib1h5gYi7yO9DNTyy",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(save_image_func,),
    )

    return ade


def generate_image_grid(
    input_folder: Path,
    output_folder: Path,
    filename: str = "captions.png",
) -> None:
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))

    for i in range(5):
        for j in range(2):
            index = 2 * i + j
            image_path = input_folder / f"auto_input_{index}_iter_11000_image.png"
            caption_path = (
                input_folder / f"auto_generated_{index}_iter_11000_caption.txt"
            )

            img = Image.open(image_path)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")

            with open(caption_path, "r") as f:
                caption = f.read()

            wrapped_caption = textwrap.fill(caption, 30)
            axs[i, j].set_title(wrapped_caption)

    plt.tight_layout()
    plt.savefig(output_folder / filename, bbox_inches="tight", dpi=200)


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_image_captioning_01_text_only()
    exp_2 = get_image_captioning_02_image_and_text()

    return [
        exp_1,
        exp_2,
    ]
