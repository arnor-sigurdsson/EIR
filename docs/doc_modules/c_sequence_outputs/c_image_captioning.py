import textwrap
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from docs.doc_modules.c_sequence_outputs.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command

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
        "--globals.basic_experiment.output_folder=eir_tutorials/tutorial_runs"
        "/c_sequence_output/03_image_captioning_text_only",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_1_text.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/image_captions.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}",
                "-L",
                "2",
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
                f"samples/10000/auto/{i}_generated.txt",
                f"figures/auto_generated_{i}_iter_11000_caption.txt",
            )
        )
        mapping.append(
            (
                f"samples/10000/auto/{i}_inputs/image_captioning.png",
                f"figures/auto_input_{i}_iter_11000_image.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/image_captions.zip")

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

            with open(caption_path) as f:
                caption = f.read()

            wrapped_caption = textwrap.fill(caption, 30)
            axs[i, j].set_title(wrapped_caption)

    plt.tight_layout()
    plt.savefig(output_folder / filename, bbox_inches="tight", dpi=200)


def get_image_captioning_02_image_and_text_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    image_base = "eir_tutorials/c_sequence_output/03_image_captioning/data/images"
    example_requests = [
        [
            {"image_captioning": f"{image_base}/000000000009.jpg", "captions": ""},
            {"image_captioning": f"{image_base}/000000000034.jpg", "captions": ""},
            {
                "image_captioning": f"{image_base}/000000581929.jpg",
                "captions": "A horse",
            },
        ],
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/c_sequence_output/03_image_captioning",
    )

    copy_inputs_to_serve = (
        copy_inputs,
        {
            "example_requests": example_requests[0],
            "output_folder": str(Path(base_path) / "serve_results"),
        },
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    ade = AutoDocServingInfo(
        name="IMAGE_TO_SEQUENCE_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(copy_inputs_to_serve,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
        ],
    )

    return ade


def example_request_function_python():
    import base64
    from io import BytesIO

    import requests
    from PIL import Image

    def encode_image_to_base64(file_path: str) -> str:
        with Image.open(file_path) as image:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    image_base = "eir_tutorials/c_sequence_output/03_image_captioning/data/images"
    payload = [
        {
            "image_captioning": encode_image_to_base64(
                f"{image_base}/000000000009.jpg"
            ),
            "captions": "",
        },
        {
            "image_captioning": encode_image_to_base64(
                f"{image_base}/000000000034.jpg"
            ),
            "captions": "",
        },
        {
            "image_captioning": encode_image_to_base64(
                f"{image_base}/000000581929.jpg"
            ),
            "captions": "A horse",
        },
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_image_captioning_01_text_only()
    exp_2 = get_image_captioning_02_image_and_text()
    exp_3 = get_image_captioning_02_image_and_text_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
