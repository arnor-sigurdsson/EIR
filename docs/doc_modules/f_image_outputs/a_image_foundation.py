import base64
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from shutil import copytree

import numpy as np
from PIL import Image
from torchvision import transforms

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.f_image_outputs.utils import get_content_root
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command, get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "01_image_foundation"


def train_image_gen_01_image_autoencoder() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = "eir_tutorials/tutorial_runs/f_image_output/01_image_foundation"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_cnn.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_image.yaml",
        f"--globals.basic_experiment.output_folder={run_output_folder}",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_0_pretrain.pdf",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/0_autoencoder/examples/auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/0_autoencoder/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/35000/auto/{i}_generated.png",
                f"figures/0_autoencoder/examples/auto_generated_iter_35000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/35000/auto/{i}_inputs/image.png",
                f"figures/0_autoencoder/examples/auto_inputs_iter_35000_{i}.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_coco.zip")

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

    copy_run_folder = (
        _copy_run_folder_to_data_path,
        {
            "src": run_output_folder,
            "dst": str(data_output_path.parent / "01_image_foundation"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="0_IMAGE_FOUNDATION_PRETRAIN",
        data_url="https://drive.google.com/file/d/1T7mIRM6QIXFGHOfMgMbDp7WeTpGfwsgz",
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


def get_image_gen_02_image_autoencoder_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    image_base = "docs/tutorials/tutorial_files/f_image_output/_static"
    _add_small_altercation_versions(static_folder=image_base)
    example_requests = [
        [
            {"image": f"{image_base}/image_0.png"},
            {"image": f"{image_base}/image_1.png"},
            {"image": f"{image_base}/image_2.png"},
        ]
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/f_image_output/01_image_foundation",
    )

    copy_inputs_to_serve = (
        copy_inputs,
        {
            "example_requests": example_requests[0],
            "output_folder": str(Path(base_path) / "serve_results"),
        },
    )

    decode_and_save_images_func = (
        decode_and_save_images,
        {
            "predictions_file": str(
                Path(base_path) / "serve_results" / "predictions.json"
            ),
            "output_folder": str(Path(base_path) / "serve_results"),
            "tensor_shape": (
                3,
                128,
                128,
            ),
        },
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    ade = AutoDocServingInfo(
        name="ARRAY_GENERATION_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(copy_inputs_to_serve, decode_and_save_images_func),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
        ],
    )

    return ade


def example_request_function_python():
    import base64

    import requests

    def encode_image_to_base64(file_path: str) -> str:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
            return base64.b64encode(image_bytes).decode("utf-8")

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    image_base = "eir_tutorials/f_image_output/01_image_foundation/data/images"
    payload = [
        {"image": encode_image_to_base64(f"{image_base}/000000000009.jpg")},
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def _add_small_altercation_versions(static_folder: str) -> None:
    transformations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 30)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        ]
    )

    for image_path in Path(static_folder).glob("*.png"):
        if "altered" in image_path.stem:
            continue

        image = Image.open(image_path).convert("RGB")

        altered_image = transformations(image)

        altered_image_path = image_path.stem + "_altered.png"
        altered_image.save(Path(static_folder) / altered_image_path)


def decode_and_save_images(
    predictions_file: str,
    output_folder: str,
    tensor_shape: tuple,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    with open(predictions_file) as file:
        predictions = json.load(file)[0]

    for i, prediction in enumerate(predictions["response"]["result"]):
        base64_array = prediction["image"]
        array_bytes = base64.b64decode(base64_array)

        array_np = np.frombuffer(array_bytes, dtype=np.float32).reshape(tensor_shape)

        array_np = np.transpose(array_np, (1, 2, 0))
        array_np = np.clip(array_np, 0.0, 1.0)
        array_np = (array_np * 255).astype(np.uint8)

        image = Image.fromarray(array_np, mode="RGB")
        image.save(Path(output_folder) / f"image_output_{i}.png")


def _copy_run_folder_to_data_path(src: str, dst: str) -> None:
    if not Path(dst).exists():
        Path(dst).mkdir(parents=True, exist_ok=True)

    copytree(src, dst, dirs_exist_ok=True)


def get_downloaded_foundation_run_folder_path() -> str:
    return "eir_tutorials/tutorial_runs/f_image_output/01_image_foundation"


def _get_model_path_for_predict() -> str:
    run_1_output_path = get_downloaded_foundation_run_folder_path()
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_image_gen_01_image_autoencoder()
    exp_2 = get_image_gen_02_image_autoencoder_serve()

    return [
        exp_1,
        exp_2,
    ]
