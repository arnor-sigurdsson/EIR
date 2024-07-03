import base64
import json
import os
from functools import partial
from pathlib import Path
from shutil import copytree
from typing import Sequence

import numpy as np
from PIL import Image

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.f_image_outputs.utils import get_content_root
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command, get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "02_image_colorization"


def train_image_col_01_one_shot_colorization() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = (
        "eir_tutorials/tutorial_runs/f_image_output/02_image_colorization"
    )

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
        f"--globals.output_folder={run_output_folder}",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_00_image_colorization.pdf",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/00_image_colorization/examples/"
                f"auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/00_image_colorization/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/8000/auto/{i}_generated.png",
                f"figures/00_image_colorization/examples/"
                f"auto_generated_iter_8000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/8000/auto/{i}_inputs/image.png",
                f"figures/00_image_colorization/examples/auto_inputs_iter_8000_{i}.png",
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

    ade = AutoDocExperimentInfo(
        name="00_IMAGE_COLORIZATION",
        data_url="https://drive.google.com/file/d/1T7mIRM6QIXFGHOfMgMbDp7WeTpGfwsgz",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_foundation_run_folder_path() -> str:
    return "eir_tutorials/tutorial_runs/f_image_output/01_image_foundation"


def _get_model_path_for_predict() -> str:
    run_1_output_path = get_foundation_run_folder_path()
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def train_image_col_02_one_shot_super_resolution() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = (
        "eir_tutorials/tutorial_runs/f_image_output/02_image_super_resolution"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_cnn_super_res.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_image_super_res.yaml",
        f"--globals.output_folder={run_output_folder}",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_01_super_resolution.pdf",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/01_super_resolution/examples/"
                f"auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/01_super_resolution/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/6000/auto/{i}_generated.png",
                f"figures/01_super_resolution/examples/"
                f"auto_generated_iter_6000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/6000/auto/{i}_inputs/image.png",
                f"figures/01_super_resolution/examples/auto_inputs_iter_6000_{i}.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_coco.zip")

    ade = AutoDocExperimentInfo(
        name="01_IMAGE_SUPER_RESOLUTION",
        data_url="https://drive.google.com/file/d/1T7mIRM6QIXFGHOfMgMbDp7WeTpGfwsgz",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def train_image_col_03_both() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = (
        "eir_tutorials/tutorial_runs/f_image_output/"
        "02_image_colorization_and_super_resolution"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_cnn_both.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_image_super_res.yaml",
        f"--globals.output_folder={run_output_folder}",
        "--globals.n_epochs=20",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_02_both.pdf",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/02_both/examples/auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/02_both/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/25000/auto/{i}_generated.png",
                f"figures/02_both/examples/auto_generated_iter_25000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/25000/auto/{i}_inputs/image.png",
                f"figures/02_both/examples/auto_inputs_iter_25000_{i}.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_coco.zip")

    ade = AutoDocExperimentInfo(
        name="02_IMAGE_BOTH",
        data_url="https://drive.google.com/file/d/1T7mIRM6QIXFGHOfMgMbDp7WeTpGfwsgz",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def train_image_col_04_both_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    image_base = (
        "docs/tutorials/tutorial_files/f_image_output/_static/02_image_colorization"
    )
    example_requests = [
        [
            {"image": f"{image_base}/image_grayscale_0.png"},
            {"image": f"{image_base}/image_grayscale_1.jpg"},
        ]
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/f_image_output/"
        "02_image_colorization_and_super_resolution",
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
        name="04_IMAGE_BOTH_SERVE",
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

    image_base = "eir_tutorials/f_image_output/02_image_colorization/data/images"
    payload = [
        {"image": encode_image_to_base64(f"{image_base}/000000000009.jpg")},
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def decode_and_save_images(
    predictions_file: str,
    output_folder: str,
    tensor_shape: tuple,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    with open(predictions_file, "r") as file:
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


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_image_col_01_one_shot_colorization()
    exp_2 = train_image_col_02_one_shot_super_resolution()
    exp_3 = train_image_col_03_both()
    exp_4 = train_image_col_04_both_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
        exp_4,
    ]
