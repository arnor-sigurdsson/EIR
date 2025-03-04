import base64
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from shutil import copytree

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.f_image_outputs.utils import get_content_root
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "03_mnist_diffusion"


def train_image_col_01_mnist_diffusion() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = "eir_tutorials/tutorial_runs/f_image_output/03_mnist_diffusion"

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
            "figures/training_curve_LOSS_01_MNIST_DIFFUSION.pdf",
        ),
    ]

    for i in range(9):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/01_mnist_diffusion/examples/auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/01_mnist_diffusion/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_generated.png",
                f"figures/01_mnist_diffusion/examples/auto_generated_iter_9000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_inputs/image.png",
                f"figures/01_mnist_diffusion/examples/auto_inputs_iter_9000_{i}.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_mnist.zip")

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

    plot_manual_generated_and_inputs_1k = (
        plot_unconditional_diffusion_grid,
        {
            "input_folder": Path(base_path) / "figures/01_mnist_diffusion/examples",
            "output_folder": Path(base_path) / "figures/01_mnist_diffusion",
            "iteration": 1000,
        },
    )

    plot_manual_generated_and_inputs_9k = (
        plot_unconditional_diffusion_grid,
        {
            "input_folder": Path(base_path) / "figures/01_mnist_diffusion/examples",
            "output_folder": Path(base_path) / "figures/01_mnist_diffusion",
            "iteration": 9000,
        },
    )

    ade = AutoDocExperimentInfo(
        name="01_MNIST_DIFFUSION",
        data_url="https://drive.google.com/file/d/1B0TRnOzV6zytEkEN9-QRGBNkbptp05sN",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            plot_manual_generated_and_inputs_1k,
            plot_manual_generated_and_inputs_9k,
        ),
    )

    return ade


def train_image_col_02_mnist_guided_diffusion() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_output_folder = (
        "eir_tutorials/tutorial_runs/f_image_output/03_mnist_diffusion_guided"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_cnn.yaml",
        f"{conf_output_path}/inputs_tabular.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_image.yaml",
        f"--globals.basic_experiment.output_folder={run_output_folder}",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_02_MNIST_DIFFUSION_GUIDED.pdf",
        ),
    ]

    for i in range(9):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/02_mnist_diffusion_guided/examples/"
                f"auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/02_mnist_diffusion_guided/examples/"
                f"auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_generated.png",
                f"figures/02_mnist_diffusion_guided/examples/"
                f"auto_generated_iter_9000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_inputs/image.png",
                f"figures/02_mnist_diffusion_guided/examples/"
                f"auto_inputs_iter_9000_{i}.png",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/image_mnist.zip")

    plot_manual_generated_and_inputs = (
        plot_conditional_diffusion_grid,
        {
            "input_folder": Path(base_path) / "serve_results",
            "output_folder": Path(base_path) / "figures/02_mnist_diffusion_guided",
        },
    )

    ade = AutoDocExperimentInfo(
        name="02_MNIST_DIFFUSION_GUIDED",
        data_url="https://drive.google.com/file/d/1B0TRnOzV6zytEkEN9-QRGBNkbptp05sN",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(plot_manual_generated_and_inputs,),
    )

    return ade


def plot_unconditional_diffusion_grid(
    input_folder: Path, output_folder: Path, iteration: int
) -> None:
    plt.figure(figsize=(12, 12))

    for i in range(9):
        output_file = input_folder / f"auto_generated_iter_{iteration}_{i}.png"
        output_image = plt.imread(output_file)

        plt.subplot(3, 3, i + 1)
        plt.imshow(output_image, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    output_folder.mkdir(exist_ok=True)
    plt.savefig(output_folder / f"unconditional_diffusion_grid_{iteration}.png")
    plt.close()


def plot_conditional_diffusion_grid(input_folder: Path, output_folder: Path) -> None:
    plt.figure(figsize=(10, 25))

    for i in range(10):
        output_file = input_folder / f"image_output_{i}.png"
        label_file = input_folder / f"mnist_tabular_{i}.json"

        output_image = plt.imread(output_file)

        with open(label_file) as f:
            label = json.load(f)["CLASS"]

        plt.subplot(5, 2, i + 1)
        plt.imshow(output_image, cmap="gray")
        plt.axis("off")
        plt.title(f"Class Condition: {label}")

    plt.tight_layout()
    output_folder.mkdir(exist_ok=True)
    plt.savefig(output_folder / "conditional_diffusion_grid.png")
    plt.close()


def mnist_diffusion_03_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    serve_command = [
        "eirserve",
        "--model-path",
        "FILL_MODEL",
    ]

    image_base = (
        "docs/tutorials/tutorial_files/f_image_output/_static/03_mnist_diffusion"
    )
    example_requests = [[]]
    for i in range(10):
        example_requests[0].append(
            {
                "image": f"{image_base}/base.png",
                "mnist_tabular": {"CLASS": f"{i}"},
            },
        )

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/f_image_output/03_mnist_diffusion_guided",
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
                1,
                28,
                28,
            ),
        },
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    ade = AutoDocServingInfo(
        name="02_MNIST_DIFFUSION_SERVE",
        base_path=Path(base_path),
        server_command=serve_command,
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

    image_base = "eir_tutorials/f_image_output/03_mnist_diffusion/data/data/images"
    payload = [
        {
            "image": encode_image_to_base64(f"{image_base}/00000.png"),
            "mnist_tabular": {"CLASS": "0"},
        },
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

    with open(predictions_file) as file:
        predictions = json.load(file)[0]

    for i, prediction in enumerate(predictions["response"]["result"]):
        base64_array = prediction["image"]
        array_bytes = base64.b64decode(base64_array)

        array_np = np.frombuffer(array_bytes, dtype=np.float32).reshape(tensor_shape)

        array_np = np.transpose(array_np, (1, 2, 0))
        array_np = np.clip(array_np, 0.0, 1.0)
        array_np = (array_np * 255).astype(np.uint8)

        image = Image.fromarray(array_np.squeeze(), mode="L")
        image.save(Path(output_folder) / f"image_output_{i}.png")


def _copy_run_folder_to_data_path(src: str, dst: str) -> None:
    if not Path(dst).exists():
        Path(dst).mkdir(parents=True, exist_ok=True)

    copytree(src, dst, dirs_exist_ok=True)


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_image_col_01_mnist_diffusion()
    exp_2 = train_image_col_02_mnist_guided_diffusion()
    exp_3 = mnist_diffusion_03_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
