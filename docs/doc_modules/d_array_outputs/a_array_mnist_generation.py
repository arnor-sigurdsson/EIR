import base64
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE

from docs.doc_modules.d_array_outputs.utils import get_content_root
from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command, get_saved_model_path
from eir.train_utils.latent_analysis import load_samples_for_viz

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "01_array_mnist_generation"


def get_array_gen_01_mnist_generation() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_mnist_array.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/0_autoencoder/training_curve_LOSS_1.pdf",
        ),
        (
            "latent_outputs/9000/fusion_modules.computed.fusion_modules.fusion.1.0/"
            "tsne.png",
            "figures/0_autoencoder/tsne_9000.png",
        ),
        (
            "latent_outputs/9000/fusion_modules.computed.fusion_modules.fusion.1.0/"
            "batch_00000.npy",
            "figures/0_autoencoder/latent_batches/batch_00000.npy",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.npy",
                f"figures/0_autoencoder/examples/auto_generated_iter_500_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/mnist.npy",
                f"figures/0_autoencoder/examples/auto_inputs_iter_500_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_generated.npy",
                f"figures/0_autoencoder/examples/auto_generated_iter_9000_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_inputs/mnist.npy",
                f"figures/0_autoencoder/examples/auto_inputs_iter_9000_{i}.npy",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/mnist_array.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}",
                "-L",
                "2",
                "-I",
                "*.zip|*Anticancer*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    plot_latents = (
        visualize_latents,
        {
            "label_file": data_output_path.parent / "mnist_labels.csv",
            "latents_folder": Path(base_path, "figures/0_autoencoder/latent_batches"),
            "output_folder": Path(base_path) / "figures/0_autoencoder",
        },
    )

    plot_generated_and_inputs = (
        create_comparison_figures,
        {
            "input_folder": Path(base_path) / "figures/0_autoencoder/examples",
            "output_folder": Path(base_path) / "figures/0_autoencoder",
        },
    )

    ade = AutoDocExperimentInfo(
        name="ARRAY_GENERATION_MNIST_1",
        data_url="https://drive.google.com/file/d/1q-ZBJJvLLW61AGBfYfLtKADvY4j4_OGb",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            plot_latents,
            plot_generated_and_inputs,
        ),
    )

    return ade


def visualize_latents(
    label_file: str,
    latents_folder: Path,
    output_folder: str,
    max_samples_for_tsne: int = 10000,
) -> None:
    labels_df = pd.read_csv(filepath_or_buffer=label_file)
    labels_df["ID"] = labels_df["ID"].astype(str)

    latents, ids = load_samples_for_viz(
        batch_dir=latents_folder,
        max_samples=max_samples_for_tsne,
    )

    latents_df = pd.DataFrame({"Latent": list(latents), "ID": ids})

    merged_df = pd.merge(latents_df, labels_df, on="ID", how="left")

    tsne = TSNE(n_components=2, random_state=42)
    latents_reduced = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))

    if "CLASS" not in merged_df.columns:
        plt.scatter(
            latents_reduced[:, 0],
            latents_reduced[:, 1],
        )
        plt.title("Latents Visualization with t-SNE")
    else:
        palette = sns.color_palette("tab10", n_colors=merged_df["CLASS"].nunique())
        sorted_labels = sorted(merged_df["CLASS"].unique())

        for label, color in zip(sorted_labels, palette, strict=False):
            subset = latents_reduced[merged_df["CLASS"] == label]
            if len(subset) > 0:
                plt.scatter(
                    subset[:, 0],
                    subset[:, 1],
                    c=[color],
                    label=f"{label}",
                )

        plt.title("Latents Visualization with t-SNE by Class")
        plt.legend(title="Class Label")

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(output_folder, "latents_visualization_tsne.png")
    plt.savefig(output_path)
    plt.close()


def create_comparison_figures(input_folder: Path, output_folder: Path) -> None:
    images_data = load_images_and_iterations(input_folder)

    for iteration in images_data["inputs"]:
        inputs = images_data["inputs"][iteration]
        generated = images_data["generated"][iteration]

        fig, axes = plt.subplots(len(inputs), 2, figsize=(5, 10))
        plt.suptitle(f"Iteration {iteration}")

        for i, (input_img, generated_img) in enumerate(
            zip(inputs, generated, strict=False)
        ):
            input_img = input_img.reshape(28, 28)
            generated_img = generated_img.reshape(28, 28)

            axes[i, 0].imshow(input_img, cmap="gray")
            axes[i, 0].axis("off")
            if i == 0:
                axes[i, 0].set_title("Input")

            axes[i, 1].imshow(generated_img, cmap="gray")
            axes[i, 1].axis("off")
            if i == 0:
                axes[i, 1].set_title("Generated")

        output_path = output_folder / f"comparison_iteration_{iteration}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


def load_images_and_iterations(
    folder_path: Path,
) -> dict:
    images_data = {"inputs": {}, "generated": {}}

    for file_name in os.listdir(folder_path):
        parts = file_name.split("_")
        image_type = parts[1]
        iteration = int(parts[3])
        index = int(parts[4].split(".")[0])

        image_path = os.path.join(folder_path, file_name)
        image_data = np.load(image_path).clip(0, 255).astype(np.uint8)

        if image_type == "inputs":
            images_data["inputs"].setdefault(iteration, [None] * 5)[index] = image_data
        elif image_type == "generated":
            images_data["generated"].setdefault(iteration, [None] * 5)[index] = (
                image_data
            )

    return images_data


def get_array_gen_02_mnist_generation() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_mnist_array_with_label.yaml",
        f"{conf_output_path}/input_mnist_label.yaml",
        "--output_configs",
        f"{conf_output_path}/output_with_label.yaml",
        "--globals.basic_experiment.output_folder="
        "eir_tutorials/tutorial_runs/d_array_output/"
        "02_array_mnist_generation_with_labels",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/1_autoencoder_augmented/training_curve_LOSS_1.pdf",
        ),
        (
            "latent_outputs/9000/fusion_modules.computed.fusion_modules.fusion.1.0/"
            "tsne.png",
            "figures/1_autoencoder_augmented/tsne_9000.png",
        ),
        (
            "latent_outputs/9000/fusion_modules.computed.fusion_modules.fusion.1.0/"
            "batch_00000.npy",
            "figures/1_autoencoder_augmented/latent_batches/batch_00000.npy",
        ),
    ]

    for i in range(5):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.npy",
                f"figures/1_autoencoder_augmented/examples/"
                f"auto_generated_iter_500_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/mnist.npy",
                f"figures/1_autoencoder_augmented/examples/"
                f"auto_inputs_iter_500_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_generated.npy",
                f"figures/1_autoencoder_augmented/examples/"
                f"auto_generated_iter_9000_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/9000/auto/{i}_inputs/mnist.npy",
                f"figures/1_autoencoder_augmented/examples/"
                f"auto_inputs_iter_9000_{i}.npy",
            )
        )

    for i in range(10, 14):
        mapping.append(
            (
                f"samples/9000/manual/{i}_generated.npy",
                f"figures/1_autoencoder_augmented/examples_manual/"
                f"manual_generated_iter_9000_{i}.npy",
            )
        )
        mapping.append(
            (
                f"samples/9000/manual/{i}_inputs/mnist_label.json",
                f"figures/1_autoencoder_augmented/examples_manual/"
                f"manual_inputs_iter_9000_{i}.json",
            )
        )

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/data/mnist_array.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}",
                "-L",
                "2",
                "-I",
                "*.zip|*Anticancer*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    plot_latents = (
        visualize_latents,
        {
            "label_file": data_output_path.parent / "mnist_labels.csv",
            "latents_folder": Path(base_path, "figures/0_autoencoder/latent_batches"),
            "output_folder": Path(base_path) / "figures/1_autoencoder_augmented",
        },
    )

    plot_generated_and_inputs = (
        create_comparison_figures,
        {
            "input_folder": Path(base_path)
            / "figures/1_autoencoder_augmented/examples",
            "output_folder": Path(base_path) / "figures/1_autoencoder_augmented",
        },
    )

    plot_manual_generated_and_inputs = (
        plot_inputs_and_images,
        {
            "input_folder": Path(base_path)
            / "figures/1_autoencoder_augmented/examples_manual",
            "output_folder": Path(base_path) / "figures/1_autoencoder_augmented",
        },
    )

    ade = AutoDocExperimentInfo(
        name="ARRAY_GENERATION_MNIST_2",
        data_url="https://drive.google.com/file/d/1q-ZBJJvLLW61AGBfYfLtKADvY4j4_OGb",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            plot_latents,
            plot_generated_and_inputs,
            plot_manual_generated_and_inputs,
        ),
    )

    return ade


def plot_inputs_and_images(input_folder: Path, output_folder: Path) -> None:
    images_and_inputs = []
    generated_files = sorted(input_folder.glob("manual_generated_*.npy"))
    for generated_file in generated_files:
        input_file = input_folder / generated_file.name.replace(
            "generated", "inputs"
        ).replace(".npy", ".json")
        with open(input_file) as json_file:
            input_data = json.load(json_file)
        image_data = np.load(str(generated_file)).squeeze()
        image_data = image_data.clip(0, 255).astype(np.uint8)
        images_and_inputs.append((input_data, image_data))

    num_images = len(images_and_inputs)
    plt.figure(figsize=(10, num_images * 5))

    for idx, (input_data, image_data) in enumerate(images_and_inputs):
        plt.subplot(num_images, 2, 2 * idx + 1)
        plt.title(f"Input for Image {idx + 1}")
        plt.text(0.5, 0.5, str(input_data), ha="center", va="center", wrap=True)
        plt.axis("off")

        plt.subplot(num_images, 2, 2 * idx + 2)
        plt.title(f"Generated Image {idx + 1}")
        plt.imshow(image_data, cmap="gray")
        plt.axis("off")

    output_folder.mkdir(exist_ok=True)
    plt.savefig(output_folder / "combined_plot.png")


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_array_gen_02_mnist_generation_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    image_base = "eir_tutorials/d_array_output/01_array_mnist_generation/data/mnist_npy"
    example_requests = [
        [
            {"mnist": f"{image_base}/10001.npy"},
            {"mnist": f"{image_base}/50496.npy"},
            {"mnist": f"{image_base}/25640.npy"},
        ],
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/d_array_output/01_array_mnist_generation",
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
            "image_shape": (28, 28),
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

    import numpy as np
    import requests

    def encode_array_to_base64(file_path: str) -> str:
        array_np = np.load(file_path)
        array_bytes = array_np.tobytes()
        return base64.b64encode(array_bytes).decode("utf-8")

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    image_base = "eir_tutorials/d_array_output/01_array_mnist_generation/data/mnist_npy"
    payload = [
        {"mnist": encode_array_to_base64(f"{image_base}/10001.npy")},
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def decode_and_save_images(
    predictions_file: str,
    output_folder: str,
    image_shape: tuple,
) -> None:
    os.makedirs(output_folder, exist_ok=True)

    with open(predictions_file) as file:
        predictions = json.load(file)[0]

    for i, prediction in enumerate(predictions["response"]["result"]):
        base64_array = prediction["mnist_output"]
        array_bytes = base64.b64decode(base64_array)

        array_np = np.frombuffer(array_bytes, dtype=np.float32).reshape(image_shape)
        array_np = array_np.clip(0, 255).astype(np.uint8)

        image = Image.fromarray(array_np, mode="L")
        image.save(Path(output_folder) / f"mnist_output_{i}.png")


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_array_gen_01_mnist_generation()
    exp_2 = get_array_gen_02_mnist_generation()
    exp_3 = get_array_gen_02_mnist_generation_serve()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
