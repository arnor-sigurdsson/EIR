import base64
import json
import os
import subprocess
import textwrap
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command
from eir.setup.config_setup_modules.config_setup_utils import load_yaml_config
from eir.utils.logging import get_logger

logger = get_logger(name=__name__)

CONTENT_ROOT = CR = "i_scaling"
TUTORIAL_NAME = TN = "03_scaling_diffusion"


def run_with_image_caption_server(command: list[str]) -> Path:
    server_path = "docs/doc_modules/i_scaling/image_caption_streamer.py"

    globals_file = next(
        (i for i in command if "globals" in i and i.endswith(".yaml")), None
    )
    if not globals_file:
        raise ValueError("Could not find globals.yaml file in command.")

    globals_dict = load_yaml_config(config_path=globals_file)
    run_folder = Path(globals_dict["basic_experiment"]["output_folder"])

    output_folder_injected = tuple(i for i in command if ".output_folder=" in i)
    if output_folder_injected:
        if len(output_folder_injected) != 1:
            raise ValueError(
                f"Expected exactly one .output_folder= override, "
                f"found {len(output_folder_injected)}"
            )
        output_folder_inject_string = output_folder_injected[0]
        run_folder = Path(output_folder_inject_string.rsplit(".output_folder=", 1)[-1])

    if run_folder.exists():
        logger.info(f"Run folder {run_folder} already exists. Skipping execution.")
        return run_folder

    server_args = ["python", server_path]

    logger.info(f"Starting image caption server: {' '.join(server_args)}")
    server_process = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    logger.info("Server process starting (PID: %d)...", server_process.pid)

    try:
        sleep_time = 60
        logger.info(f"Waiting {sleep_time} seconds for server to start...")
        time.sleep(sleep_time)

        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            logger.error("Server process failed to start.")
            logger.error(f"Server stdout:\n{stdout.decode(errors='ignore')}")
            logger.error(f"Server stderr:\n{stderr.decode(errors='ignore')}")
            raise RuntimeError(
                f"Server process at '{server_path}' terminated unexpectedly."
            )

        logger.info(f"Running main command: {' '.join(command)}")
        subprocess.run(args=command, check=True)
        logger.info("Main command finished successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Main command failed with exit code {e.returncode}")
        stdout, stderr = server_process.communicate()
        logger.error(f"Server stdout:\n{stdout.decode(errors='ignore')}")
        logger.error(f"Server stderr:\n{stderr.decode(errors='ignore')}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        raise
    finally:
        if server_process.poll() is None:
            logger.info("Terminating server process (PID: %d)...", server_process.pid)
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
                logger.info("Server process terminated.")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Server process did not terminate gracefully, killing (PID: %d)...",
                    server_process.pid,
                )
                server_process.kill()
                server_process.wait()
                logger.info("Server process killed.")
        else:
            logger.warning("Server process already terminated before finally block.")

    return run_folder


def train_image_caption_diffusion() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"
    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_image_cnn.yaml",
        f"{conf_output_path}/inputs_caption.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_image.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_IMAGE_CAPTION_DIFFUSION.pdf",
        ),
    ]

    for i in range(9):
        mapping.append(
            (
                f"samples/1000/auto/{i}_generated.png",
                f"figures/image_caption_diffusion/examples/auto_generated_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/1000/auto/{i}_inputs/image.png",
                f"figures/image_caption_diffusion/examples/auto_inputs_iter_1000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/50000/auto/{i}_generated.png",
                f"figures/image_caption_diffusion/examples/auto_generated_iter_50000_{i}.png",
            )
        )
        mapping.append(
            (
                f"samples/50000/auto/{i}_inputs/image.png",
                f"figures/image_caption_diffusion/examples/auto_inputs_iter_50000_{i}.png",
            )
        )

    data_output_path = None

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

    plot_generated_and_inputs = (
        plot_caption_guided_diffusion_grid,
        {
            "input_folder": Path(base_path) / "serve_results",
            "output_folder": Path(base_path) / "figures/image_caption_diffusion",
        },
    )

    ade = AutoDocExperimentInfo(
        name="IMAGE_CAPTION_DIFFUSION",
        data_url=None,
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            plot_generated_and_inputs,
        ),
        run_command_wrapper=run_with_image_caption_server,
    )

    return ade


def image_caption_diffusion_serve() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    serve_command = [
        "eirserve",
        "--device",
        "cuda",
        "--model-path",
        "FILL_MODEL",
    ]

    example_requests = [[]]
    for i, caption in enumerate(
        [
            "Gustav Klimt / Whispers of Gold / Symbolism / portrait / 1904",
            "Frida Kahlo / Wounded Deer / Surrealism / self-portrait / None",
            "Katsushika Hokusai / Mountain Village in Mist / Ukiyo-e / landscape "
            "/ 1830",
            "Salvador Dalí / Persistence of Memory Study / Surrealism / None / 1931",
            "Hieronymus Bosch / The Conjurer / Northern Renaissance / genre painting "
            "/ c.1502",
            "Andy Warhol / Campbell's Soup I / Pop Art / screen printing / 1968.0",
            "Alphonse Mucha / Job Cigarettes / Art Nouveau / poster / 1896",
            "Isaac Levitan / Above Eternal Peace / Realism / landscape / 1894",
            "Paul Klee / Senecio (Head of a Man) / Expressionism / portrait / 1922",
            "J.M.W. Turner / Seascape Study / Romanticism / marina / None",
        ]
    ):
        if i < 10:
            example_requests[0].append(
                {
                    "image": "eir_tutorials/f_image_output/03_mnist_diffusion/"
                    "data/data/images/00000.png",
                    "caption": caption,
                },
            )

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CR}/{TN}",
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
        name="IMAGE_CAPTION_DIFFUSION_SERVE",
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

    placeholder_image = (
        "eir_tutorials/f_image_output/03_mnist_diffusion/data/data/images/00000.png"
    )

    payload = [
        {
            "image": encode_image_to_base64(file_path=placeholder_image),
            "caption": "Rembrandt / The Scholar's Study / Dutch Golden Age "
            "/ interior / 1651",
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

    predictions["request"]
    for i, prediction in enumerate(predictions["response"]["result"]):
        base64_array = prediction["image"]
        array_bytes = base64.b64decode(base64_array)

        array_np = np.frombuffer(array_bytes, dtype=np.float32).reshape(tensor_shape)

        array_np = np.transpose(array_np, (1, 2, 0))
        array_np = np.clip(array_np, 0.0, 1.0)
        array_np = (array_np * 255).astype(np.uint8)

        image = Image.fromarray(array_np)
        image.save(Path(output_folder) / f"image_output_{i}.png")


def plot_caption_guided_diffusion_grid(input_folder: Path, output_folder: Path) -> None:
    plt.figure(figsize=(15, 25))

    num_images = min(10, len(list(input_folder.glob("image_output_*.png"))))

    for i in range(num_images):
        output_file = input_folder / f"image_output_{i}.png"
        caption_file = input_folder / f"caption_{i}.json"

        if not output_file.exists() or not caption_file.exists():
            continue

        output_image = plt.imread(output_file)

        with open(caption_file) as f:
            caption = json.load(f)

        caption = textwrap.fill(caption, width=40)

        plt.subplot(5, 2, i + 1)
        plt.imshow(output_image)
        plt.axis("off")
        plt.title(f"{caption}", fontsize=10)

    plt.tight_layout()
    output_folder.mkdir(exist_ok=True)
    plt.savefig(output_folder / "caption_guided_diffusion_grid.png")
    plt.close()


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = train_image_caption_diffusion()
    exp_2 = image_caption_diffusion_serve()

    return [
        exp_1,
        exp_2,
    ]
