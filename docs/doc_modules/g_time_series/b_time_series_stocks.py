import base64
import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from docs.doc_modules.experiments import (
    AutoDocExperimentInfo,
    get_data,
    run_capture_and_save,
)
from docs.doc_modules.g_time_series.utils import get_content_root
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import add_model_path_to_command
from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    load_deeplake_dataset,
)

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "02_time_series_stocks"


def get_time_series_stocks_01_transformer(
    target_iteration: int,
) -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"
    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--input_configs",
        f"{conf_output_path}/input_sequence.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_transformer_stocks.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/time_series_stocks.zip")

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

    run_output_folder = f"eir_tutorials/tutorial_runs/{CR}/{TN}_transformer"

    output_sequences_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_output_sequences.csv"
    )

    plot_generated_time_series = (
        plot_time_series_predictions,
        {
            "run_folder": Path(run_output_folder),
            "meta_file": Path(run_output_folder)
            / f"results/stock_output/stock_output/samples/{target_iteration}/meta.json",
            "output_file": Path(output_sequences_csv),
            "output_dir": Path(base_path) / "figures/02_time_series_stocks",
            "data_format": "sequence",
        },
    )

    ade = AutoDocExperimentInfo(
        name="TIME_SERIES_STOCKS_01",
        data_url="https://drive.google.com/file/d/1aIbYbd33yystchj5eZfCQHE3-NhMubu4",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            plot_generated_time_series,
        ),
    )

    return ade


def get_time_series_stocks_02_one_shot(
    target_iteration: int,
) -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"
    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals_one_shot.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion_array.yaml",
        "--input_configs",
        f"{conf_output_path}/input_array_prior.yaml",
        "--output_configs",
        f"{conf_output_path}/output_array.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_one_shot_stocks.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/time_series_stocks.zip")

    run_output_folder = f"eir_tutorials/tutorial_runs/{CR}/{TN}_one_shot"

    output_sequences_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_output_sequences.csv"
    )

    plot_generated_time_series = (
        plot_time_series_predictions,
        {
            "run_folder": Path(run_output_folder),
            "meta_file": Path(run_output_folder)
            / f"results/stock_output/stock_output/samples/{target_iteration}/meta.json",
            "output_file": Path(output_sequences_csv),
            "output_dir": Path(base_path) / "figures/02_time_series_stocks_one_shot",
            "data_format": "array",
        },
    )

    ade = AutoDocExperimentInfo(
        name="TIME_SERIES_STOCKS_02",
        data_url="https://drive.google.com/file/d/1aIbYbd33yystchj5eZfCQHE3-NhMubu4",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(plot_generated_time_series,),
    )

    return ade


def get_time_series_stocks_03_diffusion(
    target_iteration: int,
) -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"
    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals_diffusion.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion_array.yaml",
        "--input_configs",
        f"{conf_output_path}/input_array_prior.yaml",
        f"{conf_output_path}/input_array_diffusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_array_diffusion.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            "figures/training_curve_LOSS_diffusion_stocks.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/time_series_stocks.zip")

    run_output_folder = f"eir_tutorials/tutorial_runs/{CR}/{TN}_diffusion"

    output_sequences_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_output_sequences.csv"
    )

    plot_generated_time_series = (
        plot_time_series_predictions,
        {
            "run_folder": Path(run_output_folder),
            "meta_file": Path(run_output_folder)
            / f"results/stock_output/stock_output/samples/{target_iteration}/meta.json",
            "output_file": Path(output_sequences_csv),
            "output_dir": Path(base_path) / "figures/02_time_series_stocks_diffusion",
            "data_format": "array",
        },
    )

    ade = AutoDocExperimentInfo(
        name="TIME_SERIES_STOCKS_03",
        data_url="https://drive.google.com/file/d/1aIbYbd33yystchj5eZfCQHE3-NhMubu4",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(plot_generated_time_series,),
    )

    return ade


def plot_time_series_predictions(
    run_folder: Path,
    meta_file: Path,
    output_file: Path,
    output_dir: Path,
    data_format: str,
) -> None:
    with open(meta_file) as f:
        meta_data = json.load(f)

    output_df = pd.read_csv(output_file, index_col="ID", dtype={"ID": str})

    for idx, (sample_id, sample_data) in enumerate(meta_data.items()):
        if data_format == "array":
            name = "stock_input.npy"
        elif data_format == "sequence":
            name = "stock_input.txt"
        else:
            raise ValueError(f"Invalid data format: {data_format}")

        input_path = Path(run_folder, sample_data["inputs"], name)
        generated_path = Path(run_folder, sample_data["generated"])

        if data_format == "sequence":
            with open(input_path) as f:
                input_sequence = [int(x) for x in f.read().strip().split()]

            with open(generated_path) as f:
                generated_sequence = [int(x) for x in f.read().strip().split()]
        elif data_format == "array":
            input_sequence = np.load(input_path).squeeze()
            generated_sequence = np.load(generated_path).squeeze()

        correct_sequence = [
            int(x) for x in output_df.loc[sample_id, "Sequence"].split()
        ]

        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        plt.plot(range(len(input_sequence)), input_sequence, label="Input", marker="o")

        offset = len(input_sequence)
        plt.plot(
            range(offset, offset + len(generated_sequence)),
            generated_sequence,
            label="Generated",
            marker="s",
        )
        plt.plot(
            range(offset, offset + len(correct_sequence)),
            correct_sequence,
            label="Correct",
            marker="^",
            alpha=0.5,
        )

        plt.title(f"Stock Price Prediction - Sample ID: {sample_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

        output_path = output_dir / f"sample_{idx}_plot.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()


def get_time_series_stocks_transformer_serve_experiment() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    test_input_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_test_input_sequences.csv"
    )
    test_output_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_test_output_sequences.csv"
    )

    ids, example_requests = generate_example_requests(input_file=Path(test_input_csv))

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CR}/{TN}_transformer",
    )

    plot_generated_time_series_with_uncert_sequence = (
        plot_time_series_with_uncertainty,
        {
            "request_file": Path(
                f"docs/tutorials/tutorial_files/{CR}/{TN}/serve_results/"
                f"predictions.json"
            ),
            "output_csv": Path(test_output_csv),
            "output_folder": Path(base_path) / "figures/02_time_series_stocks_test",
            "output_ids": ids,
            "data_format": "sequence",
        },
    )

    ade = AutoDocServingInfo(
        name="SEQUENCE_TO_SEQUENCE_STOCKS_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(plot_generated_time_series_with_uncert_sequence,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[],
    )

    return ade


def get_time_series_stocks_serve_experiment_one_shot() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    test_input_deeplake = f"eir_tutorials/{CR}/02_time_series_stocks/data/deeplake_test"
    test_output_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_test_output_sequences.csv"
    )

    repeat = 5
    num_sequences = 10

    ids, example_requests = generate_example_requests_arrays(
        input_dataset=Path(test_input_deeplake),
        num_sequences=num_sequences,
        repeat=repeat,
    )

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CR}/{TN}_one_shot",
    )

    plot_generated_time_series_with_uncert_sequence = (
        plot_time_series_with_uncertainty,
        {
            "request_file": Path(
                f"docs/tutorials/tutorial_files/{CR}/{TN}/serve_results/"
                f"predictions.json"
            ),
            "output_csv": Path(test_output_csv),
            "output_folder": Path(base_path)
            / "figures/02_time_series_stocks_test_one_shot",
            "output_ids": ids,
            "data_format": "array",
        },
    )

    ade = AutoDocServingInfo(
        name="SEQUENCE_TO_SEQUENCE_STOCKS_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(plot_generated_time_series_with_uncert_sequence,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[],
    )

    return ade


def get_time_series_stocks_serve_experiment_diffusion() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    test_input_deeplake = f"eir_tutorials/{CR}/02_time_series_stocks/data/deeplake_test"
    test_output_csv = (
        f"eir_tutorials/{CR}/02_time_series_stocks/data/stock_test_output_sequences.csv"
    )

    repeat = 5
    num_sequences = 10

    ids, example_requests = generate_example_requests_arrays(
        input_dataset=Path(test_input_deeplake),
        num_sequences=num_sequences,
        repeat=repeat,
    )

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CR}/{TN}_diffusion",
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    plot_generated_time_series_with_uncert_sequence = (
        plot_time_series_with_uncertainty,
        {
            "request_file": Path(
                f"docs/tutorials/tutorial_files/{CR}/{TN}/serve_results/"
                f"predictions.json"
            ),
            "output_csv": Path(test_output_csv),
            "output_folder": Path(base_path)
            / "figures/02_time_series_stocks_test_diffusion",
            "output_ids": ids,
            "data_format": "array",
        },
    )

    ade = AutoDocServingInfo(
        name="DIFFUSION_STOCKS_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(plot_generated_time_series_with_uncert_sequence,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
        ],
    )

    return ade


def plot_time_series_with_uncertainty(
    request_file: Path,
    output_csv: Path,
    output_folder: Path,
    output_ids: list[str],
    data_format: str,
    repeat: int = 5,
) -> None:
    with open(request_file) as f:
        data = json.load(f)

    output_df = pd.read_csv(output_csv, index_col="ID")

    output_folder.mkdir(parents=True, exist_ok=True)

    num_unique_sequences = len(data[0]["request"]) // repeat
    for i in range(num_unique_sequences):
        if data_format == "sequence":
            input_sequence = [
                int(x) for x in data[0]["request"][i * repeat]["stock_input"].split()
            ]
        elif data_format == "array":
            cur_b64 = data[0]["request"][i * repeat]["stock_input"]
            input_array = decode_base64_to_array(base64_str=cur_b64, dtype=np.float32)
            input_sequence = input_array.squeeze().tolist()
        else:
            raise ValueError(f"Invalid data format: {data_format}")

        generated_sequences = []
        for j in range(repeat):
            if data_format == "sequence":
                generated_sequence = [
                    int(x)
                    for x in data[0]["response"]["result"][i * repeat + j][
                        "stock_output"
                    ].split()
                ]
                generated_sequences.append(generated_sequence)
            elif data_format == "array":
                cur_b64 = data[0]["response"]["result"][i * repeat + j]["stock_output"]
                generated_array = decode_base64_to_array(
                    base64_str=cur_b64,
                    dtype=np.float32,
                )
                generated_sequences.append(generated_array.squeeze().tolist())

        generated_mean, generated_std = calculate_stats_variable_length(
            sequences=generated_sequences
        )

        cur_id = output_ids[i]
        correct_sequence = [int(x) for x in output_df.loc[cur_id]["Sequence"].split()]

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(input_sequence)), input_sequence, label="Input", marker="o")

        offset = len(input_sequence)
        x_generated = range(offset, offset + len(generated_mean))
        plt.plot(
            x_generated,
            generated_mean,
            label="Generated (mean)",
            marker="s",
        )
        plt.fill_between(
            x_generated,
            generated_mean - generated_std,
            generated_mean + generated_std,
            alpha=0.3,
            color="orange",
            label="Generated (±1 std)",
        )

        x_correct = range(offset, offset + len(correct_sequence))
        plt.plot(
            x_correct, correct_sequence, label="Correct", marker="^", color="green"
        )

        plt.title(f"Time Series Prediction - Sample {i + 1}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

        output_path = output_folder / f"sample_{i + 1}_plot_with_uncertainty.pdf"
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()


def calculate_stats_variable_length(
    sequences: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    min_length = min(len(seq) for seq in sequences)

    truncated_sequences = [seq[:min_length] for seq in sequences]

    array_sequences = np.array(truncated_sequences)

    mean_sequence = np.mean(array_sequences, axis=0)
    std_sequence = np.std(array_sequences, axis=0)

    return mean_sequence, std_sequence


def generate_example_requests(
    input_file: Path,
    num_sequences: int = 10,
    repeat: int = 5,
) -> tuple[list[str], list[list[dict[str, str]]]]:
    df = pd.read_csv(input_file, index_col="ID")

    example_requests = []

    random_rows = df.sample(n=num_sequences)

    for i in range(num_sequences):
        sequence = random_rows.iloc[i]["Sequence"]
        for _ in range(repeat):
            example_requests.append(
                {
                    "stock_input": sequence,
                    "stock_output": "",
                }
            )

    ids = random_rows.index.tolist()
    return ids, [example_requests]


def generate_example_requests_arrays(
    input_dataset: Path,
    num_sequences: int = 10,
    repeat: int = 5,
) -> tuple[list[str], list[list[dict[str, str]]]]:
    assert is_deeplake_dataset(data_source=str(input_dataset))

    deeplake_ds = load_deeplake_dataset(data_source=str(input_dataset))

    n_samples = len(deeplake_ds)
    random_rows = np.random.choice(n_samples, size=num_sequences, replace=False)

    example_requests = []
    ids = []

    output_array_placeholder = np.zeros((64,), dtype=np.float32)
    cur_placeholder_base64 = encode_array_to_base64(array_np=output_array_placeholder)

    for idx in random_rows:
        sample = deeplake_ds[int(idx)]

        cur_id = sample["ID"]
        cur_arr = sample["input_array"].astype(np.float32)
        cur_arr_base64 = encode_array_to_base64(array_np=cur_arr)

        for _cur_repeat in range(repeat):
            example_requests.append(
                {
                    "stock_input": cur_arr_base64,
                    "stock_output": cur_placeholder_base64,
                }
            )

        ids.append(cur_id)

    return ids, [example_requests]


def encode_array_to_base64(array_np: np.ndarray) -> str:
    array_bytes = array_np.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def decode_base64_to_array(
    base64_str: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    array_bytes = base64.b64decode(base64_str)
    return np.frombuffer(array_bytes, dtype=dtype)


def example_request_function_python():
    import base64

    import numpy as np
    import requests

    input_array = np.array(
        [
            31,
            32,
            31,
            30,
            31,
            31,
            31,
            31,
            31,
            30,
            31,
            31,
            32,
            33,
            32,
            32,
            33,
            33,
            33,
            34,
            35,
            34,
            34,
            33,
            32,
            32,
            32,
            33,
            33,
            34,
            34,
            34,
            33,
            34,
            34,
            34,
            34,
            36,
            36,
            36,
            36,
            37,
            35,
            35,
            34,
            35,
            35,
            34,
            35,
            35,
            36,
            35,
            36,
            35,
            34,
            34,
            34,
            33,
            33,
            31,
            31,
            31,
            32,
            32,
        ],
        dtype=np.float32,
    )

    output_base = np.zeros(shape=input_array.shape, dtype=np.float32)

    def encode_array_to_base64(array_np: np.ndarray) -> str:
        array_bytes = array_np.tobytes()
        return base64.b64encode(array_bytes).decode("utf-8")

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    payload = [
        {
            "stock_input": encode_array_to_base64(array_np=input_array),
            "stock_output": encode_array_to_base64(array_np=output_base),
        },
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_time_series_stocks_01_transformer(target_iteration=20000)
    get_data(url=exp_1.data_url, output_path=exp_1.data_output_path)

    exp_2 = get_time_series_stocks_transformer_serve_experiment()
    exp_3 = get_time_series_stocks_02_one_shot(target_iteration=20000)
    exp_4 = get_time_series_stocks_serve_experiment_one_shot()
    exp_5 = get_time_series_stocks_03_diffusion(target_iteration=14000)
    exp_6 = get_time_series_stocks_serve_experiment_diffusion()

    return [
        exp_1,
        exp_2,
        exp_3,
        exp_4,
        exp_5,
        exp_6,
    ]
