import json
import textwrap
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

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

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "01_time_series_power"


def get_time_series_power_01_transformer(
    data: Literal["sim", "real"],
    target_iteration: int,
) -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals_{data}.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--input_configs",
        f"{conf_output_path}/input_sequence_{data}.yaml",
        "--output_configs",
        f"{conf_output_path}/output_{data}.yaml",
    ]

    mapping = [
        (
            "training_curve_LOSS",
            f"figures/training_curve_LOSS_transformer_1_{data}.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/time_series_power.zip")

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

    run_output_folder = f"eir_tutorials/tutorial_runs/{CR}/{TN}_transformer_{data}"

    output_sequences_csv = (
        f"eir_tutorials/{CR}/01_time_series_power/data/{data}_output_sequences.csv"
    )

    plot_generated_time_series = (
        plot_time_series_predictions,
        {
            "run_folder": Path(run_output_folder),
            "meta_file": Path(run_output_folder)
            / f"results/power_output/power_output/samples/{target_iteration}/meta.json",
            "output_file": Path(output_sequences_csv),
            "output_dir": Path(base_path) / f"figures/01_time_series_power_{data}",
        },
    )

    ade = AutoDocExperimentInfo(
        name=f"TIME_SERIES_POWER_01_{data.upper()}",
        data_url="https://drive.google.com/file/d/11wrOpXdS5RODWqA3Of8mjBsUU3iVebxR",
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


def plot_time_series_predictions(
    run_folder: Path,
    meta_file: Path,
    output_file: Path,
    output_dir: Path,
) -> None:
    with open(meta_file) as f:
        meta_data = json.load(f)

    output_df = pd.read_csv(output_file, index_col="ID", dtype={"ID": str})

    for idx, (sample_id, sample_data) in enumerate(meta_data.items()):
        input_path = Path(run_folder, sample_data["inputs"], "power_input.txt")
        generated_path = Path(run_folder, sample_data["generated"])

        with open(input_path) as f:
            input_sequence = [int(x) for x in f.read().strip().split()]

        with open(generated_path) as f:
            generated_sequence = [int(x) for x in f.read().strip().split()]

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

        plt.title(f"Time Series Prediction - Sample ID: {sample_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

        output_path = output_dir / f"sample_{idx}_plot.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()


def get_time_series_power_serve_experiment() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    test_input_csv = (
        f"eir_tutorials/{CR}/01_time_series_power/data/real_test_input_sequences.csv"
    )
    test_output_csv = (
        f"eir_tutorials/{CR}/01_time_series_power/data/real_test_output_sequences.csv"
    )

    ids, example_requests = generate_example_requests(input_file=Path(test_input_csv))

    add_model_path = partial(
        add_model_path_to_command,
        run_path=f"eir_tutorials/tutorial_runs/{CR}/{TN}_transformer_real",
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    bash_args = _get_example_request_bash_args()
    example_request_module_bash = build_request_example_module_from_function(
        **bash_args
    )

    plot_generated_time_series_with_uncert = (
        plot_time_series_with_uncertainty,
        {
            "request_file": Path(
                f"docs/tutorials/tutorial_files/{CR}/{TN}/"
                "serve_results/predictions.json"
            ),
            "output_csv": Path(test_output_csv),
            "output_folder": Path(base_path) / "figures/01_time_series_power_test",
            "output_ids": ids,
        },
    )

    ade = AutoDocServingInfo(
        name="SEQUENCE_TO_SEQUENCE_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(plot_generated_time_series_with_uncert,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
            example_request_module_bash,
        ],
    )

    return ade


def plot_time_series_with_uncertainty(
    request_file: Path,
    output_csv: Path,
    output_folder: Path,
    output_ids: list[str],
    repeat: int = 5,
) -> None:
    with open(request_file) as f:
        data = json.load(f)

    output_df = pd.read_csv(output_csv, index_col="ID")

    output_folder.mkdir(parents=True, exist_ok=True)

    num_unique_sequences = len(data[0]["request"]) // repeat
    for i in range(num_unique_sequences):
        input_sequence = [
            int(x) for x in data[0]["request"][i * repeat]["power_input"].split()
        ]

        generated_sequences = []
        for j in range(repeat):
            generated_sequence = [
                int(x)
                for x in data[0]["response"]["result"][i * repeat + j][
                    "power_output"
                ].split()
            ]
            generated_sequences.append(generated_sequence)

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
    num_sequences: int = 3,
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
                    "power_input": sequence,
                    "power_output": "",
                }
            )

    ids = random_rows.index.tolist()
    return ids, [example_requests]


def example_request_function_python():
    import requests

    def send_request(url: str, payload: list[dict]) -> list[dict]:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    payload = [
        {
            "power_input": "19 3 10 11 14 3 3 4 4 9 39 27 12 5 20 20 38 39 41 61 "
            "52 31 43 31",
            "power_output": "",
        },
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def _get_example_request_bash_args():
    command = textwrap.dedent(
        """\
        curl -X POST \\
        "http://localhost:8000/predict" \\
        -H "accept: application/json" \\
        -H "Content-Type: application/json" \\
        -d '[{
            "power_input": "19 3 10 11 14 3 3 4 4 9 39 27 12 5 20 20 38 39 41 \
61 52 31 43 31",
            "power_output": ""
        }]'
    """
    )

    def _function_to_run_example() -> dict:
        import json
        import subprocess

        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        result_as_dict = json.loads(result.stdout)
        return result_as_dict

    command_as_text = command
    return {
        "function": _function_to_run_example,
        "custom_body": command_as_text,
        "name": "bash",
        "language": "shell",
    }


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    """
    We call get data manually to ensure that the data is downloaded as the last
    experiments use some of it for the setup.
    """
    exp_1 = get_time_series_power_01_transformer(data="sim", target_iteration=2500)
    get_data(url=exp_1.data_url, output_path=exp_1.data_output_path)

    exp_2 = get_time_series_power_01_transformer(data="real", target_iteration=2000)
    exp_3 = get_time_series_power_serve_experiment()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
