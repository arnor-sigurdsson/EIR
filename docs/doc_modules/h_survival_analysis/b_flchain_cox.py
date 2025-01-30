import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from docs.doc_modules.experiments import (
    AutoDocExperimentInfo,
    get_data,
    run_capture_and_save,
)
from docs.doc_modules.h_survival_analysis.utils import get_content_root
from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import (
    AutoDocServingInfo,
    build_request_example_module_from_function,
)
from docs.doc_modules.utils import get_saved_model_path

CONTENT_ROOT = CR = get_content_root()
TUTORIAL_NAME = TN = "02_flchain_cox"


def get_flchain_run_1_tabular_info() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_C-INDEX",
            "figures/flchain_training_curve_C-INDEX_tabular_1.pdf",
        ),
        (
            "2400/cox_survival_curves.pdf",
            "figures/survival_curves.pdf",
        ),
        (
            "2400/cox_risk_stratification.pdf",
            "figures/cox_risk_stratification.pdf",
        ),
        (
            "2400/cox_individual_curves.pdf",
            "figures/individual_survival_curves.pdf",
        ),
        (
            "2400/attributions/flchain/feature_importance.pdf",
            "figures/feature_importance.pdf",
        ),
        (
            "2400/attributions/flchain/event/continuous_attributions_event.pdf",
            "figures/continuous_attributions.pdf",
        ),
        (
            "2400/attributions/flchain/event/categorical_attributions_sex_event.pdf",
            "figures/sex_attributions.pdf",
        ),
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/flchain_data.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                f"eir_tutorials/{CR}/{TN}/",
                "-I",
                "*test*.yaml|*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="FLCHAIN_1",
        data_url="https://drive.google.com/file/d/17iojKFXUBf-xgltogdNTbKa2qtyNP3ar",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_flchain_run_1_predict_info() -> AutoDocExperimentInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    conf_output_path = f"eir_tutorials/{CR}/{TN}/conf"

    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}/"

    command = [
        "eirpredict",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_test.yaml",
        "--fusion_configs",
        f"{conf_output_path}/fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/output_test.yaml",
        "--model_path",
        "FILL_MODEL",
        "--evaluate",
        "--output_folder",
        run_1_output_path,
    ]

    data_output_path = Path(f"eir_tutorials/{CR}/{TN}/flchain_data.zip")

    mapping = [
        (
            "calculated_metrics",
            "tutorial_data/calculated_metrics_test.json",
        ),
    ]

    ade = AutoDocExperimentInfo(
        name="FLCHAIN_1_PREDICT",
        data_url="https://drive.google.com/file/d/17iojKFXUBf-xgltogdNTbKa2qtyNP3ar",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        pre_run_command_modifications=(_add_model_path_to_command,),
        files_to_copy_mapping=mapping,
        post_run_functions=(),
        force_run_command=True,
    )

    return ade


def get_flchain_run_1_serve_info() -> AutoDocServingInfo:
    base_path = f"docs/tutorials/tutorial_files/{CR}/{TN}"

    model_path_placeholder = "FILL_MODEL"

    server_command = ["eirserve", "--model-path", model_path_placeholder]

    example_requests = generate_survival_analysis_requests(
        test_data_path=Path(f"eir_tutorials/{CR}/{TN}/data/flchain_test.csv")
    )

    example_request_module_python = build_request_example_module_from_function(
        function=example_request_function_python,
        name="python",
        language="python",
    )

    survival_plots = (
        create_all_survival_plots,
        {
            "base_path": Path(base_path),
        },
    )

    ade = AutoDocServingInfo(
        name="FLCHAIN_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(_add_model_path_to_command,),
        post_run_functions=(survival_plots,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
        request_example_modules=[
            example_request_module_python,
        ],
    )

    return ade


def create_all_survival_plots(base_path: Path):
    predictions_file = Path(base_path) / "serve_results/predictions.json"
    output_folder = Path(base_path) / "figures/survival_analysis"

    plot_configs = [
        {
            "stratify_by": "sex",
            "title": "Survival Probability over Time by Sex",
        },
        {
            "stratify_by": "mgus",
            "title": "Survival Probability over Time by MGUS Status",
        },
        {
            "stratify_by": "flcgrp",
            "title": "Survival Probability over Time by FLC Group",
        },
        {
            "stratify_by": "age",
            "title": "Survival Probability over Time by Age Group",
        },
    ]

    for config in plot_configs:
        plot_survival_analysis_results(
            request_file=predictions_file, output_folder=output_folder, **config
        )


def generate_survival_analysis_requests(
    test_data_path: Path,
    n_samples: int = 50,
) -> list[list[dict]]:
    df = pd.read_csv(filepath_or_buffer=test_data_path)
    df = df.dropna(subset=["creatinine"])

    sampled_df = df
    requests = []

    for _, row in sampled_df.iterrows():
        request = {
            "age": row["age"],
            "sex": row["sex"],
            "flcgrp": str(row["flcgrp"]),
            "kappa": float(row["kappa"]),
            "lambdaport": float(row["lambdaport"]),
            "creatinine": float(row["creatinine"]),
            "mgus": row["mgus"],
        }
        requests.append({"flchain": request})

    return [requests]


def example_request_function_python():
    import requests

    def send_request(url: str, payload: list[dict]):
        response = requests.post(url, json=payload)
        return response.json()

    payload = [
        {
            "flchain": {
                "age": 65,
                "sex": "M",
                "flcgrp": "1",
                "kappa": 1.5,
                "lambdaport": 1.2,
                "creatinine": 1.1,
                "mgus": "yes",
            }
        }
    ]

    response = send_request(url="http://localhost:8000/predict", payload=payload)
    print(response)

    # --skip-after
    return response


def plot_survival_analysis_results(
    request_file: Path,
    output_folder: Path,
    stratify_by: str = "sex",
    color_dict: dict[str, str] | None = None,
    label_dict: dict[str, str] | None = None,
    title: str | None = None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.edgecolor": "gray",
            "legend.framealpha": 0.9,
        }
    )

    def get_age_group(age: float) -> str:
        if age < 55:
            return "1"
        if age < 65:
            return "2"
        if age < 75:
            return "3"
        return "4"

    default_color_schemes = {
        "sex": {
            "M": "#2E5EAA",
            "F": "#AA2E87",
        },
        "mgus": {
            "yes": "#CC3311",
            "no": "#009988",
        },
        "age": {
            "1": "#332288",
            "2": "#88CCEE",
            "3": "#44AA99",
            "4": "#117733",
        },
        "flcgrp": {
            "1": "#332288",
            "2": "#88CCEE",
            "3": "#44AA99",
            "4": "#117733",
            "5": "#999933",
            "6": "#DDCC77",
            "7": "#CC6677",
            "8": "#882255",
            "9": "#AA4499",
            "10": "#EE7733",
        },
    }

    default_label_schemes = {
        "sex": {
            "M": "Male",
            "F": "Female",
        },
        "mgus": {
            "yes": "MGUS Present",
            "no": "No MGUS",
        },
        "age": {
            "1": "<55 years",
            "2": "55-64 years",
            "3": "65-74 years",
            "4": "≥75 years",
        },
        "flcgrp": {str(i): f"Group {i}" for i in range(1, 11)},
    }

    colors = color_dict or default_color_schemes.get(stratify_by, {})
    labels = label_dict or default_label_schemes.get(stratify_by, {})

    with open(request_file) as f:
        data = json.load(f)

    times = None
    stratified_data = {}

    for request, response in zip(
        data[0]["request"], data[0]["response"]["result"], strict=False
    ):
        strat_value = (
            get_age_group(request["flchain"]["age"])
            if stratify_by == "age"
            else request["flchain"][stratify_by]
        )
        survival_prob = response["flchain_prediction"]["event"]["survival_probs"]

        if times is None:
            times = response["flchain_prediction"]["event"]["time_points"]

        if strat_value not in stratified_data:
            stratified_data[strat_value] = []

        stratified_data[strat_value].append(survival_prob)

    fig, ax = plt.subplots(figsize=(10, 7))

    def plot_survival_curves(curves: np.ndarray, color: str, label: str):
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        n_samples = curves.shape[0]
        stderr = std_curve / np.sqrt(n_samples)
        ci_lower = mean_curve - 1.96 * stderr
        ci_upper = mean_curve + 1.96 * stderr

        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)

        times_extended = np.append(times, times[-1])
        ci_lower_extended = np.append(ci_lower, ci_lower[-1])
        ci_upper_extended = np.append(ci_upper, ci_upper[-1])

        ax.fill_between(
            times_extended,
            ci_lower_extended,
            ci_upper_extended,
            color=color,
            alpha=0.15,
            step="post",
        )

        ax.plot(times, mean_curve, color=color, label=label, lw=2, zorder=10)

    for strat_value, curves in stratified_data.items():
        color = colors.get(strat_value, f"C{len(colors)}")
        label = labels.get(strat_value, str(strat_value))
        plot_survival_curves(
            curves=np.array(curves).squeeze(), color=color, label=label
        )

    if title is None:
        title = f"Survival Probability over Time by {stratify_by.upper()}"
    ax.set_title(title, pad=20, fontweight="bold")
    ax.set_xlabel("Time (days)", labelpad=10)
    ax.set_ylabel("Survival Probability", labelpad=10)

    ax.set_ylim(0, 1.02)
    ax.set_xlim(times[0], times[-1] + 100)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.1f}"))

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        fancybox=True,
        framealpha=0.9,
        edgecolor="gray",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, which="major", linestyle="-", alpha=0.2)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)

    plt.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_folder / f"survival_curve_by_{stratify_by}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def _get_model_path_for_predict() -> str:
    run_1_output_path = f"eir_tutorials/tutorial_runs/{CR}/{TN}"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: list[str]) -> list[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo | AutoDocServingInfo]:
    exp_1 = get_flchain_run_1_tabular_info()
    get_data(url=exp_1.data_url, output_path=exp_1.data_output_path)

    exp_2 = get_flchain_run_1_predict_info()
    exp_3 = get_flchain_run_1_serve_info()

    return [
        exp_1,
        exp_2,
        exp_3,
    ]
