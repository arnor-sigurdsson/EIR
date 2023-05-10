from pathlib import Path
from typing import Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


def get_tutorial_08_run_cnn_1_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_1d_cnn.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_cnn-1d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_cnn_1.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_cnn_1.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    get_data_folder = (
        run_capture_and_save,
        {
            "command": ["tree", str(data_output_path.parent), "-L", "2", "--noreport"],
            "output_path": Path(base_path) / "commands/input_folder.txt",
        },
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/08_array_tutorial/",
                "-L",
                "3",
                "-I",
                "*01b*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="CNN_1",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_data_folder,
            get_tutorial_folder,
        ),
    )

    return ade


def get_tutorial_08_run_cnn_2_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_2d_cnn.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_cnn-2d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_cnn_2.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_cnn_2.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    ade = AutoDocExperimentInfo(
        name="CNN_2",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_tutorial_08_run_cnn_3_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_3d_cnn.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_cnn-3d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_cnn_3.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_cnn_3.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    ade = AutoDocExperimentInfo(
        name="CNN_3",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_tutorial_08_run_lcl_1_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_1d_lcl.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_lcl-1d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_lcl_1.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_lcl_1.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    ade = AutoDocExperimentInfo(
        name="LCL_1",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_tutorial_08_run_lcl_2_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_2d_lcl.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_lcl-2d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_lcl_2.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_lcl_2.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    ade = AutoDocExperimentInfo(
        name="LCL_2",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_tutorial_08_run_lcl_3_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/08_array_tutorial"

    conf_output_path = "eir_tutorials/08_array_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/input_3d_lcl.yaml",
        "--output_configs",
        f"{conf_output_path}/outputs.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs/tutorial_08_run_lcl-3d",
    ]

    mapping = [
        ("training_curve_ACC", "figures/tutorial_08_training_curve_ACC_lcl_3.pdf"),
        ("600/confusion_matrix", "figures/tutorial_08_confusion_matrix_lcl_3.pdf"),
    ]

    data_output_path = Path(
        "eir_tutorials/08_array_tutorial/data/processed_sample_data.zip"
    )

    make_comparison_plot = (
        _plot_performance,
        {
            "base_dir": Path("eir_tutorials/tutorial_runs/"),
            "output_path": Path(base_path, "figures", "val_comparison.png"),
        },
    )

    ade = AutoDocExperimentInfo(
        name="LCL_3",
        data_url="https://drive.google.com/file/d/1p-RfWqPiYGcmQI7LM60fXkIRSS5AFXM8",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(make_comparison_plot,),
    )

    return ade


def _plot_performance(base_dir: Path, output_path: Path) -> None:
    df_val = gather_validation_perf_average(base_dir=base_dir)
    fig = plot_validation_perf_average(data=df_val)

    fig.savefig(output_path, bbox_inches="tight", dpi=300)


def gather_validation_perf_average(base_dir: Path) -> pd.DataFrame:
    runs = sorted(list(base_dir.glob("tutorial_08_run_*")))
    data = []

    for run in runs:
        model_type = run.name.split("_")[3]
        model_dim = run.name.split("-")[-1]

        history_path = run / "validation_average_history.log"
        history_df = pd.read_csv(history_path)
        history_df["model_type"] = model_type.upper()
        history_df["model_dim"] = model_dim
        history_df["model"] = f"{model_type.upper()}"

        data.append(history_df)

    return pd.concat(data, ignore_index=True)


def plot_validation_perf_average(data: pd.DataFrame) -> plt.Figure:
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    custom_palette = sns.color_palette(["C0", "C1"])
    sns.set_palette(custom_palette)

    line_style_dict = {"1d": "-", "2d": "--", "3d": "-."}

    for model in data["model"].unique():
        model_type = model.split("-")[0]
        model_dim = model.split("-")[1]
        color = 0 if model_type == "CNN" else 1
        line_style = line_style_dict[model_dim.lower()]

        sns.lineplot(
            data=data[data["model"] == model],
            x="iteration",
            y="perf-average",
            label=model,
            color=custom_palette[color],
            linestyle=line_style,
            linewidth=2,
        )

    plt.title("Validation Performance Average", fontsize=18)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Performance Average", fontsize=14)
    plt.legend(title="Model", title_fontsize=12, fontsize=10)
    plt.tight_layout()

    return plt.gcf()


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_tutorial_08_run_cnn_1_info()
    exp_2 = get_tutorial_08_run_cnn_2_info()
    exp_3 = get_tutorial_08_run_cnn_3_info()
    exp_4 = get_tutorial_08_run_lcl_1_info()
    exp_5 = get_tutorial_08_run_lcl_2_info()
    exp_6 = get_tutorial_08_run_lcl_3_info()

    return [exp_1, exp_2, exp_3, exp_4, exp_5, exp_6]
