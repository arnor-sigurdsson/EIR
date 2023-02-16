from pathlib import Path
from typing import Sequence

import pandas as pd

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


def get_07_multimodal_run_1_tabular_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/07_multimodal_tutorial/"

    conf_output_path = "eir_tutorials/07_multimodal_tutorial/conf"
    output_folder = "eir_tutorials/tutorial_runs/tutorial_07a_multimodal_tabular"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/07_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/07_input_tabular.yaml",
        "--fusion_configs",
        f"{conf_output_path}/07_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/07_output.yaml",
        f"--07_globals.output_folder={output_folder}",
        "--07_globals.get_acts=true",
    ]

    mapping = [
        (
            "training_curve_MCC",
            "figures/07_multimodal_training_curve_MCC_tabular.pdf",
        ),
        (
            "4000/activations/pets_tabular/feature_importance.pdf",
            "figures/tutorial_07a_feature_importance_D.pdf",
        ),
        (
            "4000/activations/pets_tabular/D: 100+ Days/cat_features_Breed1_"
            "D: 100+ Days.pdf",
            "figures/tutorial_07a_breed_importance_D.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/07_multimodal_tutorial/pet_adoption.zip")

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/07_multimodal_tutorial/",
                "-L",
                "2",
                "-I",
                "*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    tabular_preview = (
        _show_tabular_csv_example,
        {
            "input_path": Path("eir_tutorials/07_multimodal_tutorial/data/tabular.csv"),
            "output_path": Path(base_path) / "commands/tabular_preview.html",
        },
    )

    description_preview = (
        _show_text_description_example,
        {
            "input_path": Path(
                "eir_tutorials/07_multimodal_tutorial/data/descriptions.csv"
            ),
            "output_path": Path(base_path) / "commands/description_preview.txt",
        },
    )

    image_preview = (
        _show_image_example,
        {
            "input_path": Path("eir_tutorials/07_multimodal_tutorial/data/images"),
            "output_path": Path(base_path) / "commands/image_preview.jpg",
        },
    )

    ade = AutoDocExperimentInfo(
        name="MULTIMODAL_1_TABULAR",
        data_url="https://drive.google.com/file/d/1DVS-t1ne-TMam8-6gkCz2YzKNjEHEIGr",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(
            get_tutorial_folder,
            tabular_preview,
            description_preview,
            image_preview,
        ),
    )

    return ade


def _show_tabular_csv_example(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, nrows=1).T

    html = df.to_html(index=True, header=False)
    with open(output_path, "w") as f:
        f.write(html)


def _show_text_description_example(input_path: Path, output_path: Path) -> None:
    description_text = pd.read_csv(input_path, nrows=1)["Sequence"].values[0]

    with open(output_path, "w") as f:
        f.write(description_text)


def _show_image_example(input_path: Path, output_path: Path) -> None:
    example_file = next(input_path.iterdir())

    with open(example_file, "rb") as f:
        img_bytes = f.read()

    with open(output_path, "wb") as f:
        f.write(img_bytes)


def get_07_multimodal_run_2_tabular_description_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/07_multimodal_tutorial/"

    conf_output_path = "eir_tutorials/07_multimodal_tutorial/conf"
    output_folder = (
        "eir_tutorials/tutorial_runs/tutorial_07b_multimodal_tabular_description"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/07_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/07_input_tabular.yaml",
        f"{conf_output_path}/07_input_description.yaml",
        "--fusion_configs",
        f"{conf_output_path}/07_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/07_output.yaml",
        f"--07_globals.output_folder={output_folder}",
    ]

    mapping = [
        (
            "training_curve_MCC",
            "figures/07_multimodal_training_curve_MCC_tabular_description.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/07_multimodal_tutorial/pet_adoption.zip")

    ade = AutoDocExperimentInfo(
        name="MULTIMODAL_2_TABULAR_DESCRIPTION",
        data_url="https://drive.google.com/file/d/1DVS-t1ne-TMam8-6gkCz2YzKNjEHEIGr",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def get_07_multimodal_run_3_tabular_description_image_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/07_multimodal_tutorial/"

    conf_output_path = "eir_tutorials/07_multimodal_tutorial/conf"
    output_folder = (
        "eir_tutorials/tutorial_runs/tutorial_07c_multimodal_tabular_description_image"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/07_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/07_input_tabular.yaml",
        f"{conf_output_path}/07_input_description.yaml",
        f"{conf_output_path}/07_input_image.yaml",
        "--fusion_configs",
        f"{conf_output_path}/07_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/07_output.yaml",
        f"--07_globals.output_folder={output_folder}",
        "--07_globals.device='cuda:0'",
        "--07_globals.dataloader_workers=4",
    ]

    mapping = [
        (
            "training_curve_MCC",
            "figures/07_multimodal_training_curve_MCC_tabular_description_image.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/07_multimodal_tutorial/pet_adoption.zip")

    ade = AutoDocExperimentInfo(
        name="MULTIMODAL_3_TABULAR_DESCRIPTION_IMAGE",
        data_url="https://drive.google.com/file/d/1DVS-t1ne-TMam8-6gkCz2YzKNjEHEIGr",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def get_07_mm_apx_run_1_tab_desc_pre_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/07_multimodal_tutorial/"

    conf_output_path = "eir_tutorials/07_multimodal_tutorial/conf"
    output_folder = (
        "eir_tutorials/tutorial_runs/tutorial_07-apx-a_multimodal_tabular_"
        "description_pretrained"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/07_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/07_input_tabular.yaml",
        f"{conf_output_path}/07_input_description.yaml",
        f"{conf_output_path}/07_apx-a_input_description_pretrained.yaml",
        f"{conf_output_path}/07_input_image.yaml",
        "--fusion_configs",
        f"{conf_output_path}/07_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/07_output.yaml",
        f"--07_globals.output_folder={output_folder}",
        "--07_globals.device='cuda:0'",
        "--07_globals.dataloader_workers=4",
    ]

    mapping = [
        (
            "training_curve_MCC",
            "figures/"
            "07_multimodal_training_curve_MCC_tabular_description_pretrained.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/07_multimodal_tutorial/pet_adoption.zip")

    ade = AutoDocExperimentInfo(
        name="MULTIMODAL_APX-1_TABULAR_DESCRIPTION_IMAGE_PRETRAINED",
        data_url="https://drive.google.com/file/d/1DVS-t1ne-TMam8-6gkCz2YzKNjEHEIGr",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def get_07_mm_apx_run_2_tab_desc_mt_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/07_multimodal_tutorial/"

    conf_output_path = "eir_tutorials/07_multimodal_tutorial/conf"
    output_folder = (
        "eir_tutorials/tutorial_runs/tutorial_07-apx-b_multimodal_tabular_"
        "description_multi_task"
    )

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/07_globals.yaml",
        "--input_configs",
        f"{conf_output_path}/07_apx-b_mt_input_tabular.yaml",
        f"{conf_output_path}/07_input_description.yaml",
        f"{conf_output_path}/07_apx-a_input_description_pretrained.yaml",
        f"{conf_output_path}/07_input_image.yaml",
        "--fusion_configs",
        f"{conf_output_path}/07_fusion.yaml",
        "--output_configs",
        f"{conf_output_path}/07_apx-b_mt_output.yaml",
        f"--07_globals.output_folder={output_folder}",
        "--07_globals.device='cuda:0'",
        "--07_globals.dataloader_workers=4",
    ]

    mapping = [
        (
            "PERF-AVERAGE",
            "figures/"
            "07_multimodal_training_curve_"
            "perf-average_tabular_description_multi_task.pdf",
        ),
        (
            "training_curve_MCC",
            "figures/"
            "07_multimodal_training_curve_MCC_tabular_description_multi_task.pdf",
        ),
        (
            "Age/training_curve_R2.pdf",
            "figures/"
            "07_multimodal_training_curve_R2_tabular_description_multi_task_Age.pdf",
        ),
        (
            "Quantity/training_curve_R2.pdf",
            "figures/"
            "07_multimodal_training_curve_"
            "R2_tabular_description_multi_task_Quantity.pdf",
        ),
        (
            "Age/samples/2000/regression_predictions.pdf",
            "figures/regression_predictions_age.pdf",
        ),
        (
            "Quantity/samples/2000/regression_predictions.pdf",
            "figures/regression_predictions_quantity.pdf",
        ),
    ]

    data_output_path = Path("eir_tutorials/07_multimodal_tutorial/pet_adoption.zip")

    ade = AutoDocExperimentInfo(
        name="MULTIMODAL_APX-2_TABULAR_DESCRIPTION_IMAGE_PRETRAINED_MT",
        data_url="https://drive.google.com/file/d/1DVS-t1ne-TMam8-6gkCz2YzKNjEHEIGr",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_07_multimodal_run_1_tabular_info()
    exp_2 = get_07_multimodal_run_2_tabular_description_info()
    exp_3 = get_07_multimodal_run_3_tabular_description_image_info()
    exp_a1 = get_07_mm_apx_run_1_tab_desc_pre_info()
    exp_a2 = get_07_mm_apx_run_2_tab_desc_mt_info()

    return [exp_1, exp_2, exp_3, exp_a1, exp_a2]
