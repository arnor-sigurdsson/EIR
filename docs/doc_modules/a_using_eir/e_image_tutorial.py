from functools import partial
from pathlib import Path
from typing import Sequence

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.serve_experiments_utils import copy_inputs, load_data_for_serve
from docs.doc_modules.serving_experiments import AutoDocServingInfo
from docs.doc_modules.utils import add_model_path_to_command


def get_05_hot_dog_run_1_resnet_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/05_image_tutorial/"

    conf_output_path = "eir_tutorials/a_using_eir/05_image_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/05_image_training_curve_ACC_resnet_1.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/05_image_tutorial/data/hot_dog.zip"
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_tutorials/a_using_eir/05_image_tutorial/",
                "-L",
                "3",
                "-I",
                "*.zip",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="IMAGE_1_RESNET",
        data_url="https://drive.google.com/file/d/1g5slDIwtXcksjKlJ5anAiVCZGCM9AAHI",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(get_tutorial_folder,),
    )

    return ade


def get_05_hot_dog_run_2_resnet_pretrained_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/05_image_tutorial/"

    conf_output_path = "eir_tutorials/a_using_eir/05_image_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_resnet18.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_05_is_it_a_hot_dog_pretrained_resnet",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/05_image_training_curve_ACC_resnet_pretrained_1.pdf",
        ),
        (
            ".*/600/.*_Not Hot Dog.pdf",
            "figures/pretrained_resnet_not_hot_dog_attributions.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/05_image_tutorial/data/hot_dog.zip"
    )

    ade = AutoDocExperimentInfo(
        name="IMAGE_2_PRETRAINED_RESNET",
        data_url="https://drive.google.com/file/d/1g5slDIwtXcksjKlJ5anAiVCZGCM9AAHI",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_05_hot_dog_run_2_combined_pretrained_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/05_image_tutorial/"

    conf_output_path = "eir_tutorials/a_using_eir/05_image_tutorial/conf"

    command = [
        "eirtrain",
        "--global_configs",
        f"{conf_output_path}/globals.yaml",
        "--input_configs",
        f"{conf_output_path}/inputs_efficientnet_b0.yaml",
        f"{conf_output_path}/inputs_resnet18.yaml",
        "--output_configs",
        f"{conf_output_path}/output.yaml",
        "--globals.output_folder=eir_tutorials/tutorial_runs"
        "/a_using_eir/tutorial_05_is_it_a_hot_dog_pretrained_combined",
    ]

    mapping = [
        (
            "training_curve_ACC",
            "figures/05_image_training_curve_ACC_combined_pretrained_1.pdf",
        ),
        (
            ".*/600/.*resnet18.*_Not Hot Dog.pdf",
            "figures/pretrained_combined_resnet_not_hot_dog_attributions.pdf",
        ),
        (
            ".*/600/.*efficientnet.*_Not Hot Dog.pdf",
            "figures/pretrained_combined_efficientnet_not_hot_dog_attributions.pdf",
        ),
    ]

    data_output_path = Path(
        "eir_tutorials/a_using_eir/05_image_tutorial/data/hot_dog.zip"
    )

    ade = AutoDocExperimentInfo(
        name="IMAGE_3_PRETRAINED_EFFICIENTNET",
        data_url="https://drive.google.com/file/d/1g5slDIwtXcksjKlJ5anAiVCZGCM9AAHI",
        data_output_path=data_output_path,
        conf_output_path=Path(conf_output_path),
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=mapping,
        post_run_functions=(),
    )

    return ade


def get_05_hot_dog_run_1_serve_info() -> AutoDocServingInfo:
    base_path = "docs/tutorials/tutorial_files/a_using_eir/05_image_tutorial"

    server_command = ["eirserve", "--model-path", "FILL_MODEL"]

    base = (
        "eir_tutorials/a_using_eir/05_image_tutorial/data/"
        "hot_dog_not_hot_dog/food_images"
    )
    example_requests = [
        {
            "hot_dog_efficientnet": f"{base}/1040579.jpg",
            "hot_dog_resnet18": f"{base}/1040579.jpg",
        },
        {
            "hot_dog_efficientnet": f"{base}/108743.jpg",
            "hot_dog_resnet18": f"{base}/108743.jpg",
        },
    ]

    add_model_path = partial(
        add_model_path_to_command,
        run_path="eir_tutorials/tutorial_runs/a_using_eir/"
        "tutorial_05_is_it_a_hot_dog_pretrained_combined",
    )

    copy_inputs_to_serve = (
        copy_inputs,
        {
            "example_requests": example_requests,
            "output_folder": str(Path(base_path) / "serve_results"),
        },
    )

    ade = AutoDocServingInfo(
        name="IMAGE_DEPLOY",
        base_path=Path(base_path),
        server_command=server_command,
        pre_run_command_modifications=(add_model_path,),
        post_run_functions=(copy_inputs_to_serve,),
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
    )

    return ade


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_05_hot_dog_run_1_resnet_info()
    exp_2 = get_05_hot_dog_run_2_resnet_pretrained_info()
    exp_3 = get_05_hot_dog_run_2_combined_pretrained_info()
    exp_4 = get_05_hot_dog_run_1_serve_info()

    return [exp_1, exp_2, exp_3, exp_4]
