from typing import Dict, Sequence, List

import pandas as pd
import pytest
import timm
import torch
from timm.models.registry import get_model_default_value

from eir.models.model_training_utils import trace_eir_model
from tests.test_models.model_testing_utils import prepare_example_batch


def get_timm_models_to_test() -> List[str]:
    """
    LeVIT and ConVIT models fail tracing currently, therefore skipped.
    """

    url = (
        "https://raw.githubusercontent.com/rwightman/pytorch-image-models/"
        "master/results/results-imagenet.csv"
    )

    df = pd.read_csv(filepath_or_buffer=url)
    df = df[df["param_count"] <= 12]

    models = list(df["model"])
    models_pt = [i for i in models if not i.startswith("tf")]
    models_filtered = [i for i in models_pt if i in timm.list_models()]

    models_manual_filtered = []
    not_allowed = {"levit", "convit"}
    for model_name in models_filtered:
        if any(i in model_name for i in not_allowed):
            continue
        else:
            models_manual_filtered.append(model_name)

    return models_manual_filtered


def get_test_image_models_parametrization() -> Sequence[Dict]:

    models = ["ResNet"]
    timm_models = get_timm_models_to_test()
    all_models = models + timm_models

    all_parametrizations = []

    for model_type in all_models:

        if model_type == "ResNet":
            model_init_config = {
                "layers": [1, 1, 1, 1],
                "block": "BasicBlock",
            }
        else:
            model_init_config = {}

        size = get_model_default_value(model_name=model_type, cfg_key="input_size")
        if not size:
            size = 224
        else:
            size = size[-1]

        cur_params = {
            "injections": {
                "input_configs": [
                    {
                        "input_info": {
                            "input_name": "test_image",
                        },
                        "input_type_info": {
                            "auto_augment": False,
                            "size": (size,),
                        },
                        "model_config": {
                            "model_type": model_type,
                            "pretrained_model": False,
                            "model_init_config": model_init_config,
                        },
                    }
                ]
            }
        }
        all_parametrizations.append(cur_params)

    return all_parametrizations


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "multi", "modalities": ("image",)},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    get_test_image_models_parametrization(),
    indirect=True,
)
def test_image_models(
    parse_test_cl_args,
    create_test_data,
    create_test_config,
    create_test_model,
    create_test_labels,
):
    model = create_test_model

    example_batch = prepare_example_batch(
        configs=create_test_config, labels=create_test_labels, model=model
    )

    model.eval()
    with torch.no_grad():
        _ = trace_eir_model(fusion_model=model, example_inputs=example_batch.inputs)
