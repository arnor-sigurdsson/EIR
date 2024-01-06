import random
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from aislib.misc_utils import get_logger
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from docs.doc_modules.serving_experiments import run_serve_experiment_from_command
from eir import train
from eir.serve_modules.serve_network_utils import _deserialize_array
from eir.setup.schemas import InputConfig, OutputConfig
from tests.test_modelling.test_modelling_utils import check_performance_result_wrapper
from tests.test_modelling.test_sequence_modelling.test_sequence_output_modelling import (  # noqa
    get_expected_keywords_set,
)

if TYPE_CHECKING:
    from tests.setup_tests.fixtures_create_configs import TestConfigInits
    from tests.setup_tests.fixtures_create_data import TestDataConfig
    from tests.setup_tests.fixtures_create_experiment import ModelTestConfig


logger = get_logger(name=__name__)


def get_parametrization():
    params = [get_base_parametrization(compiled=False)]

    return params


def get_base_parametrization(compiled: bool = False) -> dict:
    params = {
        "injections": {
            "global_configs": {
                "output_folder": "multi_task_multi_modal",
                "n_epochs": 10,
                "gradient_clipping": 1.0,
                "lr": 0.001,
                "compile_model": compiled,
                "checkpoint_interval": 200,
            },
            "input_configs": [
                {
                    "input_info": {"input_name": "test_genotype"},
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {"l1": 1e-04},
                    },
                },
                {
                    "input_info": {"input_name": "test_sequence"},
                },
                {
                    "input_info": {"input_name": "test_array"},
                    "input_type_info": {
                        "normalization": "element",
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "rb_do": 0.25,
                            "channel_exp_base": 3,
                            "l1": 1e-04,
                            "kernel_height": 1,
                            "attention_inclusion_cutoff": 256,
                        },
                    },
                },
                {
                    "input_info": {"input_name": "test_image"},
                    "model_config": {
                        "model_init_config": {
                            "layers": [2],
                            "kernel_width": 2,
                            "kernel_height": 2,
                            "down_stride_width": 2,
                            "down_stride_height": 2,
                        },
                    },
                },
                {
                    "input_info": {"input_name": "test_tabular"},
                    "input_type_info": {
                        "input_cat_columns": ["OriginExtraCol"],
                        "input_con_columns": ["ExtraTarget"],
                    },
                    "model_config": {
                        "model_type": "tabular",
                        "model_init_config": {"l1": 1e-04},
                    },
                },
            ],
            "fusion_configs": {
                "model_config": {
                    "fc_task_dim": 256,
                    "fc_do": 0.10,
                    "rb_do": 0.10,
                },
            },
            "output_configs": [
                {
                    "output_info": {"output_name": "test_output_copy"},
                    "output_type_info": {
                        "target_cat_columns": [],
                        "target_con_columns": ["Height"],
                    },
                },
                {
                    "output_info": {"output_name": "test_output_tabular"},
                    "output_type_info": {
                        "target_cat_columns": ["Origin"],
                        "target_con_columns": ["Height"],
                    },
                },
                {
                    "output_info": {"output_name": "test_output_sequence"},
                },
                {
                    "output_info": {
                        "output_name": "test_output_array_cnn",
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 3,
                            "allow_pooling": False,
                        },
                    },
                },
            ],
        },
    }

    return params


@pytest.mark.parametrize(
    "create_test_data",
    [
        {
            "task_type": "multi_task",
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "extras": {"array_dims": 1},
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task",
            "random_samples_dropped_from_modalities": True,
            "source": "local",
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    get_parametrization(),
    indirect=True,
)
def test_multi_serving(
    create_test_config_init_base: Tuple["TestConfigInits", "TestDataConfig"],
    prep_modelling_test_configs: Tuple[train.Experiment, "ModelTestConfig"],
):
    """
    We have the retries below as sometimes we can get random IDs of samples
    that have partially dropped modalities, and currently the function that
    crafts the example code (as well as the serving module) expects all
    modalities to be present. This is something we can fix/update in the future.
    """

    _, test_data_config = create_test_config_init_base
    experiment, test_config = prep_modelling_test_configs

    train.train(experiment=experiment)

    check_performance_result_wrapper(
        outputs=experiment.outputs,
        run_path=test_config.run_path,
        max_thresholds=(0.80, 0.80),
        min_thresholds=(2.0, 2.0),
    )

    saved_model_path = next((test_config.run_path / "saved_models").iterdir())

    command = ["eirserve", "--model-path", str(saved_model_path)]

    input_configs = experiment.configs.input_configs

    ids = []
    example_requests = []

    n_sample_requests = 10
    n_retries_per_request = 5
    for i in range(n_sample_requests):
        n_cur_retries = 0

        try:
            random_id, example_request = _craft_example_request(
                scoped_test_path=test_data_config.scoped_tmp_path,
                input_configs=input_configs,
            )
        except Exception as e:
            if n_cur_retries < n_retries_per_request:
                n_cur_retries += 1
                continue
            else:
                raise e

        ids.append(random_id)
        example_requests.append(example_request)

    response = run_serve_experiment_from_command(
        command=command,
        url="http://localhost:8000/predict",
        example_requests=example_requests,
        data_loading_function=load_data_for_serve,
    )

    for idx, random_id in enumerate(ids):
        cur_response = response[idx]
        assert _check_prediction(
            id_from_request=random_id,
            response=cur_response,
            output_configs=experiment.configs.output_configs,
            labels_csv_path=str(test_data_config.scoped_tmp_path / "labels.csv"),
            experiment=experiment,
        )


def _craft_example_request(
    scoped_test_path: Path,
    input_configs: Sequence[InputConfig],
) -> tuple[str, dict[str, Any]]:
    example_request = {}

    labels_df = pd.read_csv(scoped_test_path / "labels.csv")
    random_id = random.choice(labels_df["ID"].values)

    for config in input_configs:
        input_type = config.input_info.input_type
        input_name = config.input_info.input_name

        if input_type == "omics":
            omics_path = scoped_test_path / "omics" / f"{random_id}.npy"
            assert omics_path.is_file()
            example_request[input_name] = str(omics_path)

        elif input_type == "sequence":
            sequence_path = scoped_test_path / "sequence" / f"{random_id}.txt"
            assert sequence_path.is_file()
            example_request[input_name] = str(sequence_path)

        elif input_type == "image":
            image_path = scoped_test_path / "image" / f"{random_id}.png"
            assert image_path.is_file()
            example_request[input_name] = str(image_path)

        elif input_type == "array":
            array_path = scoped_test_path / "array" / f"{random_id}.npy"
            assert array_path.is_file()
            example_request[input_name] = str(array_path)

        elif input_type == "tabular":
            cat_columns = list(config.input_type_info.input_cat_columns)
            con_columns = list(config.input_type_info.input_con_columns)
            all_columns = cat_columns + con_columns

            tabular_data = {}

            for col in all_columns:
                value = labels_df.loc[labels_df["ID"] == random_id, col].iloc[0]

                # for now, we just fail here if we have a missing value
                # allowing only fully complete samples
                # later we can add something more sophisticated if needed
                if pd.isna(value):
                    assert False

                tabular_data[col] = value

            example_request[input_name] = tabular_data

    return random_id, example_request


def _check_prediction(
    id_from_request: str,
    response: dict,
    output_configs: Sequence[OutputConfig],
    labels_csv_path: str,
    experiment: train.Experiment,
) -> bool:
    labels_df = pd.read_csv(filepath_or_buffer=labels_csv_path)
    expected_row = labels_df[labels_df["ID"] == id_from_request]

    actual_results = response["response"]["result"]

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        if output_type == "tabular":
            actual_output = actual_results.get(output_name, {})
            if not _validate_tabular_output(
                actual_output=actual_output,
                expected_row=expected_row,
                output_config=output_config,
            ):
                return False

        elif output_type == "sequence":
            actual_output = actual_results.get(output_name, "")
            if not _validate_sequence_output(
                actual_output=actual_output,
                expected_set=get_expected_keywords_set(),
            ):
                return False

        elif output_type == "array":
            actual_output = actual_results.get(output_name, "")
            output_object = experiment.outputs[output_name]
            data_dimensions = output_object.data_dimensions
            array_folder = Path(labels_csv_path).parent / "array"
            expected_array_file = array_folder / f"{id_from_request}.npy"
            expected_array = np.load(expected_array_file)
            if not _validate_array_output(
                actual_output=actual_output,
                expected_array=expected_array,
                data_dimensions=data_dimensions.full_shape(),
            ):
                return False

    return True


def _validate_tabular_output(
    actual_output: dict,
    expected_row: pd.Series,
    output_config: OutputConfig,
) -> bool:
    for cat_col in output_config.output_type_info.target_cat_columns:
        expected_value = expected_row[cat_col].iloc[0]

        actual_category_predictions = actual_output.get(cat_col, {})
        if not actual_category_predictions:
            return False

        predicted_category = max(
            actual_category_predictions,
            key=actual_category_predictions.get,
        )

        if predicted_category != expected_value:
            return False

    for con_col in output_config.output_type_info.target_con_columns:
        expected_value = expected_row[con_col].iloc[0]
        actual_value = actual_output.get(con_col, None).get(con_col, None)
        if actual_value is None or not isclose(
            actual_value,
            expected_value,
            rel_tol=5.0,
        ):
            return False

    return True


def _validate_sequence_output(actual_output: str, expected_set: set) -> bool:
    content = actual_output.split(" ")

    intersection = set(content).intersection(expected_set)
    if not intersection:
        return False
    if len(intersection) < 2:
        return False

    return True


def _validate_array_output(
    actual_output: str,
    expected_array: np.ndarray,
    data_dimensions: tuple[int, ...],
    mse_threshold: float = 0.2,
    cosine_similarity_threshold: float = 0.6,
) -> bool:
    array_np = _deserialize_array(
        array_str=actual_output,
        dtype=np.float32,
        shape=data_dimensions,
    )

    mse = mean_squared_error(
        y_true=expected_array.ravel(),
        y_pred=array_np.ravel(),
    )
    assert mse < mse_threshold

    expected_array[expected_array < 1e-8] = 0.0

    cosine_similarity = 1 - cosine(
        u=expected_array.ravel().astype(np.float32),
        v=array_np.ravel(),
    )
    assert cosine_similarity > cosine_similarity_threshold

    return True
