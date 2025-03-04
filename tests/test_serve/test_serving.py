import asyncio
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from math import isclose
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error

from docs.doc_modules.serve_experiments_utils import load_data_for_serve
from eir import train
from eir.data_load.data_preparation_modules.prepare_array import fill_nans_from_stats
from eir.serve import app, load_experiment
from eir.serve_modules.serve_api import create_info_endpoint, create_predict_endpoint
from eir.serve_modules.serve_network_utils import deserialize_array
from eir.setup.schemas import InputConfig, OutputConfig
from eir.utils.logging import get_logger
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


@contextmanager
def run_server_in_thread(app: FastAPI, port: int = 8888) -> uvicorn.Server:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config=config)

    async def serve():
        await server.serve()

    with ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, serve())
        time.sleep(2)
        try:
            yield server
        finally:
            server.should_exit = True
            future.result()


def get_base_parametrization(compiled: bool = False) -> dict:
    params = {
        "injections": {
            "global_configs": {
                "basic_experiment": {
                    "output_folder": "multi_task_multi_modal",
                    "n_epochs": 12,
                },
                "optimization": {
                    "gradient_clipping": 1.0,
                    "lr": 0.001,
                    "wd": 1e-04,
                },
                "model": {
                    "compile_model": compiled,
                },
                "evaluation_checkpoint": {
                    "checkpoint_interval": 200,
                },
            },
            "input_configs": [
                {
                    "input_info": {"input_name": "test_genotype"},
                    "model_config": {
                        "model_type": "genome-local-net",
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
                            "kernel_width": 3,
                            "down_stride_height": 1,
                            "down_stride_width": 1,
                            "attention_inclusion_cutoff": 256,
                            "allow_first_conv_size_reduction": False,
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
                    "fc_task_dim": 512,
                    "fc_do": 0.10,
                    "rb_do": 0.10,
                    "layers": [2],
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
                {
                    "output_info": {
                        "output_name": "test_array",
                    },
                    "output_type_info": {
                        "loss": "diffusion",
                        "diffusion_time_steps": 50,
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 3,
                            "allow_pooling": False,
                        },
                    },
                },
                {
                    "output_info": {
                        "output_name": "test_output_image",
                    },
                    "output_type_info": {
                        "loss": "mse",
                        "size": [16, 16],
                    },
                    "model_config": {
                        "model_type": "cnn",
                        "model_init_config": {
                            "channel_exp_base": 4,
                            "allow_pooling": False,
                        },
                    },
                },
                {
                    "output_info": {"output_name": "test_output_survival"},
                    "output_type_info": {
                        "event_column": "BinaryOrigin",
                        "time_column": "Time",
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
            "manual_test_data_creator": lambda: "test_multi_modal_multi_task_serving",
            "random_samples_dropped_from_modalities": False,
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
    create_test_config_init_base: tuple["TestConfigInits", "TestDataConfig"],
    prep_modelling_test_configs: tuple[train.Experiment, "ModelTestConfig"],
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

    serve_experiment = load_experiment(model_path=str(saved_model_path), device="cpu")

    app.state.serve_experiment = serve_experiment

    create_info_endpoint(
        app=app,
        serve_experiment=serve_experiment,
    )

    create_predict_endpoint(
        app=app,
        configs=serve_experiment.configs.input_configs,
        serve_experiment=serve_experiment,
    )

    client = TestClient(app=app)

    input_configs = experiment.configs.input_configs

    ids = []
    example_requests = []

    n_sample_requests = 10
    n_retries_per_request = 10
    for _i in range(n_sample_requests):
        n_cur_retries = 0

        for _ in range(n_retries_per_request):
            try:
                random_id, example_request = _craft_example_request(
                    scoped_test_path=test_data_config.scoped_tmp_path,
                    input_configs=input_configs,
                )
            except Exception as e:
                logger.error(f"Failed to craft example request: {e}")
                if n_cur_retries < n_retries_per_request:
                    n_cur_retries += 1
                    continue
                raise e

            ids.append(random_id)
            example_requests.append(example_request)

    assert len(example_requests) > 0

    with run_server_in_thread(app=app):
        all_responses = []
        for example_request in example_requests:
            loaded_items = load_data_for_serve(data=example_request)
            response = client.post(url="/predict", json=[loaded_items])
            assert response.status_code == 200
            result = response.json()
            all_responses.append(result["result"])

        assert len(all_responses) >= 10

        pass_threshold = 0.6
        n_passed = 0
        n_total = 0
        for idx, random_id in enumerate(ids):
            cur_response = all_responses[idx][0]
            passed = _check_prediction(
                id_from_request=random_id,
                response=cur_response,
                output_configs=experiment.configs.output_configs,
                labels_csv_path=str(test_data_config.scoped_tmp_path / "labels.csv"),
                experiment=experiment,
            )
            if passed:
                n_passed += 1
            n_total += 1

        assert n_passed / n_total >= pass_threshold


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
            assert omics_path.is_file(), f"Missing omics file: {omics_path}"
            example_request[input_name] = str(omics_path)

        elif input_type == "sequence":
            sequence_path = scoped_test_path / "sequence" / f"{random_id}.txt"
            assert sequence_path.is_file(), f"Missing sequence file: {sequence_path}"
            example_request[input_name] = str(sequence_path)

        elif input_type == "image":
            image_path = scoped_test_path / "image" / f"{random_id}.png"
            assert image_path.is_file(), f"Missing image file: {image_path}"
            example_request[input_name] = str(image_path)

        elif input_type == "array":
            array_path = scoped_test_path / "array" / f"{random_id}.npy"
            assert array_path.is_file(), f"Missing array file: {array_path}"
            example_request[input_name] = str(array_path)

        elif input_type == "tabular":
            cat_columns = list(config.input_type_info.input_cat_columns)
            con_columns = list(config.input_type_info.input_con_columns)
            all_columns = cat_columns + con_columns

            any_na = (
                labels_df.loc[labels_df["ID"] == random_id, all_columns]
                .isna()
                .any(axis=1)
                .iloc[0]
            )

            if any_na:
                raise AssertionError(f"Missing values for ID: {random_id}")

            tabular_data = {}
            for col in all_columns:
                value = labels_df.loc[labels_df["ID"] == random_id, col].iloc[0]

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

    actual_result = response

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        if output_type == "tabular":
            actual_output = actual_result.get(output_name, {})
            if not _validate_tabular_output(
                actual_output=actual_output,
                expected_row=expected_row,
                output_config=output_config,
            ):
                return False

        elif output_type == "sequence":
            actual_output = actual_result.get(output_name, "")
            if not _validate_sequence_output(
                actual_output=actual_output,
                expected_set=get_expected_keywords_set(),
            ):
                return False

        elif output_type == "array":
            actual_output = actual_result.get(output_name, "")
            output_object = experiment.outputs[output_name]
            data_dimensions = output_object.data_dimensions
            array_folder = Path(labels_csv_path).parent / "array"
            expected_array_file = array_folder / f"{id_from_request}.npy"
            expected_array = np.load(expected_array_file)

            expected_ndim = len(output_object.normalization_stats.shape)

            tensor = torch.from_numpy(expected_array).float()
            while len(tensor.shape) < expected_ndim:
                tensor = tensor.unsqueeze(0)

            expected_array_no_nan = fill_nans_from_stats(
                array=tensor,
                normalization_stats=output_object.normalization_stats,
            )
            expected_array_no_nan_npy = expected_array_no_nan.numpy()

            do_check_cosine = not output_object.diffusion_config
            if not _validate_array_output(
                actual_output=actual_output,
                expected_array=expected_array_no_nan_npy,
                data_dimensions=data_dimensions.full_shape(),
                check_cosine=do_check_cosine,
            ):
                return False
        elif output_type == "image":
            actual_output = actual_result.get(output_name, "")
            image_folder = Path(labels_csv_path).parent / "image"
            expected_image_file = image_folder / f"{id_from_request}.png"
            expected_image = np.array(Image.open(expected_image_file)) / 255.0
            if not _validate_array_output(
                actual_output=actual_output,
                expected_array=expected_image,
                data_dimensions=(1, 16, 16),
                check_cosine=False,
            ):
                return False
        elif output_type == "survival":
            actual_output = actual_result.get(output_name, {})
            if not _validate_survival_output(
                actual_output=actual_output,
                expected_row=expected_row,
                output_config=output_config,
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

        if pd.isna(expected_value):
            logger.info("Skipping NA value.")
            continue

        actual_category_predictions = actual_output.get(cat_col, {})
        if not actual_category_predictions:
            logger.error(f"Missing category predictions for: {cat_col}")
            return False

        predicted_category = max(
            actual_category_predictions,
            key=actual_category_predictions.get,
        )

        if predicted_category != expected_value:
            logger.error(
                f"Expected: {expected_value}, got: {predicted_category} for: {cat_col}",
            )
            return False

    for con_col in output_config.output_type_info.target_con_columns:
        expected_value = expected_row[con_col].iloc[0]

        if pd.isna(expected_value):
            logger.info("Skipping NA value.")
            continue

        actual_value = actual_output.get(con_col).get(con_col, None)
        if actual_value is None or not isclose(
            actual_value,
            expected_value,
            rel_tol=5.0,
        ):
            logger.error(
                f"Expected: {expected_value}, got: {actual_value} for: {con_col}",
            )
            return False

    return True


def _validate_sequence_output(actual_output: str, expected_set: set) -> bool:
    content = actual_output.split(" ")

    intersection = set(content).intersection(expected_set)
    if not intersection:
        return False
    return not len(intersection) < 2


def _validate_array_output(
    actual_output: str,
    expected_array: np.ndarray,
    data_dimensions: tuple[int, ...],
    mse_threshold: float = 0.35,
    cosine_similarity_threshold: float = 0.6,
    check_cosine: bool = False,
) -> bool:
    array_np = deserialize_array(
        array_str=actual_output,
        dtype=np.float32,
        shape=data_dimensions,
    )

    mse = mean_squared_error(
        y_true=expected_array.ravel(),
        y_pred=array_np.ravel(),
    )
    if not mse < mse_threshold:
        logger.error(f"High MSE: {mse}")
        return False

    expected_array[expected_array < 1e-8] = 0.0

    if check_cosine:
        cosine_similarity = 1 - cosine(
            u=expected_array.ravel().astype(np.float32),
            v=array_np.ravel(),
        )

        if expected_array.sum() == 0.0:
            logger.info("Skipping cosine similarity check for zero array.")
            return True

        if pd.isna(cosine_similarity):
            logger.error(f"Invalid cosine similarity: {cosine_similarity}")
            return False

        if not cosine_similarity > cosine_similarity_threshold:
            logger.error(f"Low cosine similarity: {cosine_similarity}")
            return False

    return True


def _validate_survival_output(
    actual_output: dict,
    expected_row: pd.Series,
    output_config: OutputConfig,
) -> bool:
    actual_output = actual_output["BinaryOrigin"]

    if not {"time_points", "survival_probs"}.issubset(actual_output.keys()):
        logger.error("Missing required keys in survival output")
        return False

    survival_probs = actual_output["survival_probs"]

    if not all(x > y for x, y in zip(survival_probs, survival_probs[1:], strict=False)):
        logger.error("Survival probabilities are not strictly decreasing")
        return False

    if not all(0 <= p <= 1 for p in survival_probs):
        logger.error("Survival probabilities outside valid range [0, 1]")
        return False

    binary_origin = expected_row["BinaryOrigin"].item()
    final_prob = survival_probs[-1]

    threshold = 0.70
    if binary_origin == 0:
        if final_prob < threshold:
            logger.error(
                f"Final probability below threshold: {final_prob}, "
                f"expected >= {threshold} for non-event",
            )
            return False
    else:
        if final_prob > (1.0 - threshold):
            logger.error(
                f"Final probability above threshold: {final_prob}, "
                f"expected <= {1.0 - threshold} for event",
            )
            return False

    return True
