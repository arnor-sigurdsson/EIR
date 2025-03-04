import shutil
import subprocess
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def api_process():
    file_target = "tests/test_modelling/test_streaming_modelling/setup_test_api.py"
    process = subprocess.Popen(
        [
            "python",
            file_target,
        ]
    )
    time.sleep(5)
    yield process
    process.terminate()
    process.wait()


def test_streaming_training(api_process):
    base = "tests/test_modelling/test_streaming_modelling"
    training_process = subprocess.Popen(
        [
            "eirtrain",
            "--global_configs",
            f"{base}/test_streaming_configs/global_config.yaml",
            "--input_configs",
            f"{base}/test_streaming_configs/sequence_config.yaml",
            f"{base}/test_streaming_configs/tabular_config.yaml",
            f"{base}/test_streaming_configs/omics_config.yaml",
            f"{base}/test_streaming_configs/array_config.yaml",
            f"{base}/test_streaming_configs/image_config.yaml",
            "--output_configs",
            f"{base}/test_streaming_configs/output_config.yaml",
            f"{base}/test_streaming_configs/output_array_config.yaml",
            f"{base}/test_streaming_configs/output_image_config.yaml",
            f"{base}/test_streaming_configs/output_sequence_config.yaml",
            f"{base}/test_streaming_configs/output_survival_config.yaml",
        ]
    )

    training_process.wait()

    output_folder = Path("runs/test_streaming_run")
    assert output_folder.exists()
    assert (output_folder / "results").exists()
    assert (output_folder / "saved_models").exists()

    shutil.rmtree(path=output_folder)
