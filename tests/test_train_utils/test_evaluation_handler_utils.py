from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from eir.setup.input_setup_modules.setup_image import ImageNormalizationStats
from eir.train import Experiment
from eir.train_utils.evaluation_modules.evaluation_handlers_utils import (
    convert_image_input_to_raw,
    convert_tabular_input_to_raw,
    prepare_manual_sample_data,
)
from tests.setup_tests.fixtures_create_experiment import ModelTestConfig
from tests.setup_tests.setup_modelling_test_data.setup_array_test_data import (
    _set_up_base_test_array,
)
from tests.setup_tests.setup_modelling_test_data.setup_omics_test_data import (
    _set_up_base_test_omics_array,
)


def test_convert_image_input_to_raw():
    normalization_stats = ImageNormalizationStats(
        means=torch.Tensor([0.5, 0.5, 0.5]),
        stds=torch.Tensor([0.5, 0.5, 0.5]),
    )
    valid_input = torch.randn((3, 64, 64))
    valid_output = convert_image_input_to_raw(
        data=valid_input, normalization_stats=normalization_stats
    )
    assert isinstance(valid_output, Image.Image)

    invalid_input = torch.randn((1, 3, 64, 64))
    with pytest.raises(AssertionError):
        convert_image_input_to_raw(
            data=invalid_input, normalization_stats=normalization_stats
        )


def test_convert_tabular_input_to_raw():
    input_transformers = {}
    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, 1))
    input_transformers["column1"] = scaler

    valid_input = {"column1": torch.tensor([0.5])}
    valid_output = convert_tabular_input_to_raw(
        data=valid_input, input_transformers=input_transformers
    )
    assert isinstance(valid_output, dict)

    invalid_input = {"invalid_column": torch.tensor([0.5])}
    with pytest.raises(KeyError):
        convert_tabular_input_to_raw(
            data=invalid_input, input_transformers=input_transformers
        )

    edge_input = {"column1": torch.tensor([])}
    with pytest.raises(AssertionError):
        convert_tabular_input_to_raw(
            data=edge_input, input_transformers=input_transformers
        )


def _generate_manual_sample_test_data(tmp_path) -> dict[str, Any]:
    sample_inputs = {}

    # 1. Omics
    omics_array, *_ = _set_up_base_test_omics_array(n_snps=1000)
    omics_array = omics_array.astype(np.uint8)
    omics_file_path = tmp_path / "omics.npy"
    np.save(str(omics_file_path), omics_array)
    sample_inputs["test_genotype"] = Path(omics_file_path)

    # 2. Sequence
    sequence_data = "hello world"
    sample_inputs["test_sequence"] = sequence_data

    # 3. Bytes
    byte_data = b"some byte data"
    byte_data_file_path = tmp_path / "byte_data.bin"
    with byte_data_file_path.open("wb") as f:
        f.write(byte_data)
    sample_inputs["test_bytes"] = Path(byte_data_file_path)

    # 4. Image
    image_base = np.zeros((16, 16), dtype=np.uint8)
    img = Image.fromarray(image_base, mode="L")
    image_file_path = tmp_path / "image.png"
    img.save(image_file_path)
    sample_inputs["test_image"] = Path(image_file_path)

    # 5. Tabular
    tabular_data = {
        "OriginExtraCol": ["Europe"],
        "ExtraTarget": [0.1337],
    }
    sample_inputs["test_tabular"] = tabular_data

    # 6. Array
    array_data, _ = _set_up_base_test_array(dims=1, class_integer=0)
    array_file_path = tmp_path / "array.npy"
    np.save(array_file_path, array_data)
    sample_inputs["test_array"] = Path(array_file_path)

    return sample_inputs


@pytest.mark.parametrize(
    argnames="create_test_data",
    argvalues=[
        {
            "task_type": "multi",
            "split_to_test": True,
            "modalities": (
                "omics",
                "sequence",
                "image",
                "array",
            ),
            "source": "local",
            "extras": {"array_dims": 1},
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "test_manual_samples_preparation",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "genome-local-net"},
                    },
                    {
                        "input_info": {"input_name": "test_sequence"},
                    },
                    {
                        "input_info": {"input_name": "test_bytes"},
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
                        "model_config": {"model_type": "tabular"},
                    },
                    {
                        "input_info": {"input_name": "test_array"},
                        "model_config": {
                            "model_type": "cnn",
                            "model_init_config": {
                                "rb_do": 0.25,
                                "channel_exp_base": 3,
                                "l1": 1e-04,
                                "kernel_height": 1,
                            },
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
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": [],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_prepare_sequence_output_manual_sample_data(
    prep_modelling_test_configs: tuple["Experiment", "ModelTestConfig"],
    tmp_path: Path,
):
    experiment, test_config = prep_modelling_test_configs

    input_objects = experiment.inputs

    test_data = _generate_manual_sample_test_data(tmp_path=tmp_path)
    prepared_test_data = prepare_manual_sample_data(
        sample_inputs=test_data, input_objects=input_objects
    )

    expected_keys = [
        "test_genotype",
        "test_tabular",
        "test_sequence",
        "test_bytes",
        "test_image",
        "test_array",
    ]
    assert set(prepared_test_data.keys()) == set(expected_keys)

    assert prepared_test_data["test_genotype"].shape == (1, 4, 1000)

    assert prepared_test_data["test_image"].shape == (1, 16, 16)

    assert prepared_test_data["test_sequence"].shape == (63,)

    assert prepared_test_data["test_array"].shape == (1, 1, 100)

    assert set(prepared_test_data["test_tabular"].keys()) == {
        "OriginExtraCol",
        "ExtraTarget",
    }
    assert prepared_test_data["test_tabular"]["OriginExtraCol"].dtype == torch.int64
    assert prepared_test_data["test_tabular"]["ExtraTarget"].dtype in (
        torch.float64,
        torch.float32,
    )

    assert prepared_test_data["test_bytes"].shape == (128,)
