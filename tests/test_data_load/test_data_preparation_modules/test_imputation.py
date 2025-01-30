from typing import Any

import pytest
import torch

from eir import train
from eir.data_load.data_preparation_modules import imputation
from eir.setup.config import Configs
from eir.target_setup.target_label_setup import gather_all_ids_from_output_configs
from tests.setup_tests.fixtures_create_data import TestDataConfig


@pytest.mark.parametrize(
    "create_test_data",
    [
        {"task_type": "binary", "modalities": ["omics", "sequence"]},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
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
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": ["OriginExtraCol"],
                            "input_con_columns": ["ExtraTarget"],
                        },
                        "model_config": {"model_type": "tabular"},
                    },
                ],
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
        }
    ],
    indirect=True,
)
def test_impute_missing_modalities(
    create_test_config: Configs,
    create_test_data: "TestDataConfig",
    parse_test_cl_args: dict[str, Any],
):
    test_experiment_config = create_test_config
    test_data_config = create_test_data

    all_array_ids = gather_all_ids_from_output_configs(
        output_configs=test_experiment_config.output_configs
    )
    train_ids, valid_ids = train.split_ids(
        ids=all_array_ids, valid_size=test_experiment_config.gc.be.valid_size
    )

    input_objects = train.set_up_inputs_for_training(
        inputs_configs=test_experiment_config.input_configs,
        train_ids=train_ids,
        valid_ids=valid_ids,
        hooks=None,
    )
    impute_dtypes = imputation._get_default_impute_dtypes(inputs_objects=input_objects)
    impute_fill_values = imputation._get_default_impute_fill_values(
        inputs_objects=input_objects
    )

    test_inputs_all_avail = {k: torch.empty(10) for k in input_objects}

    no_fill = imputation.impute_missing_modalities(
        inputs_values=test_inputs_all_avail,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )
    assert no_fill == test_inputs_all_avail

    test_inputs_missing_tabular = {
        k: v for k, v in test_inputs_all_avail.items() if "tabular" not in k
    }
    filled_tabular = imputation.impute_missing_modalities(
        inputs_values=test_inputs_missing_tabular,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )

    transformers = input_objects["test_tabular"].labels.label_transformers
    origin_extra_label_encoder = transformers["OriginExtraCol"]
    na_transformer_index = origin_extra_label_encoder.transform(["__NULL__"]).item()
    filled_na_value = filled_tabular["test_tabular"]["OriginExtraCol"]
    assert na_transformer_index == filled_na_value
    assert filled_tabular["test_tabular"]["ExtraTarget"] == 0.0

    test_inputs_missing_omics = {
        k: v for k, v in test_inputs_all_avail.items() if k != "test_genotype"
    }
    with_imputed_omics = imputation.impute_missing_modalities(
        inputs_values=test_inputs_missing_omics,
        inputs_objects=input_objects,
        fill_values=impute_fill_values,
        dtypes=impute_dtypes,
    )
    assert len(with_imputed_omics) == 3
    assert with_imputed_omics["test_genotype"].numel() == test_data_config.n_snps * 4


def test_impute_single_missing_modality():
    imputed_test_tensor = imputation.impute_single_missing_modality(
        shape=(10, 10),
        fill_value=0,
        dtype=torch.float,
        approach="constant",
    )
    assert imputed_test_tensor.numel() == 100
    assert len(imputed_test_tensor.shape) == 2
    assert imputed_test_tensor.shape[0] == 10
    assert imputed_test_tensor.shape[1] == 10

    assert (imputed_test_tensor == 0.0).all()


def test_impute_single_missing_modality_random():
    imputed_test_tensor = imputation.impute_single_missing_modality(
        shape=(10, 10),
        fill_value=0,
        dtype=torch.float,
        approach="random",
    )
    assert imputed_test_tensor.numel() == 100
    assert len(imputed_test_tensor.shape) == 2
    assert imputed_test_tensor.shape[0] == 10
    assert imputed_test_tensor.shape[1] == 10

    assert not (imputed_test_tensor == 0.0).all()
