from unittest import mock

import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir import train
from eir.predict_modules import predict_data, predict_input_setup, predict_target_setup
from eir.setup import config
from eir.setup.output_setup import set_up_outputs_for_training
from eir.target_setup.target_label_setup import (
    gather_all_ids_from_output_configs,
    get_tabular_target_file_infos,
)
from tests.setup_tests.fixtures_create_data import TestDataConfig
from tests.test_data_load.test_datasets import check_dataset


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {"basic_experiment": {"memory_dataset": True}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {"basic_experiment": {"memory_dataset": False}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("with_target_labels", [True])
def test_set_up_test_labels(
    create_test_data: TestDataConfig,
    create_test_config: config.Configs,
    with_target_labels: bool,
):
    test_configs = create_test_config

    test_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )

    tabular_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(tabular_file_infos) == 1
    target_tabular_info = tabular_file_infos["test_output_tabular"]

    with mock.patch(
        target="eir.predict_modules.predict_target_setup.load_transformers",
        return_value=_get_mock_transformers(
            cat_target_column=target_tabular_info.cat_columns[0],
            con_target_column=target_tabular_info.con_columns[0],
        ),
    ):
        merged_test_target_labels = predict_target_setup.get_target_labels_for_testing(
            configs_overloaded_for_predict=test_configs,
            ids=test_ids,
        )

    df_labels = merged_test_target_labels.predict_labels
    transformers = merged_test_target_labels.label_transformers["test_output_tabular"]

    assert len(target_tabular_info.cat_columns) == 1
    assert len(target_tabular_info.con_columns) == 1
    con_column = target_tabular_info.con_columns[0]
    cat_column = target_tabular_info.cat_columns[0]

    assert df_labels[f"test_output_tabular__{con_column}"].dtype == pl.Float32
    assert df_labels[f"test_output_tabular__{cat_column}"].dtype == pl.Float32

    df_raw_data = pl.read_csv(source=target_tabular_info.file_path)
    con_transformer = transformers["test_output_tabular"][con_column]

    func = con_transformer.transform
    expected_transformed = func(df_raw_data["Height"].to_numpy().reshape(-1, 1))
    expected_transformed = expected_transformed.squeeze()

    actually_transformed = df_labels[f"test_output_tabular__{con_column}"].to_numpy()
    assert np.allclose(actually_transformed, expected_transformed)


def set_random_con_targets_to_missing(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray]:
    fraction = 0.2
    num_rows_to_nan = int(fraction * len(df))
    random_indices = np.random.randint(0, len(df), num_rows_to_nan)

    return (
        df.with_row_index()
        .with_columns(
            pl.when(pl.col("index").is_in(random_indices))
            .then(None)
            .otherwise(pl.col("test_output_tabular__Height"))
            .cast(pl.Float32)
            .alias("test_output_tabular__Height")
        )
        .drop("index"),
        random_indices,
    )


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {"basic_experiment": {"memory_dataset": True}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
        {
            "injections": {
                "global_configs": {"basic_experiment": {"memory_dataset": False}},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output_tabular"},
                        "output_type_info": {
                            "target_cat_columns": ["Origin"],
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("with_target_labels", [True, False])
def test_set_up_test_dataset(
    create_test_data: TestDataConfig,
    create_test_config: config.Configs,
    with_target_labels: bool,
):
    test_data_config = create_test_data
    test_configs = create_test_config

    hooks = train.get_default_hooks(configs=test_configs)
    train.get_default_experiment(configs=test_configs, hooks=hooks)

    test_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )

    tabular_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(tabular_file_infos) == 1
    target_tabular_info = tabular_file_infos["test_output_tabular"]

    with mock.patch(
        target="eir.predict_modules.predict_target_setup.load_transformers",
        return_value=_get_mock_transformers(
            cat_target_column=target_tabular_info.cat_columns[0],
            con_target_column=target_tabular_info.con_columns[0],
        ),
    ):
        test_target_labels = predict_target_setup.get_target_labels_for_testing(
            configs_overloaded_for_predict=test_configs,
            ids=test_ids,
        )

    transformers = test_target_labels.label_transformers["test_output_tabular"]

    test_inputs = predict_input_setup.set_up_inputs_for_predict(
        test_inputs_configs=test_configs.input_configs,
        ids=test_ids,
        hooks=None,
        output_folder=test_configs.gc.be.output_folder,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=test_inputs,
        target_transformers=transformers,
    )

    test_dataset = predict_data.set_up_default_dataset(
        configs=test_configs,
        target_labels_df=test_target_labels.predict_labels,
        inputs_as_dict=test_inputs,
        outputs_as_dict=outputs_as_dict,
        missing_ids_per_output=test_target_labels.missing_ids_per_output,
    )

    classes_tested = sorted(test_data_config.target_classes.keys())
    exp_no_samples = test_data_config.n_per_class * len(classes_tested)

    check_dataset(
        dataset=test_dataset,
        exp_no_sample=exp_no_samples,
        classes_tested=classes_tested,
        target_transformers=transformers,
        target_column=target_tabular_info.cat_columns[0],
        check_targets=with_target_labels,
    )


def _get_mock_transformers(
    cat_target_column: str,
    con_target_column: str,
):
    mock_label_encoder = LabelEncoder().fit(["Asia", "Europe", "Africa"])
    transformers = {"test_output_tabular": {cat_target_column: mock_label_encoder}}
    mock_standard_scaler = StandardScaler().fit([[1], [2], [3]])
    transformers["test_output_tabular"][con_target_column] = mock_standard_scaler

    return transformers
