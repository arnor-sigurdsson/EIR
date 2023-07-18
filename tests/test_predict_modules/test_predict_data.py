from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
                "global_configs": {"memory_dataset": True},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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
                "global_configs": {"memory_dataset": False},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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
    target_tabular_info = tabular_file_infos["test_output"]

    df_labels_test = pd.DataFrame(index=test_ids)
    for output_name, tabular_info in tabular_file_infos.items():
        all_columns = list(tabular_info.cat_columns) + list(tabular_info.con_columns)
        if not all_columns:
            raise ValueError(f"No columns specified in {tabular_file_infos}.")

        df_cur_labels = predict_target_setup._load_labels_for_predict(
            tabular_info=tabular_info,
            ids_to_keep=test_ids,
            custom_label_ops=None,
        )
        df_cur_labels["Output Name"] = output_name

        df_labels_test = pd.concat((df_labels_test, df_cur_labels))

    df_labels_test = df_labels_test.set_index("Output Name", append=True)
    df_labels_test = df_labels_test.dropna(how="all")
    df_labels_test, dropped_con_indices = set_random_con_targets_to_missing(
        df_labels_test=df_labels_test
    )

    assert len(target_tabular_info.cat_columns) == 1
    assert len(target_tabular_info.con_columns) == 1
    cat_column = target_tabular_info.cat_columns[0]
    con_column = target_tabular_info.con_columns[0]

    transformers = _get_mock_transformers(
        cat_target_column=cat_column,
        con_target_column=con_column,
    )

    test_target_labels = None
    if with_target_labels:
        test_target_labels = predict_target_setup.parse_labels_for_predict(
            con_columns=target_tabular_info.con_columns,
            cat_columns=target_tabular_info.cat_columns,
            df_labels_test=df_labels_test,
            all_output_label_transformers=transformers,
        )

    con_transformer = transformers["test_output"][con_column]
    con_mean = con_transformer.mean_.reshape(1, -1)
    expected_transformed_value = con_transformer.transform(con_mean).squeeze()

    all_dropped_ids = set(i[0] for i in dropped_con_indices)
    for id_, target in test_target_labels.items():
        cur_value = target["test_output"][con_column]
        if id_ in all_dropped_ids:
            assert cur_value == expected_transformed_value


def set_random_con_targets_to_missing(
    df_labels_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    fraction = 0.2

    num_rows_to_nan = int(fraction * len(df_labels_test))

    random_indices = np.random.choice(
        df_labels_test.index, num_rows_to_nan, replace=False
    )

    df_labels_test.loc[random_indices, "Height"] = np.nan

    return df_labels_test, random_indices


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {"memory_dataset": True},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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
                "global_configs": {"memory_dataset": False},
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                ],
                "output_configs": [
                    {
                        "output_info": {"output_name": "test_output"},
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

    test_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )

    tabular_file_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(tabular_file_infos) == 1
    target_tabular_info = tabular_file_infos["test_output"]

    df_labels_test = pd.DataFrame(index=test_ids)
    for output_name, tabular_info in tabular_file_infos.items():
        all_columns = list(tabular_info.cat_columns) + list(tabular_info.con_columns)
        if not all_columns:
            raise ValueError(f"No columns specified in {tabular_file_infos}.")

        df_cur_labels = predict_target_setup._load_labels_for_predict(
            tabular_info=tabular_info,
            ids_to_keep=test_ids,
            custom_label_ops=None,
        )
        df_cur_labels["Output Name"] = output_name

        df_labels_test = pd.concat((df_labels_test, df_cur_labels))

    df_labels_test = df_labels_test.set_index("Output Name", append=True)
    df_labels_test = df_labels_test.dropna(how="all")

    assert len(target_tabular_info.cat_columns) == 1
    assert len(target_tabular_info.con_columns) == 1
    transformers = _get_mock_transformers(
        cat_target_column=target_tabular_info.cat_columns[0],
        con_target_column=target_tabular_info.con_columns[0],
    )

    test_target_labels = None
    if with_target_labels:
        test_target_labels = predict_target_setup.parse_labels_for_predict(
            con_columns=target_tabular_info.con_columns,
            cat_columns=target_tabular_info.cat_columns,
            df_labels_test=df_labels_test,
            all_output_label_transformers=transformers,
        )

    test_inputs = predict_input_setup.set_up_inputs_for_predict(
        test_inputs_configs=test_configs.input_configs,
        ids=test_ids,
        hooks=None,
        output_folder=test_configs.global_config.output_folder,
    )

    outputs_as_dict = set_up_outputs_for_training(
        output_configs=create_test_config.output_configs,
        input_objects=test_inputs,
        target_transformers=transformers,
    )

    test_dataset = predict_data.set_up_default_dataset(
        configs=test_configs,
        target_labels_dict=test_target_labels,
        inputs_as_dict=test_inputs,
        outputs_as_dict=outputs_as_dict,
    )

    classes_tested = sorted(list(test_data_config.target_classes.keys()))
    exp_no_samples = test_data_config.n_per_class * len(classes_tested)

    check_dataset(
        dataset=test_dataset,
        exp_no_sample=exp_no_samples,
        classes_tested=classes_tested,
        target_transformers=transformers,
        target_column=target_tabular_info.cat_columns[0],
        check_targets=with_target_labels,
    )


def _get_mock_transformers(cat_target_column: str, con_target_column: str):
    mock_label_encoder = LabelEncoder().fit(["Asia", "Europe", "Africa"])
    transformers = {"test_output": {cat_target_column: mock_label_encoder}}
    mock_standard_scaler = StandardScaler().fit([[1], [2], [3]])
    transformers["test_output"][con_target_column] = mock_standard_scaler

    return transformers
