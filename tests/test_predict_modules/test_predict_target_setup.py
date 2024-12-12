import pytest

from eir.predict_modules.predict_target_setup import _load_labels_for_predict
from eir.setup import config
from eir.target_setup.target_label_setup import (
    gather_all_ids_from_output_configs,
    get_tabular_target_file_infos,
)
from tests.setup_tests.fixtures_create_data import TestDataConfig


@pytest.mark.parametrize("create_test_data", [{"task_type": "multi"}], indirect=True)
@pytest.mark.parametrize(
    "create_test_config_init_base",
    [
        {
            "injections": {
                "global_configs": {
                    "basic_experiment": {
                        "output_folder": "extra_inputs",
                    }
                },
                "input_configs": [
                    {
                        "input_info": {"input_name": "test_genotype"},
                        "model_config": {"model_type": "linear"},
                    },
                    {
                        "input_info": {"input_name": "test_tabular"},
                        "input_type_info": {
                            "input_cat_columns": [],
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
                            "target_con_columns": ["Height"],
                        },
                    },
                ],
            },
        },
    ],
    indirect=True,
)
def test_load_labels_for_predict(
    create_test_data: TestDataConfig,
    create_test_config: config.Configs,
):
    """
    Note here we are treating the generated test data (i.e. by tests, not test-set-data)
    as the testing-set.
    """
    test_configs = create_test_config

    test_ids = gather_all_ids_from_output_configs(
        output_configs=test_configs.output_configs
    )

    tabular_infos = get_tabular_target_file_infos(
        output_configs=test_configs.output_configs
    )
    assert len(tabular_infos) == 1
    target_tabular_info = tabular_infos["test_output_tabular"]

    df_test = _load_labels_for_predict(
        tabular_info=target_tabular_info, ids_to_keep=test_ids
    )

    # make sure that target columns are unchanged (within expected bounds)
    assert len(target_tabular_info.con_columns) == 1
    con_target_column = target_tabular_info.con_columns[0]
    assert df_test[con_target_column].max() < 260
    assert df_test[con_target_column].min() > 120

    assert len(target_tabular_info.cat_columns) == 1
    cat_target_column = target_tabular_info.cat_columns[0]
    assert set(df_test[cat_target_column]) == {"Asia", "Africa", "Europe"}
