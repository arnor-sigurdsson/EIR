from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence

import pandas as pd

from eir.data_load import label_setup
from eir.data_load.label_setup import (
    TabularFileInfo,
    al_all_column_ops,
    al_label_transformers,
    transform_label_df,
)
from eir.experiment_io.experiment_io import load_transformers
from eir.setup import schemas
from eir.setup.input_setup_modules import setup_tabular

if TYPE_CHECKING:
    from eir.train_utils.step_logic import Hooks


@dataclass
class ComputedPredictTabularInputInfo:
    labels: "PredictInputLabels"
    input_config: schemas.InputConfig


@dataclass
class PredictInputLabels:
    predict_labels: pd.DataFrame
    label_transformers: al_label_transformers

    @property
    def all_labels(self):
        return self.predict_labels


def setup_tabular_input_for_testing(
    input_config: schemas.InputConfig,
    ids: Sequence[str],
    hooks: "Hooks",
    output_folder: str,
) -> ComputedPredictTabularInputInfo:
    input_type_info = input_config.input_type_info
    assert isinstance(input_type_info, schemas.TabularInputDataConfig)

    tabular_file_info = setup_tabular.get_tabular_input_file_info(
        input_source=input_config.input_info.input_source,
        tabular_data_type_config=input_type_info,
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    predict_labels = get_input_labels_for_predict(
        tabular_file_info=tabular_file_info,
        input_name=input_config.input_info.input_name,
        custom_label_ops=custom_ops,
        ids=ids,
        output_folder=output_folder,
    )

    predict_input_info = ComputedPredictTabularInputInfo(
        labels=predict_labels, input_config=input_config
    )

    return predict_input_info


def get_input_labels_for_predict(
    tabular_file_info: TabularFileInfo,
    input_name: str,
    custom_label_ops: al_all_column_ops,
    ids: Sequence[str],
    output_folder: str,
) -> PredictInputLabels:
    if len(tabular_file_info.con_columns) + len(tabular_file_info.cat_columns) < 1:
        raise ValueError(f"No label columns specified in {tabular_file_info}.")

    parse_wrapper = label_setup.get_label_parsing_wrapper(
        label_parsing_chunk_size=tabular_file_info.parsing_chunk_size
    )
    df_labels_test = parse_wrapper(
        label_file_tabular_info=tabular_file_info,
        ids_to_keep=ids,
        custom_label_ops=custom_label_ops,
    )

    label_setup.pre_check_label_df(df=df_labels_test, name="Testing DataFrame")

    all_columns = list(tabular_file_info.con_columns) + list(
        tabular_file_info.cat_columns
    )
    label_transformers_with_input_name = load_transformers(
        transformers_to_load={input_name: all_columns}, output_folder=output_folder
    )
    loaded_fit_label_transformers = label_transformers_with_input_name[input_name]

    con_transformers = _extract_input_con_transformers(
        loaded_fit_label_transformers=loaded_fit_label_transformers,
        con_columns=tabular_file_info.con_columns,
    )
    train_con_column_means = prep_missing_con_dict(con_transformers=con_transformers)

    df_labels_test_no_na = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=tabular_file_info.cat_columns,
        con_label_columns=tabular_file_info.con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
        impute_missing=True,
    )

    df_labels_test_final = transform_label_df(
        df_labels=df_labels_test_no_na,
        label_transformers=loaded_fit_label_transformers,
        impute_missing=True,
    )

    labels_data_object = PredictInputLabels(
        predict_labels=df_labels_test_final,
        label_transformers=loaded_fit_label_transformers,
    )

    return labels_data_object


def _extract_input_con_transformers(
    loaded_fit_label_transformers, con_columns: Sequence[str]
):
    con_transformers = {
        k: v for k, v in loaded_fit_label_transformers.items() if k in con_columns
    }

    assert len(con_transformers) == len(con_columns)

    return con_transformers


def prep_missing_con_dict(con_transformers: al_label_transformers) -> Dict[str, float]:
    train_means = {
        column: transformer.mean_[0] for column, transformer in con_transformers.items()
    }

    return train_means
