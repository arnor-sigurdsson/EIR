from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd

from eir.data_load import label_setup
from eir.data_load.label_setup import (
    al_label_dict,
    al_label_transformers,
    TabularFileInfo,
    al_all_column_ops,
    transform_label_df,
)
from eir.experiment_io.experiment_io import load_transformers
from eir.predict_modules.predict_input_setup import prep_missing_con_dict
from eir.setup.config import Configs
from eir.train import df_to_nested_dict, get_tabular_target_file_infos


@dataclass
class PredictTargetLabels:
    label_dict: al_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return self.label_dict


def get_target_labels_for_predict(
    output_folder: str,
    tabular_file_infos: Dict[str, TabularFileInfo],
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> PredictTargetLabels:

    df_labels_test = pd.DataFrame(index=ids)
    label_transformers = {}
    con_columns = []
    cat_columns = []

    for output_name, tabular_info in tabular_file_infos.items():

        all_columns = list(tabular_info.cat_columns) + list(tabular_info.con_columns)
        if not all_columns:
            raise ValueError(f"No columns specified in {tabular_file_infos}.")

        con_columns += tabular_info.con_columns
        cat_columns += tabular_info.cat_columns

        df_cur_labels = _load_labels_for_predict(
            tabular_info=tabular_info,
            ids_to_keep=ids,
            custom_label_ops=custom_column_label_parsing_ops,
        )
        df_cur_labels["Output Name"] = output_name

        df_labels_test = pd.concat((df_labels_test, df_cur_labels))

        cur_transformers = load_transformers(
            output_folder=output_folder,
            transformers_to_load={output_name: all_columns},
        )
        label_transformers[output_name] = cur_transformers[output_name]

    df_labels_test = df_labels_test.set_index("Output Name", append=True)
    df_labels_test = df_labels_test.dropna(how="all")

    labels_dict = parse_labels_for_predict(
        con_columns=con_columns,
        cat_columns=cat_columns,
        df_labels_test=df_labels_test,
        label_transformers=label_transformers,
    )

    labels = PredictTargetLabels(
        label_dict=labels_dict, label_transformers=label_transformers
    )

    return labels


def _load_labels_for_predict(
    tabular_info: TabularFileInfo,
    ids_to_keep: Sequence[str],
    custom_label_ops: al_all_column_ops = None,
) -> pd.DataFrame:

    parse_wrapper = label_setup.get_label_parsing_wrapper(
        label_parsing_chunk_size=tabular_info.parsing_chunk_size
    )
    df_labels_test = parse_wrapper(
        label_file_tabular_info=tabular_info,
        ids_to_keep=ids_to_keep,
        custom_label_ops=custom_label_ops,
    )

    return df_labels_test


def parse_labels_for_predict(
    con_columns: Sequence[str],
    cat_columns: Sequence[str],
    df_labels_test: pd.DataFrame,
    label_transformers: Dict[str, al_label_transformers],
) -> al_label_dict:

    con_transformers = {k: v for k, v in label_transformers.items() if k in con_columns}
    train_con_column_means = prep_missing_con_dict(con_transformers=con_transformers)

    df_labels_test = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=cat_columns,
        con_label_columns=con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
    )

    assert len(label_transformers) > 0
    for name, output_transformer_set in label_transformers.items():
        df_labels_test = transform_label_df(
            df_labels=df_labels_test, label_transformers=output_transformer_set
        )

    test_labels_dict = df_to_nested_dict(df=df_labels_test)

    return test_labels_dict


def get_target_labels_for_testing(
    configs_overloaded_for_predict: Configs,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> PredictTargetLabels:
    """
    NOTE:   This can be extended to more tabular data, including other files, if we
            update the parameters slightly.
    """

    target_infos = get_tabular_target_file_infos(
        output_configs=configs_overloaded_for_predict.output_configs
    )

    target_labels = get_target_labels_for_predict(
        output_folder=configs_overloaded_for_predict.global_config.output_folder,
        tabular_file_infos=target_infos,
        custom_column_label_parsing_ops=custom_column_label_parsing_ops,
        ids=ids,
    )

    return target_labels
