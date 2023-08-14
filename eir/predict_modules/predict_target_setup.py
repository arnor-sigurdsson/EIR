from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from eir.data_load import label_setup
from eir.data_load.label_setup import (
    TabularFileInfo,
    al_all_column_ops,
    al_label_dict,
    al_label_transformers,
    al_target_label_dict,
    transform_label_df,
)
from eir.experiment_io.experiment_io import load_transformers
from eir.predict_modules.predict_tabular_input_setup import prep_missing_con_dict
from eir.setup.config import Configs
from eir.setup.schemas import (
    ArrayOutputTypeConfig,
    OutputConfig,
    SequenceOutputTypeConfig,
    TabularOutputTypeConfig,
)
from eir.target_setup.target_label_setup import (
    df_to_nested_dict,
    gather_data_pointers_from_data_source,
    get_tabular_target_file_infos,
)


@dataclass
class PredictTargetLabels:
    label_dict: al_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return self.label_dict


@dataclass
class MergedPredictTargetLabels:
    label_dict: al_target_label_dict
    label_transformers: dict[str, dict[str, al_label_transformers]]

    @property
    def all_labels(self):
        return self.label_dict


def get_tabular_target_labels_for_predict(
    output_folder: str,
    tabular_info: TabularFileInfo,
    output_name: str,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> PredictTargetLabels:
    all_columns = list(tabular_info.cat_columns) + list(tabular_info.con_columns)
    if not all_columns:
        raise ValueError(f"No columns specified in {tabular_info}.")

    con_columns = list(tabular_info.con_columns)
    cat_columns = list(tabular_info.cat_columns)

    df_labels_test = _load_labels_for_predict(
        tabular_info=tabular_info,
        ids_to_keep=ids,
        custom_label_ops=custom_column_label_parsing_ops,
    )

    label_transformers = load_transformers(
        output_folder=output_folder,
        transformers_to_load={output_name: all_columns},
    )

    labels_dict = parse_labels_for_predict(
        con_columns=con_columns,
        cat_columns=cat_columns,
        df_labels_test=df_labels_test,
        label_transformers=label_transformers[output_name],
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
    label_transformers: al_label_transformers,
) -> al_label_dict:
    con_transformers = _extract_target_con_transformers(
        label_transformers=label_transformers,
        con_columns=con_columns,
    )
    train_con_column_means = prep_missing_con_dict(con_transformers=con_transformers)

    df_labels_test = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=cat_columns,
        con_label_columns=con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
    )

    assert len(label_transformers) > 0
    df_labels_test = transform_label_df(
        df_labels=df_labels_test, label_transformers=label_transformers
    )

    test_labels_dict = df_labels_test.to_dict(orient="index")

    return test_labels_dict


def _extract_target_con_transformers(
    label_transformers: al_label_transformers,
    con_columns: Sequence[str],
) -> Dict[str, StandardScaler]:
    con_transformers = {}

    for target_column, transformer_object in label_transformers.items():
        if target_column not in con_columns:
            continue

        assert target_column not in con_transformers
        assert isinstance(transformer_object, StandardScaler)

        con_transformers[target_column] = transformer_object

    assert len(con_transformers) == len(con_columns)

    return con_transformers


def get_target_labels_for_testing(
    configs_overloaded_for_predict: Configs,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
) -> MergedPredictTargetLabels:
    pc = configs_overloaded_for_predict
    output_configs = pc.output_configs

    df_labels_test = pd.DataFrame(index=list(ids))
    tabular_label_transformers = {}

    tabular_target_files_info = get_tabular_target_file_infos(
        output_configs=pc.output_configs
    )

    for output_config in output_configs:
        output_type_info = output_config.output_type_info
        output_name = output_config.output_info.output_name

        match output_type_info:
            case TabularOutputTypeConfig():
                cur_tabular_info = tabular_target_files_info[output_name]
                cur_labels = get_tabular_target_labels_for_predict(
                    output_folder=pc.global_config.output_folder,
                    tabular_info=cur_tabular_info,
                    output_name=output_name,
                    custom_column_label_parsing_ops=custom_column_label_parsing_ops,
                    ids=ids,
                )
                tabular_label_transformers[output_name] = cur_labels.label_transformers

            case ArrayOutputTypeConfig():
                cur_labels = _set_up_predict_file_target_labels(
                    ids=ids,
                    output_config=output_config,
                )

            case SequenceOutputTypeConfig():
                cur_labels = set_up_delayed_predict_target_labels(
                    ids=ids,
                    output_name=output_name,
                )

            case _:
                raise NotImplementedError(
                    f"Output type {output_type_info} " f"not implemented"
                )

        df_labels_cur = pd.DataFrame.from_dict(cur_labels.label_dict, orient="index")

        df_labels_cur["Output Name"] = output_name

        df_labels_test = pd.concat((df_labels_test, df_labels_cur))

    df_labels_test = df_labels_test.set_index("Output Name", append=True)

    df_labels_test = df_labels_test.dropna(how="all")

    test_labels_dict = df_to_nested_dict(df=df_labels_test)

    test_labels_object = MergedPredictTargetLabels(
        label_dict=test_labels_dict,
        label_transformers=tabular_label_transformers,
    )

    return test_labels_object


def _set_up_predict_file_target_labels(
    ids: Sequence[str],
    output_config: OutputConfig,
) -> PredictTargetLabels:
    output_name = output_config.output_info.output_name
    output_source = output_config.output_info.output_source

    id_to_data_pointer_mapping = gather_data_pointers_from_data_source(
        data_source=Path(output_source),
        validate=True,
    )

    ids_set = set(ids)
    labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping[id_]} for id_ in ids_set
    }

    return PredictTargetLabels(
        label_dict=labels,
        label_transformers={},
    )


def set_up_delayed_predict_target_labels(
    ids: Sequence[str],
    output_name: str,
) -> PredictTargetLabels:
    ids_set = set(ids)
    labels: al_label_dict = {id_: {output_name: torch.nan} for id_ in ids_set}

    return PredictTargetLabels(
        label_dict=labels,
        label_transformers={},
    )
