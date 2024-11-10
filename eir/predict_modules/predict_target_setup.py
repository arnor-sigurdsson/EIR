from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Set

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from eir.data_load import label_setup
from eir.data_load.data_source_modules.deeplake_ops import is_deeplake_dataset
from eir.data_load.label_setup import (
    TabularFileInfo,
    al_all_column_ops,
    al_label_dict,
    al_label_transformers,
    al_target_label_dict,
    transform_label_df,
)
from eir.experiment_io.label_transformer_io import load_transformers
from eir.predict_modules.predict_tabular_input_setup import prep_missing_con_dict
from eir.setup.config import Configs
from eir.setup.schemas import (
    ArrayOutputTypeConfig,
    ImageOutputTypeConfig,
    OutputConfig,
    SequenceOutputTypeConfig,
    SurvivalOutputTypeConfig,
    TabularOutputTypeConfig,
)
from eir.target_setup.target_label_setup import (
    MissingTargetsInfo,
    compute_missing_ids_per_tabular_output,
    convert_dtypes,
    gather_data_pointers_from_data_source,
    gather_torch_nan_missing_ids,
    get_all_output_and_target_names,
    get_missing_targets_info,
    get_tabular_target_file_infos,
    synchronize_missing_survival_values,
    transform_durations_with_nans,
    update_labels_dict,
)
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class PredictTargetLabels:
    predict_labels: pd.DataFrame
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return self.predict_labels


@dataclass
class MergedPredictTargetLabels:
    label_dict: al_target_label_dict
    label_transformers: dict[str, dict[str, al_label_transformers]]
    missing_ids_per_output: MissingTargetsInfo

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
        predict_labels=labels_dict,
        label_transformers=label_transformers,
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
) -> pd.DataFrame:
    all_con_transformers = _extract_target_con_transformers(
        label_transformers=label_transformers,
        con_columns=con_columns,
    )

    con_standardscaler_transformers = {
        key: value
        for key, value in all_con_transformers.items()
        if isinstance(value, StandardScaler)
    }

    train_con_column_means = prep_missing_con_dict(
        con_transformers=con_standardscaler_transformers
    )

    df_labels_test = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=cat_columns,
        con_label_columns=con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
        impute_missing=False,
    )

    assert len(label_transformers) > 0
    df_labels_test = transform_label_df(
        df_labels=df_labels_test,
        label_transformers=label_transformers,
        impute_missing=False,
    )

    return df_labels_test


def _extract_target_con_transformers(
    label_transformers: al_label_transformers,
    con_columns: Sequence[str],
) -> Dict[str, StandardScaler | KBinsDiscretizer | IdentityTransformer]:
    con_transformers = {}

    for target_column, transformer_object in label_transformers.items():
        if target_column not in con_columns:
            continue

        assert target_column not in con_transformers
        assert isinstance(
            transformer_object, (StandardScaler, KBinsDiscretizer, IdentityTransformer)
        )

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

    all_ids: set[str] = set(ids)
    per_modality_missing_ids: dict[str, set[str]] = {}
    within_modality_missing_ids: dict[str, dict[str, set[str]]] = {}
    tabular_label_transformers: dict[str, Any] = {}
    dtypes: dict[str, dict[str, Any]] = {}

    test_labels_dict: dict[str, dict[str, dict[str, Any]]] = {}

    tabular_target_files_info = get_tabular_target_file_infos(
        output_configs=pc.output_configs
    )

    for output_config in output_configs:
        output_source = output_config.output_info.output_source
        output_type_info = output_config.output_type_info
        output_name = output_config.output_info.output_name
        logger.info(f"Setting up target labels for {output_name}.")

        match output_type_info:
            case TabularOutputTypeConfig():
                process_tabular_output_for_testing(
                    output_name=output_name,
                    tabular_target_files_info=tabular_target_files_info,
                    output_folder=pc.global_config.basic_experiment.output_folder,
                    custom_column_label_parsing_ops=custom_column_label_parsing_ops,
                    ids=ids,
                    all_ids=all_ids,
                    tabular_label_transformers=tabular_label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    within_modality_missing_ids=within_modality_missing_ids,
                    test_labels_dict=test_labels_dict,
                    dtypes=dtypes,
                )

            case SequenceOutputTypeConfig():
                process_sequence_output_for_testing(
                    output_name=output_name,
                    ids=ids,
                    per_modality_missing_ids=per_modality_missing_ids,
                    test_labels_dict=test_labels_dict,
                    dtypes=dtypes,
                )

            case ArrayOutputTypeConfig() | ImageOutputTypeConfig():
                process_array_or_image_output_for_testing(
                    output_name=output_name,
                    ids=ids,
                    output_config=output_config,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    test_labels_dict=test_labels_dict,
                    dtypes=dtypes,
                )

            case SurvivalOutputTypeConfig():
                process_survival_output_for_testing(
                    output_name=output_name,
                    tabular_target_files_info=tabular_target_files_info,
                    output_folder=pc.global_config.basic_experiment.output_folder,
                    custom_column_label_parsing_ops=custom_column_label_parsing_ops,
                    ids=ids,
                    all_ids=all_ids,
                    tabular_label_transformers=tabular_label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    within_modality_missing_ids=within_modality_missing_ids,
                    test_labels_dict=test_labels_dict,
                    dtypes=dtypes,
                )
            case _:
                raise NotImplementedError(
                    f"Output type {output_type_info} not implemented"
                )

    outputs_and_targets = get_all_output_and_target_names(output_configs=output_configs)
    missing_target_info = get_missing_targets_info(
        missing_ids_per_modality=per_modality_missing_ids,
        missing_ids_within_modality=within_modality_missing_ids,
        output_and_target_names=outputs_and_targets,
        output_configs=output_configs,
    )

    test_labels_object = MergedPredictTargetLabels(
        label_dict=test_labels_dict,
        label_transformers=tabular_label_transformers,
        missing_ids_per_output=missing_target_info,
    )

    return test_labels_object


def process_tabular_output_for_testing(
    output_name: str,
    tabular_target_files_info: Dict[str, Any],
    output_folder: str,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
    all_ids: Set[str],
    tabular_label_transformers: Dict[str, Any],
    per_modality_missing_ids: Dict[str, Set[str]],
    within_modality_missing_ids: Dict[str, Dict[str, Set[str]]],
    test_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
):
    cur_tabular_info = tabular_target_files_info[output_name]
    cur_labels = get_tabular_target_labels_for_predict(
        output_folder=output_folder,
        tabular_info=cur_tabular_info,
        output_name=output_name,
        custom_column_label_parsing_ops=custom_column_label_parsing_ops,
        ids=ids,
    )
    tabular_label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.index)
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    missing_ids_per_target_column = compute_missing_ids_per_tabular_output(
        all_labels_df=all_labels,
        tabular_info=cur_tabular_info,
        output_name=output_name,
    )
    within_modality_missing_ids.update(missing_ids_per_target_column)

    update_labels_dict(
        labels_dict=test_labels_dict,
        labels_df=cur_labels.predict_labels,
        output_name=output_name,
    )

    if output_name not in dtypes:
        dtypes[output_name] = cur_labels.predict_labels.dtypes.to_dict()


def process_sequence_output_for_testing(
    output_name: str,
    ids: Sequence[str],
    per_modality_missing_ids: Dict[str, Set[str]],
    test_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
):
    cur_labels = set_up_delayed_predict_target_labels(
        ids=ids,
        output_name=output_name,
    )

    cur_missing_ids = gather_torch_nan_missing_ids(
        labels=cur_labels.all_labels,
        output_name=output_name,
    )

    per_modality_missing_ids[output_name] = cur_missing_ids

    update_labels_dict(
        labels_dict=test_labels_dict,
        labels_df=cur_labels.predict_labels,
        output_name=output_name,
    )

    if output_name not in dtypes:
        dtypes[output_name] = cur_labels.predict_labels.dtypes.to_dict()


def process_array_or_image_output_for_testing(
    output_name: str,
    ids: Sequence[str],
    output_config: Any,
    output_source: str,
    per_modality_missing_ids: Dict[str, Set[str]],
    test_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
):
    cur_labels = _set_up_predict_file_target_labels(
        ids=ids,
        output_config=output_config,
    )

    cur_missing_ids = gather_torch_nan_missing_ids(
        labels=cur_labels.all_labels,
        output_name=output_name,
    )
    per_modality_missing_ids[output_name] = cur_missing_ids

    is_deeplake = is_deeplake_dataset(data_source=output_source)
    if is_deeplake:
        dtypes[output_name] = {output_name: np.dtype("int64")}
    else:
        dtypes[output_name] = {output_name: np.dtype("O")}

    dtypes = convert_dtypes(dtypes=dtypes)

    update_labels_dict(
        labels_dict=test_labels_dict,
        labels_df=cur_labels.predict_labels,
        output_name=output_name,
        dtypes=dtypes,
    )


def process_survival_output_for_testing(
    output_name: str,
    tabular_target_files_info: Dict[str, Any],
    output_folder: str,
    custom_column_label_parsing_ops: al_all_column_ops,
    ids: Sequence[str],
    all_ids: Set[str],
    tabular_label_transformers: Dict[str, Any],
    per_modality_missing_ids: Dict[str, Set[str]],
    within_modality_missing_ids: Dict[str, Dict[str, Set[str]]],
    test_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
) -> None:
    """
    Process survival output data for testing phase.

    Note: Unlike training, we only transform the time values and don't process
    event indicators, as they're not needed for prediction.
    """
    cur_tabular_info = tabular_target_files_info[output_name]

    cur_labels = get_tabular_target_labels_for_predict(
        output_folder=output_folder,
        tabular_info=cur_tabular_info,
        output_name=output_name,
        custom_column_label_parsing_ops=custom_column_label_parsing_ops,
        ids=ids,
    )

    msg_con = "Expected exactly one continuous column for survival data"
    assert len(cur_tabular_info.con_columns) == 1, msg_con
    msg_cat = "Expected exactly one categorical column for survival data"
    assert len(cur_tabular_info.cat_columns) == 1, msg_cat

    time_column = cur_tabular_info.con_columns[0]
    event_column = cur_tabular_info.cat_columns[0]

    df_time = pd.read_csv(
        cur_tabular_info.file_path,
        usecols=["ID", time_column],
        index_col="ID",
        dtype={"ID": str},
    )
    df_time_test = df_time.loc[df_time.index.isin(ids)]

    df_time_test, cur_labels.predict_labels = synchronize_missing_survival_values(
        df_time=df_time_test,
        df_labels=cur_labels.predict_labels,
        time_column=time_column,
        event_column=event_column,
    )

    dur_transformer = cur_labels.label_transformers[output_name][time_column]
    cur_labels.predict_labels[time_column] = transform_durations_with_nans(
        df=df_time_test,
        time_column=time_column,
        transformer=dur_transformer,
    )

    tabular_label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.index)
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    missing_ids_per_target_column = compute_missing_ids_per_tabular_output(
        all_labels_df=all_labels,
        tabular_info=cur_tabular_info,
        output_name=output_name,
    )
    within_modality_missing_ids.update(missing_ids_per_target_column)

    update_labels_dict(
        labels_dict=test_labels_dict,
        labels_df=cur_labels.predict_labels,
        output_name=output_name,
    )

    if output_name not in dtypes:
        dtypes[output_name] = cur_labels.predict_labels.dtypes.to_dict()


def _set_up_predict_file_target_labels(
    ids: Sequence[str],
    output_config: OutputConfig,
) -> PredictTargetLabels:
    output_name = output_config.output_info.output_name
    output_source = output_config.output_info.output_source
    output_inner_key = output_config.output_info.output_inner_key

    id_to_data_pointer_mapping = gather_data_pointers_from_data_source(
        data_source=Path(output_source),
        validate=True,
        output_inner_key=output_inner_key,
    )

    ids_set = set(ids)
    labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping[id_]} for id_ in ids_set
    }

    df_labels = pd.DataFrame.from_dict(labels, orient="index")
    df_labels["Output Name"] = output_name

    return PredictTargetLabels(
        predict_labels=df_labels,
        label_transformers={},
    )


def set_up_delayed_predict_target_labels(
    ids: Sequence[str],
    output_name: str,
) -> PredictTargetLabels:
    ids_set = set(ids)
    labels: al_label_dict = {id_: {output_name: torch.nan} for id_ in ids_set}

    df_labels = pd.DataFrame.from_dict(labels, orient="index")
    df_labels["Output Name"] = output_name

    return PredictTargetLabels(
        predict_labels=df_labels,
        label_transformers={},
    )
