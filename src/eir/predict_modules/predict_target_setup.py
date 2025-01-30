from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from eir.data_load import label_setup
from eir.data_load.data_source_modules.deeplake_ops import is_deeplake_dataset
from eir.data_load.label_setup import (
    TabularFileInfo,
    al_label_transformers,
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
    gather_data_pointers_from_data_source,
    gather_torch_null_missing_ids,
    get_missing_targets_info,
    get_tabular_target_file_infos,
    synchronize_missing_survival_values,
    transform_durations_with_nans,
    update_labels_df,
)
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass
class PredictTargetLabels:
    predict_labels: pl.DataFrame
    label_transformers: dict[str, al_label_transformers]

    @property
    def all_labels(self) -> pl.DataFrame:
        return self.predict_labels


@dataclass
class MergedPredictTargetLabels:
    predict_labels: pl.DataFrame
    label_transformers: dict[str, dict[str, al_label_transformers]]
    missing_ids_per_output: MissingTargetsInfo

    @property
    def all_labels(self) -> pl.DataFrame:
        return self.predict_labels


def get_tabular_target_labels_for_predict(
    output_folder: str,
    tabular_info: TabularFileInfo,
    output_name: str,
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
    )

    label_transformers = load_transformers(
        output_folder=output_folder,
        transformers_to_load={output_name: all_columns},
    )

    labels_df = parse_labels_for_predict(
        con_columns=con_columns,
        cat_columns=cat_columns,
        df_labels_test=df_labels_test,
        label_transformers=label_transformers[output_name],
    )

    labels = PredictTargetLabels(
        predict_labels=labels_df,
        label_transformers=label_transformers,
    )

    return labels


def _load_labels_for_predict(
    tabular_info: TabularFileInfo,
    ids_to_keep: Sequence[str],
) -> pl.DataFrame:
    parse_wrapper = label_setup.get_label_parsing_wrapper(
        label_parsing_chunk_size=tabular_info.parsing_chunk_size
    )
    df_labels_test = parse_wrapper(
        label_file_tabular_info=tabular_info,
        ids_to_keep=ids_to_keep,
    )

    return df_labels_test


def parse_labels_for_predict(
    con_columns: Sequence[str],
    cat_columns: Sequence[str],
    df_labels_test: pl.DataFrame,
    label_transformers: al_label_transformers,
) -> pl.DataFrame:
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

    assert len(label_transformers) > 0, "No label transformers found"

    df_labels_test = transform_label_df(
        df_labels=df_labels_test,
        label_transformers=label_transformers,
        impute_missing=False,
    )

    return df_labels_test


def _extract_target_con_transformers(
    label_transformers: al_label_transformers,
    con_columns: Sequence[str],
) -> dict[str, StandardScaler | KBinsDiscretizer | IdentityTransformer]:
    con_transformers = {}

    for target_column, transformer_object in label_transformers.items():
        if target_column not in con_columns:
            continue

        assert target_column not in con_transformers
        assert isinstance(
            transformer_object, StandardScaler | KBinsDiscretizer | IdentityTransformer
        )

        con_transformers[target_column] = transformer_object

    assert len(con_transformers) == len(con_columns)

    return con_transformers


def get_target_labels_for_testing(
    configs_overloaded_for_predict: Configs,
    ids: Sequence[str],
) -> MergedPredictTargetLabels:
    pc = configs_overloaded_for_predict
    output_configs = pc.output_configs

    all_ids: set[str] = set(ids)
    per_modality_missing_ids: dict[str, set[str]] = {}
    label_transformers: dict[str, Any] = {}

    test_labels_df = pl.DataFrame(schema={"ID": pl.Utf8})

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
                test_labels_df = process_tabular_output_for_testing(
                    output_name=output_name,
                    tabular_target_files_info=tabular_target_files_info,
                    output_folder=pc.global_config.basic_experiment.output_folder,
                    ids=ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    test_labels_df=test_labels_df,
                )
            case SequenceOutputTypeConfig():
                test_labels_df = process_sequence_output_for_testing(
                    output_name=output_name,
                    ids=ids,
                    per_modality_missing_ids=per_modality_missing_ids,
                    test_labels_df=test_labels_df,
                )

            case ArrayOutputTypeConfig() | ImageOutputTypeConfig():
                test_labels_df = process_array_or_image_output_for_testing(
                    output_name=output_name,
                    ids=ids,
                    output_config=output_config,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    test_labels_df=test_labels_df,
                )

            case SurvivalOutputTypeConfig():
                test_labels_df = process_survival_output_for_testing(
                    output_name=output_name,
                    tabular_target_files_info=tabular_target_files_info,
                    output_folder=pc.global_config.basic_experiment.output_folder,
                    ids=ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    test_labels_df=test_labels_df,
                    per_modality_missing_ids=per_modality_missing_ids,
                )
            case _:
                raise NotImplementedError(
                    f"Output type {output_type_info} not implemented"
                )

    missing_target_info = get_missing_targets_info(
        missing_ids_per_modality=per_modality_missing_ids,
    )

    test_labels_object = MergedPredictTargetLabels(
        predict_labels=test_labels_df,
        label_transformers=label_transformers,
        missing_ids_per_output=missing_target_info,
    )

    return test_labels_object


def process_tabular_output_for_testing(
    output_name: str,
    tabular_target_files_info: dict[str, Any],
    output_folder: str,
    ids: Sequence[str],
    all_ids: set[str],
    label_transformers: dict[str, Any],
    per_modality_missing_ids: dict[str, set[str]],
    test_labels_df: pl.DataFrame,
) -> pl.DataFrame:
    cur_tabular_info = tabular_target_files_info[output_name]
    cur_labels = get_tabular_target_labels_for_predict(
        output_folder=output_folder,
        tabular_info=cur_tabular_info,
        output_name=output_name,
        ids=ids,
    )
    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.get_column("ID").to_list())
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    test_labels_df = update_labels_df(
        master_df=test_labels_df,
        new_labels=cur_labels.predict_labels,
        output_name=output_name,
    )

    return test_labels_df


def process_sequence_output_for_testing(
    output_name: str,
    ids: Sequence[str],
    per_modality_missing_ids: dict[str, set[str]],
    test_labels_df: pl.DataFrame,
) -> pl.DataFrame:
    cur_labels = set_up_delayed_predict_target_labels(
        ids=ids,
        output_name=output_name,
    )

    cur_missing_ids = gather_torch_null_missing_ids(
        labels=cur_labels.all_labels,
        output_name=output_name,
    )

    per_modality_missing_ids[output_name] = cur_missing_ids

    test_labels_df = update_labels_df(
        master_df=test_labels_df,
        new_labels=cur_labels.predict_labels,
        output_name=output_name,
    )

    return test_labels_df


def process_array_or_image_output_for_testing(
    output_name: str,
    ids: Sequence[str],
    output_config: Any,
    output_source: str,
    per_modality_missing_ids: dict[str, set[str]],
    test_labels_df: pl.DataFrame,
) -> pl.DataFrame:
    cur_labels = _set_up_predict_file_target_labels(
        ids=ids,
        output_config=output_config,
    )

    cur_missing_ids = gather_torch_null_missing_ids(
        labels=cur_labels.all_labels,
        output_name=output_name,
    )
    per_modality_missing_ids[output_name] = cur_missing_ids

    is_deeplake = is_deeplake_dataset(data_source=output_source)
    col_name = f"{output_name}__{output_name}"

    polars_dtype: type[pl.Int64] | type[pl.Utf8]
    polars_dtype = pl.Int64 if is_deeplake else pl.Utf8

    test_labels_df = update_labels_df(
        master_df=test_labels_df,
        new_labels=cur_labels.predict_labels,
        output_name=output_name,
    ).with_columns([pl.col(col_name).cast(polars_dtype)])

    return test_labels_df


def process_survival_output_for_testing(
    output_name: str,
    tabular_target_files_info: dict[str, Any],
    output_folder: str,
    ids: Sequence[str],
    all_ids: set[str],
    label_transformers: dict[str, Any],
    per_modality_missing_ids: dict[str, set[str]],
    test_labels_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Process survival output data for testing phase using Polars.

    Note: Unlike training, we only transform the time values and don't process
    event indicators, as they're not needed for prediction.
    """
    cur_tabular_info = tabular_target_files_info[output_name]

    cur_labels = get_tabular_target_labels_for_predict(
        output_folder=output_folder,
        tabular_info=cur_tabular_info,
        output_name=output_name,
        ids=ids,
    )

    msg_con = "Expected exactly one continuous column for survival data"
    assert len(cur_tabular_info.con_columns) == 1, msg_con
    msg_cat = "Expected exactly one categorical column for survival data"
    assert len(cur_tabular_info.cat_columns) == 1, msg_cat

    time_column = cur_tabular_info.con_columns[0]
    event_column = cur_tabular_info.cat_columns[0]

    df_time = pl.read_csv(
        cur_tabular_info.file_path,
        columns=["ID", time_column],
    ).with_columns([pl.col("ID").cast(pl.Utf8), pl.col(time_column).cast(pl.Float32)])

    df_time_test = df_time.filter(pl.col("ID").is_in(ids))

    df_time_test, cur_labels.predict_labels = synchronize_missing_survival_values(
        df_time=df_time_test,
        df_labels=cur_labels.predict_labels,
        time_column=time_column,
        event_column=event_column,
    )

    dur_transformer = cur_labels.label_transformers[output_name][time_column]

    cur_labels.predict_labels = cur_labels.predict_labels.with_columns(
        [
            transform_durations_with_nans(
                df=df_time_test,
                time_column=time_column,
                transformer=dur_transformer,
            ).alias(time_column)
        ]
    )

    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.get_column("ID").to_list())
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    test_labels_df = update_labels_df(
        master_df=test_labels_df,
        new_labels=cur_labels.predict_labels,
        output_name=output_name,
    )

    return test_labels_df


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

    values = [id_to_data_pointer_mapping.get(id_, None) for id_ in ids]

    df_labels = pl.DataFrame(
        {
            "ID": list(ids),
            f"{output_name}": values,
        }
    )

    df_labels = df_labels.with_columns([pl.col(f"{output_name}").cast(pl.Utf8)])

    return PredictTargetLabels(
        predict_labels=df_labels,
        label_transformers={},
    )


def set_up_delayed_predict_target_labels(
    ids: Sequence[str],
    output_name: str,
) -> PredictTargetLabels:
    df_labels = pl.DataFrame(
        {
            "ID": list(ids),
            f"{output_name}": ["DELAYED"] * len(ids),
        },
        schema_overrides={f"{output_name}": pl.Categorical},
    )

    return PredictTargetLabels(
        predict_labels=df_labels,
        label_transformers={},
    )
