import math
import reprlib
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    is_deeplake_sample_missing,
    load_deeplake_dataset,
)
from eir.data_load.label_setup import (
    Labels,
    TabularFileInfo,
    al_all_column_ops,
    al_label_dict,
    al_label_transformers,
    al_target_label_dict,
    gather_ids_from_data_source,
    gather_ids_from_tabular_file,
    get_file_path_iterator,
    set_up_train_and_valid_tabular_data,
)
from eir.experiment_io.label_transformer_io import save_transformer_set
from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_array import ArrayOutputTypeConfig
from eir.setup.schema_modules.output_schemas_sequence import SequenceOutputTypeConfig
from eir.setup.schema_modules.output_schemas_survival import SurvivalOutputTypeConfig
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.target_setup.target_setup_utils import IdentityTransformer
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.train import Hooks


logger = get_logger(name=__name__)


def set_up_all_targets_wrapper(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    run_folder: Path,
    output_configs: Sequence[schemas.OutputConfig],
    hooks: Optional["Hooks"],
) -> "MergedTargetLabels":
    logger.info("Setting up target labels.")

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    target_labels = set_up_all_target_labels_wrapper(
        output_configs=output_configs,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    save_transformer_set(
        transformers_per_source=target_labels.label_transformers,
        run_folder=run_folder,
    )

    return target_labels


@dataclass
class LinkedTargets:
    """
    This is specifically to track which targets should be treated as linked w.r.t.
    to missingness, e.g. in survival analysis, the event column is linked to the time
    column. While one could implicitly set this up my forcing them to have the
    same NaN values, this allows us to track this explicitly, for example when
    we filter missing outputs and target labels, we can check for this and properly
    filter the time column even though it's not an output from the model.
    """

    target_name: str
    linked_target_name: str


@dataclass
class MissingTargetsInfo:
    missing_ids_per_modality: dict[str, set[str]]
    missing_ids_within_modality: dict[str, dict[str, set[str]]]
    precomputed_missing_ids: dict[str, dict[str, set[str]]]
    linked_targets: dict[str, LinkedTargets]


def get_missing_targets_info(
    missing_ids_per_modality: dict[str, set[str]],
    missing_ids_within_modality: dict[str, dict[str, set[str]]],
    output_and_target_names: dict[str, list[str]],
    output_configs: Sequence[schemas.OutputConfig],
) -> MissingTargetsInfo:

    precomputed_missing_ids = _precompute_missing_ids(
        missing_ids_per_modality=missing_ids_per_modality,
        missing_ids_within_modality=missing_ids_within_modality,
        output_and_target_names=output_and_target_names,
    )

    linked_survival_targets = build_linked_survival_targets_for_missing_ids(
        output_configs=output_configs
    )

    return MissingTargetsInfo(
        missing_ids_per_modality=missing_ids_per_modality,
        missing_ids_within_modality=missing_ids_within_modality,
        precomputed_missing_ids=precomputed_missing_ids,
        linked_targets=linked_survival_targets,
    )


def build_linked_survival_targets_for_missing_ids(
    output_configs: Sequence[schemas.OutputConfig],
) -> dict[str, LinkedTargets]:
    linked_survival_targets: dict[str, LinkedTargets] = {}
    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type
        if output_type == "survival":
            output_type_info = output_config.output_type_info
            assert isinstance(output_type_info, SurvivalOutputTypeConfig)
            event_column = output_type_info.event_column
            time_column = output_type_info.time_column
            linked_survival_targets[output_name] = LinkedTargets(
                target_name=event_column,
                linked_target_name=time_column,
            )

    return linked_survival_targets


def _precompute_missing_ids(
    missing_ids_per_modality: dict[str, set[str]],
    missing_ids_within_modality: dict[str, dict[str, set[str]]],
    output_and_target_names: dict[str, list[str]],
) -> dict[str, dict[str, set[str]]]:
    """
    One could potentially optimize this space-wise by tracking modality-inner_key
    combinations that lead to the same set of missing IDs, then having them point
    to the same object (set of missing IDs) in memory.
    """

    precomputed_missing_ids: dict[str, dict[str, set[str]]] = {}

    for modality, target_names in output_and_target_names.items():
        if modality not in precomputed_missing_ids:
            precomputed_missing_ids[modality] = {}

        ids_this_modality = missing_ids_per_modality.get(modality, set())
        missing_ids_within = missing_ids_within_modality.get(modality, {})

        for target_name in target_names:
            cur_missing_within = missing_ids_within.get(target_name, set())
            combined_ids = ids_this_modality.union(cur_missing_within)

            if combined_ids:
                precomputed_missing_ids[modality][target_name] = combined_ids

    return precomputed_missing_ids


def get_all_output_and_target_names(
    output_configs: Sequence[schemas.OutputConfig],
) -> dict[str, list[str]]:
    output_and_target_names = {}

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        match output_config.output_type_info:
            case TabularOutputTypeConfig(
                target_con_columns=con_columns, target_cat_columns=cat_columns
            ):
                all_columns = list(con_columns) + list(cat_columns)
                output_and_target_names[output_name] = all_columns
            case ArrayOutputTypeConfig() | SequenceOutputTypeConfig():
                output_and_target_names[output_name] = [output_name]
            case SurvivalOutputTypeConfig(
                event_column=event_column, time_column=time_column
            ):
                output_and_target_names[output_name] = [event_column, time_column]

    return output_and_target_names


def log_missing_targets_info(
    missing_targets_info: MissingTargetsInfo, all_ids: set[str]
) -> None:
    repr_formatter = reprlib.Repr()
    repr_formatter.maxset = 10

    logger.info(
        "Checking for missing target information. "
        "These will be ignored during loss and metric computation."
    )

    total_ids_count = len(all_ids)
    missing_within = missing_targets_info.missing_ids_within_modality
    max_columns_to_log = 5

    for modality, missing_ids in missing_targets_info.missing_ids_per_modality.items():
        missing_count = len(missing_ids)

        if missing_count == 0:
            logger.info(f"Output modality '{modality}' has no missing target IDs.")
            continue

        formatted_missing_ids = repr_formatter.repr(missing_ids)
        complete_count = total_ids_count - missing_count
        fraction_complete = (complete_count / total_ids_count) * 100

        logger.info(
            f"Missing target IDs for modality: '{modality}'\n"
            f"  - Missing IDs: {formatted_missing_ids}\n"
            f"  - Stats: Missing: {missing_count}, "
            f"Complete: {complete_count}/{total_ids_count} "
            f"({fraction_complete:.2f}% complete)\n"
        )

        if modality in missing_within:
            columns_logged = 0
            for target_column, ids in missing_within[modality].items():
                missing_within_count = len(ids)

                if missing_within_count == 0:
                    logger.info(
                        f"  - No missing target IDs in modality '{modality}', "
                        f"Column: '{target_column}'."
                    )
                    columns_logged += 1
                    if columns_logged >= max_columns_to_log:
                        break
                    continue

                if columns_logged >= max_columns_to_log:
                    additional_columns = (
                        len(missing_within[modality]) - max_columns_to_log
                    )
                    logger.info(
                        f"  - There are {additional_columns} "
                        f"more columns with missing IDs in modality '{modality}' "
                        f"not displayed."
                    )
                    break

                complete_within_count = total_ids_count - missing_within_count
                fraction_complete_within = (
                    complete_within_count / total_ids_count
                ) * 100

                formatted_ids = repr_formatter.repr(ids)
                logger.info(
                    f"  - Missing target IDs within modality '{modality}', "
                    f"Column: '{target_column}'\n"
                    f"      - Missing IDs: {formatted_ids}\n"
                    f"      - Stats: Missing: {missing_within_count}, "
                    f"Complete: {complete_within_count}/{total_ids_count} "
                    f"({fraction_complete_within:.2f}% complete)\n"
                )
                columns_logged += 1


@dataclass
class MergedTargetLabels:
    train_labels: al_target_label_dict
    valid_labels: al_target_label_dict
    label_transformers: dict[str, al_label_transformers]
    missing_ids_per_output: MissingTargetsInfo

    @property
    def all_labels(self):
        return {**self.train_labels, **self.valid_labels}


def set_up_all_target_labels_wrapper(
    output_configs: Sequence[schemas.OutputConfig],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> MergedTargetLabels:
    """
    We have different ways of setting up the missing labels information depending
    on the output type.

    For tabular, we have a csv file with the labels, so we can just read that in.
    Missing rows in the csv file are not included, and we can directly infer
    the missing IDs from full set of IDs compared to the IDs in the csv file.

    For sequence, the labels are computed on the fly during training, so we
    simply set up a dictionary with the IDs as keys and torch.nan as values.
    TODO:   Fill out the values here with the actual labels depending on available
            files / rows in sequence input .csv.

    For array, we read what files are available on the disk, and flag the missing
    ones as torch.nan.
    """

    all_ids: set[str] = set(train_ids).union(set(valid_ids))
    per_modality_missing_ids: dict[str, set[str]] = {}
    within_modality_missing_ids: dict[str, dict[str, set[str]]] = {}
    label_transformers: dict[str, Any] = {}
    dtypes: dict[str, dict[str, Any]] = {}

    train_labels_dict: dict[str, dict[str, dict[str, Any]]] = {}
    valid_labels_dict: dict[str, dict[str, dict[str, Any]]] = {}

    tabular_target_labels_info = get_tabular_target_file_infos(
        output_configs=output_configs
    )

    for output_config in output_configs:
        output_source = output_config.output_info.output_source
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type
        logger.info(f"Setting up target labels for {output_name}.")

        match output_type:
            case "tabular":
                process_tabular_output(
                    output_name=output_name,
                    tabular_target_labels_info=tabular_target_labels_info,
                    custom_label_ops=custom_label_ops,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    within_modality_missing_ids=within_modality_missing_ids,
                    train_labels_dict=train_labels_dict,
                    valid_labels_dict=valid_labels_dict,
                    dtypes=dtypes,
                )
            case "sequence":
                process_sequence_output(
                    output_name=output_name,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_dict=train_labels_dict,
                    valid_labels_dict=valid_labels_dict,
                )
            case "array" | "image":
                process_array_or_image_output(
                    output_name=output_name,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_config=output_config,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_dict=train_labels_dict,
                    valid_labels_dict=valid_labels_dict,
                    dtypes=dtypes,
                )
            case "survival":
                output_type_info = output_config.output_type_info
                assert isinstance(output_type_info, SurvivalOutputTypeConfig)
                n_bins = output_type_info.num_durations
                process_survival_output(
                    n_bins=n_bins,
                    output_name=output_name,
                    tabular_target_labels_info=tabular_target_labels_info,
                    custom_label_ops=custom_label_ops,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    within_modality_missing_ids=within_modality_missing_ids,
                    train_labels_dict=train_labels_dict,
                    valid_labels_dict=valid_labels_dict,
                    dtypes=dtypes,
                )
            case _:
                raise ValueError(f"Unknown output type: '{output_type}'.")

    output_and_target_names = get_all_output_and_target_names(
        output_configs=output_configs
    )
    missing_target_info = get_missing_targets_info(
        missing_ids_per_modality=per_modality_missing_ids,
        missing_ids_within_modality=within_modality_missing_ids,
        output_and_target_names=output_and_target_names,
        output_configs=output_configs,
    )
    log_missing_targets_info(missing_targets_info=missing_target_info, all_ids=all_ids)

    labels_data_object = MergedTargetLabels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
        missing_ids_per_output=missing_target_info,
    )

    return labels_data_object


def process_tabular_output(
    output_name: str,
    tabular_target_labels_info: Dict[str, Any],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    all_ids: set[str],
    label_transformers: Dict[str, Any],
    per_modality_missing_ids: Dict[str, set[str]],
    within_modality_missing_ids: Dict[str, Dict[str, set[str]]],
    train_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    valid_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
) -> None:
    tabular_info = tabular_target_labels_info[output_name]
    cur_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_info,
        custom_label_ops=custom_label_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.index)
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    logger.debug("Estimating missing IDs for tabular output %s.", output_name)
    missing_ids_per_target_column = compute_missing_ids_per_tabular_output(
        all_labels_df=all_labels,
        tabular_info=tabular_info,
        output_name=output_name,
    )

    within_modality_missing_ids.update(missing_ids_per_target_column)

    update_labels_dict(
        labels_dict=train_labels_dict,
        labels_df=cur_labels.train_labels,
        output_name=output_name,
    )
    update_labels_dict(
        labels_dict=valid_labels_dict,
        labels_df=cur_labels.valid_labels,
        output_name=output_name,
    )

    if output_name not in dtypes:
        dtypes[output_name] = cur_labels.train_labels.dtypes.to_dict()


def process_sequence_output(
    output_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_source: str,
    per_modality_missing_ids: Dict[str, set[str]],
    train_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    valid_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
):
    cur_labels = set_up_delayed_target_labels(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_name=output_name,
    )

    logger.debug("Estimating missing IDs for sequence output %s.", output_name)
    missing_sequence_ids = find_sequence_output_missing_ids(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_source=output_source,
    )

    per_modality_missing_ids[output_name] = missing_sequence_ids

    update_labels_dict(
        labels_dict=train_labels_dict,
        labels_df=cur_labels.train_labels,
        output_name=output_name,
    )
    update_labels_dict(
        labels_dict=valid_labels_dict,
        labels_df=cur_labels.valid_labels,
        output_name=output_name,
    )


def process_array_or_image_output(
    output_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_config: schemas.OutputConfig,
    output_source: str,
    per_modality_missing_ids: Dict[str, set[str]],
    train_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    valid_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
):
    cur_labels = set_up_file_target_labels(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_config=output_config,
    )

    logger.debug("Estimating missing IDs for array output %s.", output_name)
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
        labels_dict=train_labels_dict,
        labels_df=cur_labels.train_labels,
        output_name=output_name,
        dtypes=dtypes,
    )
    update_labels_dict(
        labels_dict=valid_labels_dict,
        labels_df=cur_labels.valid_labels,
        output_name=output_name,
        dtypes=dtypes,
    )


def process_survival_output(
    n_bins: int,
    output_name: str,
    tabular_target_labels_info: Dict[str, Any],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    all_ids: set[str],
    label_transformers: Dict[str, Any],
    per_modality_missing_ids: Dict[str, set[str]],
    within_modality_missing_ids: Dict[str, Dict[str, set[str]]],
    train_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    valid_labels_dict: Dict[str, Dict[str, Dict[str, Any]]],
    dtypes: Dict[str, Dict[str, Any]],
) -> None:
    tabular_info = tabular_target_labels_info[output_name]

    tabular_info_copy = copy(tabular_info)
    tabular_info_copy.con_columns = []

    cur_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_info_copy,
        custom_label_ops=custom_label_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
        do_transform_labels=True,
    )

    assert len(tabular_info.con_columns) == 1
    event_column = tabular_info.cat_columns[0]
    time_column = tabular_info.con_columns[0]

    df_time = pd.read_csv(
        tabular_info.file_path,
        usecols=["ID", time_column],
        index_col="ID",
        dtype={"ID": str},
    )
    df_time_train = df_time.loc[df_time.index.isin(train_ids)]
    df_time_valid = df_time.loc[df_time.index.isin(valid_ids)]

    df_time_train, cur_labels.train_labels = synchronize_missing_survival_values(
        df_time=df_time_train,
        df_labels=cur_labels.train_labels,
        time_column=time_column,
        event_column=event_column,
    )

    df_time_valid, cur_labels.valid_labels = synchronize_missing_survival_values(
        df_time=df_time_valid,
        df_labels=cur_labels.valid_labels,
        time_column=time_column,
        event_column=event_column,
    )

    dur_input = _streamline_duration_transformer_input(
        df_time_train=df_time_train,
        time_column=time_column,
    )
    dur_transformer = fit_duration_transformer(durations=dur_input, n_bins=n_bins)

    cur_labels.train_labels[time_column] = transform_durations_with_nans(
        df=df_time_train,
        time_column=time_column,
        transformer=dur_transformer,
    )

    cur_labels.valid_labels[time_column] = transform_durations_with_nans(
        df=df_time_valid,
        time_column=time_column,
        transformer=dur_transformer,
    )

    cur_labels.label_transformers[time_column] = dur_transformer
    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.index)
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    logger.debug("Estimating missing IDs for survival output %s.", output_name)
    missing_ids_per_target_column = compute_missing_ids_per_tabular_output(
        all_labels_df=all_labels,
        tabular_info=tabular_info,
        output_name=output_name,
    )

    within_modality_missing_ids.update(missing_ids_per_target_column)

    update_labels_dict(
        labels_dict=train_labels_dict,
        labels_df=cur_labels.train_labels,
        output_name=output_name,
    )
    update_labels_dict(
        labels_dict=valid_labels_dict,
        labels_df=cur_labels.valid_labels,
        output_name=output_name,
    )

    if output_name not in dtypes:
        dtypes[output_name] = cur_labels.train_labels.dtypes.to_dict()


def synchronize_missing_survival_values(
    df_time: pd.DataFrame, df_labels: pd.DataFrame, time_column: str, event_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_time = df_time.copy()
    df_labels = df_labels.copy()

    na_ids_time = set(df_time.index[df_time[time_column].isna()])
    na_ids_labels = set(df_labels.index[df_labels[event_column].isna()])

    all_na_ids = list(na_ids_time.union(na_ids_labels))

    df_time.loc[all_na_ids, time_column] = np.nan
    df_labels.loc[all_na_ids, event_column] = np.nan

    return df_time, df_labels


def _streamline_duration_transformer_input(
    df_time_train: pd.DataFrame,
    time_column: str,
) -> np.ndarray:
    train_nan_mask = df_time_train[time_column].isna()

    if train_nan_mask.any():
        values = df_time_train[~train_nan_mask][time_column].to_numpy()
    else:
        values = df_time_train[time_column].to_numpy()

    return values.reshape(-1, 1)


def fit_duration_transformer(
    durations: np.ndarray,
    n_bins: int,
) -> KBinsDiscretizer | IdentityTransformer:

    if not n_bins:
        return IdentityTransformer()

    transformer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform",
    )

    transformer.fit(durations)

    return transformer


def transform_durations_with_nans(
    df: pd.DataFrame,
    time_column: str,
    transformer: KBinsDiscretizer | IdentityTransformer,
) -> np.ndarray:
    nan_mask = df[time_column].isna()

    if not nan_mask.any():
        return transformer.transform(
            df[time_column].to_numpy().reshape(-1, 1)
        ).flatten()

    df_clean = df[~nan_mask]
    transformed = transformer.transform(
        df_clean[time_column].to_numpy().reshape(-1, 1)
    ).flatten()

    result = np.full(len(df), np.nan)
    result[~nan_mask] = transformed
    return result


def discretize_durations(
    df_time_train: pd.DataFrame,
    df_time_valid: pd.DataFrame,
    cur_labels: Labels,
    time_column: str,
    n_bins: int,
) -> None:
    train_nan_mask = df_time_train[time_column].isna()
    dur = (
        df_time_train[~train_nan_mask][time_column].to_numpy().reshape(-1, 1)
        if train_nan_mask.any()
        else df_time_train[time_column].to_numpy().reshape(-1, 1)
    )

    dur_transformer = fit_duration_transformer(durations=dur, n_bins=n_bins)

    cur_labels.train_labels[time_column] = transform_durations_with_nans(
        df=df_time_train,
        time_column=time_column,
        transformer=dur_transformer,
    )

    cur_labels.valid_labels[time_column] = transform_durations_with_nans(
        df=df_time_valid,
        time_column=time_column,
        transformer=dur_transformer,
    )


def update_labels_dict(
    labels_dict: dict[str, dict[str, dict[str, Any]]],
    labels_df: pd.DataFrame,
    output_name: str,
    dtypes: Optional[dict[str, dict[str, type]]] = None,
) -> None:
    """
    We skip storing nan / missing values all together here, as we have a
    data structure tracking missing IDs separately.
    """
    if "Output Name" in labels_df.columns:
        labels_df = labels_df.drop(columns=["Output Name"])

    if dtypes is None:
        dtypes = {}

    cur_dtypes: dict[str, Any] = dtypes.get(output_name, {})

    records = labels_df.to_dict("records")
    for id_value, record in zip(labels_df.index, records):
        parsed_record: dict[str, Any] = {}

        for k, v in record.items():
            if pd.notna(v):
                cur_dtype = cur_dtypes.get(str(k), None)
                if cur_dtype:
                    v = cur_dtype(v)

                parsed_record[str(k)] = v

        if not parsed_record:
            continue

        if id_value not in labels_dict:
            labels_dict[id_value] = {}
        labels_dict[id_value][output_name] = parsed_record


def compute_missing_ids_per_tabular_output(
    all_labels_df: pd.DataFrame,
    tabular_info: TabularFileInfo,
    output_name: str = "output",
) -> Dict[str, Dict[str, set[str]]]:
    missing_per_target_column: Dict[str, Dict[str, set[str]]] = {output_name: {}}
    all_columns = list(tabular_info.con_columns) + list(tabular_info.cat_columns)

    for target_column in all_columns:
        cur_missing = set(all_labels_df.index[all_labels_df[target_column].isna()])
        missing_per_target_column[output_name][target_column] = cur_missing

    return missing_per_target_column


def set_up_delayed_target_labels(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_name: str,
) -> Labels:
    train_ids_set = set(train_ids)
    valid_ids_set = set(valid_ids)
    train_labels: al_label_dict = {
        id_: {output_name: torch.nan} for id_ in train_ids_set
    }
    valid_labels: al_label_dict = {
        id_: {output_name: torch.nan} for id_ in valid_ids_set
    }

    df_train = pd.DataFrame.from_dict(train_labels, orient="index")
    df_train["Output Name"] = output_name
    df_valid = pd.DataFrame.from_dict(valid_labels, orient="index")
    df_valid["Output Name"] = output_name

    return Labels(
        train_labels=df_train,
        valid_labels=df_valid,
        label_transformers={},
    )


def find_sequence_output_missing_ids(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_source: str,
) -> set[str]:

    seq_ids = set(gather_ids_from_data_source(data_source=Path(output_source)))

    train_ids_set = set(train_ids)
    valid_ids_set = set(valid_ids)
    all_ids = train_ids_set.union(valid_ids_set)

    missing_ids = all_ids.difference(seq_ids)

    return missing_ids


def set_up_file_target_labels(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_config: schemas.OutputConfig,
) -> Labels:
    """
    Note we have the .get(id_, torch.nan) here because we want to be able to
    handle if we have partially missing output modalities, e.g. in the case
    of output arrays, but we don't want to just drop those IDs.
    """
    output_name = output_config.output_info.output_name
    output_source = output_config.output_info.output_source
    output_name_inner_key = output_config.output_info.output_inner_key

    id_to_data_pointer_mapping = gather_data_pointers_from_data_source(
        data_source=Path(output_source),
        validate=True,
        output_inner_key=output_name_inner_key,
    )

    train_ids_set = set(train_ids)
    train_labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping.get(id_, torch.nan)}
        for id_ in train_ids_set
    }

    valid_ids_set = set(valid_ids)
    valid_labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping.get(id_, torch.nan)}
        for id_ in valid_ids_set
    }

    df_train = pd.DataFrame.from_dict(train_labels, orient="index")
    df_train["Output Name"] = output_name
    df_valid = pd.DataFrame.from_dict(valid_labels, orient="index")
    df_valid["Output Name"] = output_name

    return Labels(
        train_labels=df_train,
        valid_labels=df_valid,
        label_transformers={},
    )


def gather_torch_nan_missing_ids(labels: pd.DataFrame, output_name: str) -> set[str]:
    missing_ids = set()
    for id_, label in labels.iterrows():
        cur_label = label[output_name]
        if isinstance(cur_label, float) and math.isnan(cur_label):
            missing_ids.add(str(id_))

    return missing_ids


def convert_dtypes(dtypes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    dtype_mapping = {
        "int64": int,
    }

    converted_dtypes = {}
    for output_name, inner_target_name in dtypes.items():
        converted_columns = {}
        for column_name, dtype in inner_target_name.items():
            dtype_name = dtype.name
            if dtype_name in dtype_mapping:
                primitive_type = dtype_mapping[dtype_name]
                converted_columns[column_name] = primitive_type
        converted_dtypes[output_name] = converted_columns

    return converted_dtypes


def gather_data_pointers_from_data_source(
    data_source: Path,
    validate: bool = True,
    output_inner_key: Optional[str] = None,
) -> dict[str, Path | int]:
    """
    Disk: ID -> file path
    Deeplake: ID -> integer index
    """
    iterator: (
        Generator[tuple[str, Path], None, None] | Generator[tuple[str, int], None, None]
    )
    if is_deeplake_dataset(data_source=str(data_source)):
        assert output_inner_key is not None
        iterator = build_deeplake_available_pointer_iterator(
            data_source=data_source,
            inner_key=output_inner_key,
        )
    else:
        iterator_base = get_file_path_iterator(
            data_source=data_source,
            validate=validate,
        )
        iterator = ((f.stem, f) for f in iterator_base)

    logger.debug("Gathering data pointers from %s.", data_source)
    id_to_pointer_mapping = {}
    for id_, pointer in tqdm(iterator, desc="Progress"):
        if id_ in id_to_pointer_mapping:
            raise ValueError(f"Duplicate ID: {id_}")

        id_to_pointer_mapping[id_] = pointer

    return id_to_pointer_mapping


def build_deeplake_available_pointer_iterator(
    data_source: Path, inner_key: str
) -> Generator[tuple[str, int], None, None]:
    deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
    columns = {col.name for col in deeplake_ds.schema.columns}
    existence_col = f"{inner_key}_exists"
    for int_pointer, row in enumerate(deeplake_ds):
        if is_deeplake_sample_missing(
            row=row,
            existence_col=existence_col,
            columns=columns,
        ):
            pass

        id_ = row["ID"]

        yield id_, int(int_pointer)


def gather_all_ids_from_output_configs(
    output_configs: Sequence[schemas.OutputConfig],
) -> Tuple[str, ...]:
    all_ids: set[str] = set()
    for config in output_configs:
        cur_source = Path(config.output_info.output_source)
        logger.debug("Gathering IDs from %s.", cur_source)
        if cur_source.suffix == ".csv":
            cur_ids = gather_ids_from_tabular_file(file_path=cur_source)
        elif cur_source.is_dir():
            cur_ids = gather_ids_from_data_source(data_source=cur_source)
        else:
            raise NotImplementedError(
                "Only csv and directory data sources are supported."
                f" Got: {cur_source}"
            )
        all_ids.update(cur_ids)

    return tuple(all_ids)


def read_manual_ids_if_exist(
    manual_valid_ids_file: Union[None, str]
) -> Union[Sequence[str], None]:
    if not manual_valid_ids_file:
        return None

    with open(manual_valid_ids_file, "r") as infile:
        manual_ids = tuple(line.strip() for line in infile)

    return manual_ids


def get_tabular_target_file_infos(
    output_configs: Iterable[schemas.OutputConfig],
) -> Dict[str, TabularFileInfo]:
    logger.debug("Setting up target labels.")

    tabular_files_info = {}

    for output_config in output_configs:
        output_type = output_config.output_info.output_type
        output_name = output_config.output_info.output_name
        if output_type == "tabular":
            output_type_info = output_config.output_type_info
            assert isinstance(output_type_info, TabularOutputTypeConfig)

            tabular_info = TabularFileInfo(
                file_path=Path(output_config.output_info.output_source),
                con_columns=output_type_info.target_con_columns,
                cat_columns=output_type_info.target_cat_columns,
                parsing_chunk_size=output_type_info.label_parsing_chunk_size,
            )
            tabular_files_info[output_name] = tabular_info
        elif output_type == "survival":
            output_type_info = output_config.output_type_info
            assert isinstance(output_type_info, SurvivalOutputTypeConfig)

            tabular_info = TabularFileInfo(
                file_path=Path(output_config.output_info.output_source),
                cat_columns=[output_type_info.event_column],
                con_columns=[output_type_info.time_column],
                parsing_chunk_size=output_type_info.label_parsing_chunk_size,
            )
            tabular_files_info[output_name] = tabular_info

        else:
            continue

    return tabular_files_info
