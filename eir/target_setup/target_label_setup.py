import math
import reprlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from tqdm import tqdm

from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
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
    save_transformer_set,
    set_up_train_and_valid_tabular_data,
)
from eir.setup import schemas
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.predict_modules.predict_target_setup import PredictTargetLabels
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
class MissingTargetsInfo:
    missing_ids_per_modality: dict[str, set[str]]
    missing_ids_within_modality: dict[str, dict[str, set[str]]]


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
            logger.info(f"Modality '{modality}' has no missing target IDs.")
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
    df_labels_train = pd.DataFrame(index=list(train_ids))
    df_labels_valid = pd.DataFrame(index=list(valid_ids))
    label_transformers = {}

    all_ids: set[str] = set(train_ids).union(set(valid_ids))
    per_modality_missing_ids: dict[str, set[str]] = {}
    within_modality_missing_ids: dict[str, dict[str, set[str]]] = {}

    tabular_target_labels_info = get_tabular_target_file_infos(
        output_configs=output_configs
    )

    for output_config in output_configs:
        output_name = output_config.output_info.output_name
        output_type = output_config.output_info.output_type

        match output_type:
            case "tabular":
                tabular_info = tabular_target_labels_info[output_name]
                cur_labels = set_up_train_and_valid_tabular_data(
                    tabular_file_info=tabular_info,
                    custom_label_ops=custom_label_ops,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                )
                cur_transformers = cur_labels.label_transformers
                label_transformers[output_name] = cur_transformers

                per_modality_missing_ids[output_name] = all_ids.difference(
                    set(cur_labels.all_labels.keys())
                )

                missing_ids_per_target_column = compute_missing_ids_per_tabular_output(
                    tabular_labels=cur_labels,
                    tabular_info=tabular_info,
                    output_name=output_name,
                )
                within_modality_missing_ids = {
                    **within_modality_missing_ids,
                    **missing_ids_per_target_column,
                }

            case "sequence":
                cur_labels = set_up_delayed_target_labels(
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_name=output_name,
                )

                missing_sequence_ids = _find_sequence_output_missing_ids(
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_source=output_config.output_info.output_source,
                )

                per_modality_missing_ids[output_name] = missing_sequence_ids

            case "array":
                cur_labels = set_up_file_target_labels(
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_config=output_config,
                )

                cur_missing_ids = gather_torch_nan_missing_ids(
                    labels=cur_labels.all_labels,
                    output_name=output_name,
                )
                per_modality_missing_ids[output_name] = cur_missing_ids

            case _:
                raise ValueError(f"Unknown output type: '{output_type}'.")

        df_train_cur = pd.DataFrame.from_dict(cur_labels.train_labels, orient="index")
        df_valid_cur = pd.DataFrame.from_dict(cur_labels.valid_labels, orient="index")

        df_train_cur["Output Name"] = output_name
        df_valid_cur["Output Name"] = output_name

        df_labels_train = pd.concat((df_labels_train, df_train_cur))
        df_labels_valid = pd.concat((df_labels_valid, df_valid_cur))

    df_labels_train = df_labels_train.set_index("Output Name", append=True)
    df_labels_valid = df_labels_valid.set_index("Output Name", append=True)

    df_labels_train = df_labels_train.dropna(how="all")
    df_labels_valid = df_labels_valid.dropna(how="all")

    train_labels_dict = df_to_nested_dict(df=df_labels_train)
    valid_labels_dict = df_to_nested_dict(df=df_labels_valid)

    missing_target_info = MissingTargetsInfo(
        missing_ids_per_modality=per_modality_missing_ids,
        missing_ids_within_modality=within_modality_missing_ids,
    )
    log_missing_targets_info(missing_targets_info=missing_target_info, all_ids=all_ids)

    labels_data_object = MergedTargetLabels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
        missing_ids_per_output=missing_target_info,
    )

    return labels_data_object


def compute_missing_ids_per_tabular_output(
    tabular_labels: Union[Labels, "PredictTargetLabels"],
    tabular_info: TabularFileInfo,
    output_name: str,
) -> dict[str, dict[str, set[str]]]:
    missing_per_target_column: dict[str, dict[str, set[str]]] = {output_name: {}}

    all_ids = set(tabular_labels.all_labels.keys())

    all_columns = list(tabular_info.con_columns) + list(tabular_info.cat_columns)

    for target_column in all_columns:
        cur_missing = set()

        for id_ in all_ids:
            cur_label = tabular_labels.all_labels[id_][target_column]
            if isinstance(cur_label, float) and math.isnan(cur_label):
                cur_missing.add(id_)

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

    return Labels(
        train_labels=train_labels,
        valid_labels=valid_labels,
        label_transformers={},
    )


def _find_sequence_output_missing_ids(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_source: str,
) -> set[str]:
    if is_deeplake_dataset(data_source=output_source):
        deeplake_ds = load_deeplake_dataset(data_source=output_source)
        seq_ids = set(str(i) for i in deeplake_ds["ID"].numpy())
    else:
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

    id_to_data_pointer_mapping = gather_data_pointers_from_data_source(
        data_source=Path(output_source),
        validate=True,
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

    return Labels(
        train_labels=train_labels,
        valid_labels=valid_labels,
        label_transformers={},
    )


def gather_torch_nan_missing_ids(labels: al_label_dict, output_name: str) -> set[str]:
    missing_ids = set()
    for id_, label in labels.items():
        cur_label = label[output_name]
        if isinstance(cur_label, float) and math.isnan(cur_label):
            missing_ids.add(id_)

    return missing_ids


def gather_data_pointers_from_data_source(
    data_source: Path,
    validate: bool = True,
) -> dict[str, Path | int]:
    """
    Disk: ID -> file path
    Deeplake: ID -> integer index
    """
    if is_deeplake_dataset(data_source=str(data_source)):
        deeplake_ds = load_deeplake_dataset(data_source=str(data_source))
        iterator = (sample.numpy().item() for sample in deeplake_ds["ID"])
        iterator = ((str(i), index) for i, index in enumerate(iterator))
    else:
        iterator = get_file_path_iterator(data_source=data_source, validate=validate)
        iterator = ((f.stem, f) for f in iterator)

    logger.debug("Gathering data pointers from %s.", data_source)
    id_to_pointer_mapping = {}
    for id_, pointer in tqdm(iterator, desc="Progress"):
        if id_ in id_to_pointer_mapping:
            raise ValueError(f"Duplicate ID: {id_}")

        id_to_pointer_mapping[id_] = pointer

    return id_to_pointer_mapping


def df_to_nested_dict(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, Union[float, int, Path]]]]:
    """
    Convert a DataFrame with a 2-level multi index ['ID', 'Output Name']
    to a nested dict.

    The resulting structure is

    {'ID': {output_name: {inner_output_identifier: inner_output_value}}}.

    For each output_name, only relevant columns are included.
    """
    parsed_dict: Dict[str, Dict[str, Dict[str, Union[float, int, Path]]]] = {}

    for (cur_id, cur_output_name), group_df in df.groupby(level=[0, 1]):
        if cur_id not in parsed_dict:
            parsed_dict[cur_id] = {}

        filtered_dict = group_df.dropna(axis=1, how="all").to_dict(orient="records")[0]

        parsed_dict[cur_id][cur_output_name] = filtered_dict

    return parsed_dict


def gather_all_ids_from_output_configs(
    output_configs: Sequence[schemas.OutputConfig],
) -> Tuple[str, ...]:
    all_ids: set[str] = set()
    for config in output_configs:
        cur_source = Path(config.output_info.output_source)
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
        if output_config.output_info.output_type != "tabular":
            continue

        output_name = output_config.output_info.output_name
        output_type_info = output_config.output_type_info
        assert isinstance(output_type_info, TabularOutputTypeConfig)

        tabular_info = TabularFileInfo(
            file_path=Path(output_config.output_info.output_source),
            con_columns=output_type_info.target_con_columns,
            cat_columns=output_type_info.target_cat_columns,
            parsing_chunk_size=output_type_info.label_parsing_chunk_size,
        )
        tabular_files_info[output_name] = tabular_info

    return tabular_files_info
