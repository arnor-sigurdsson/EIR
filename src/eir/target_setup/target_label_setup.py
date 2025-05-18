import reprlib
from collections.abc import Generator, Iterable, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np
import polars as pl
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
    al_label_transformers,
    gather_ids_from_data_source,
    gather_ids_from_tabular_file,
    get_file_path_iterator,
    set_up_train_and_valid_tabular_data,
)
from eir.experiment_io.label_transformer_io import save_transformer_set
from eir.setup import schemas
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

    target_labels = set_up_all_target_labels_wrapper(
        output_configs=output_configs,
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
    """
    In the case for example when we only have 1 tabular output with multiple
    targets coming from the same .csv and we have an input where we have different
    set of overall IDs (i.e., rows in the .csv) compared to the targets, we
    do not need the specific filtering of IDs later per step as all samples
    w/o inputs or targets have been filtered out in the dataset creation. We
    can track this using the all_have_same_set, which allows us to skip
    checking for the missing IDs completely before loss/metric calculation.
    """

    missing_ids_per_modality: dict[str, set[str]]
    all_have_same_set: bool


def get_missing_targets_info(
    missing_ids_per_modality: dict[str, set[str]],
) -> MissingTargetsInfo:
    if not missing_ids_per_modality:
        return MissingTargetsInfo(
            missing_ids_per_modality=missing_ids_per_modality,
            all_have_same_set=True,
        )

    sets = list(missing_ids_per_modality.values())
    all_same = len(sets) > 1 and all(s == sets[0] for s in sets[1:])

    return MissingTargetsInfo(
        missing_ids_per_modality=missing_ids_per_modality,
        all_have_same_set=all_same,
    )


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


@dataclass
class MergedTargetLabels:
    train_labels: pl.DataFrame
    valid_labels: pl.DataFrame
    label_transformers: dict[str, al_label_transformers]
    missing_ids_per_output: MissingTargetsInfo

    @property
    def all_labels(self) -> pl.DataFrame:
        return pl.concat([self.train_labels, self.valid_labels])


def update_labels_df(
    master_df: pl.DataFrame,
    new_labels: pl.DataFrame,
    output_name: str,
) -> pl.DataFrame:
    if master_df.height == 0:
        master_df = pl.DataFrame(schema={"ID": pl.Utf8})

    rename_exprs = []
    for col in new_labels.columns:
        if col == "ID":
            rename_exprs.append(pl.col("ID"))
        else:
            new_name = f"{output_name}__{col}"
            rename_exprs.append(pl.col(col).alias(new_name))

    new_labels_renamed = new_labels.select(rename_exprs)

    if master_df.columns == ["ID"]:
        return new_labels_renamed
    return master_df.join(new_labels_renamed, on="ID", how="full", coalesce=True)


def set_up_all_target_labels_wrapper(
    output_configs: Sequence[schemas.OutputConfig],
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> MergedTargetLabels:
    all_ids: set[str] = set(train_ids).union(set(valid_ids))
    per_modality_missing_ids: dict[str, set[str]] = {}
    label_transformers: dict[str, Any] = {}

    train_labels_df = pl.DataFrame(schema={"ID": pl.Utf8})
    valid_labels_df = pl.DataFrame(schema={"ID": pl.Utf8})

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
                train_labels_df, valid_labels_df = process_tabular_output(
                    output_name=output_name,
                    tabular_target_labels_info=tabular_target_labels_info,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_df=train_labels_df,
                    valid_labels_df=valid_labels_df,
                )
            case "sequence":
                train_labels_df, valid_labels_df = process_sequence_output(
                    output_name=output_name,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_df=train_labels_df,
                    valid_labels_df=valid_labels_df,
                )
            case "array" | "image":
                train_labels_df, valid_labels_df = process_array_or_image_output(
                    output_name=output_name,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_config=output_config,
                    output_source=output_source,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_df=train_labels_df,
                    valid_labels_df=valid_labels_df,
                )
            case "survival":
                output_type_info = output_config.output_type_info
                assert isinstance(output_type_info, SurvivalOutputTypeConfig)
                n_bins = output_type_info.num_durations
                train_labels_df, valid_labels_df = process_survival_output(
                    n_bins=n_bins,
                    output_name=output_name,
                    tabular_target_labels_info=tabular_target_labels_info,
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    all_ids=all_ids,
                    label_transformers=label_transformers,
                    per_modality_missing_ids=per_modality_missing_ids,
                    train_labels_df=train_labels_df,
                    valid_labels_df=valid_labels_df,
                )
            case _:
                raise ValueError(f"Unknown output type: '{output_type}'.")

    missing_target_info = get_missing_targets_info(
        missing_ids_per_modality=per_modality_missing_ids,
    )
    log_missing_targets_info(missing_targets_info=missing_target_info, all_ids=all_ids)

    return MergedTargetLabels(
        train_labels=train_labels_df,
        valid_labels=valid_labels_df,
        label_transformers=label_transformers,
        missing_ids_per_output=missing_target_info,
    )


def process_tabular_output(
    output_name: str,
    tabular_target_labels_info: dict[str, Any],
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    all_ids: set[str],
    label_transformers: dict[str, Any],
    per_modality_missing_ids: dict[str, set[str]],
    train_labels_df: pl.DataFrame,
    valid_labels_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    tabular_info = tabular_target_labels_info[output_name]
    cur_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_info,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.get_column("ID").to_list())
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    train_labels_df = update_labels_df(
        master_df=train_labels_df,
        new_labels=cur_labels.train_labels,
        output_name=output_name,
    )
    valid_labels_df = update_labels_df(
        master_df=valid_labels_df,
        new_labels=cur_labels.valid_labels,
        output_name=output_name,
    )

    return train_labels_df, valid_labels_df


def process_sequence_output(
    output_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_source: str,
    per_modality_missing_ids: dict[str, set[str]],
    train_labels_df: pl.DataFrame,
    valid_labels_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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

    train_labels_df = update_labels_df(
        master_df=train_labels_df,
        new_labels=cur_labels.train_labels,
        output_name=output_name,
    )
    valid_labels_df = update_labels_df(
        master_df=valid_labels_df,
        new_labels=cur_labels.valid_labels,
        output_name=output_name,
    )

    return train_labels_df, valid_labels_df


def process_array_or_image_output(
    output_name: str,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_config: schemas.OutputConfig,
    output_source: str,
    per_modality_missing_ids: dict[str, set[str]],
    train_labels_df: pl.DataFrame,
    valid_labels_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cur_labels = set_up_file_target_labels(
        train_ids=train_ids,
        valid_ids=valid_ids,
        output_config=output_config,
    )

    logger.debug("Estimating missing IDs for array output %s.", output_name)
    cur_missing_ids = gather_torch_null_missing_ids(
        labels=cur_labels.all_labels,
        output_name=output_name,
    )
    per_modality_missing_ids[output_name] = cur_missing_ids

    is_deeplake = is_deeplake_dataset(data_source=output_source)
    col_name = f"{output_name}__{output_name}"

    polars_dtype: type[pl.Int64] | type[pl.Utf8]
    polars_dtype = pl.Int64 if is_deeplake else pl.Utf8

    train_labels_df = update_labels_df(
        master_df=train_labels_df,
        new_labels=cur_labels.train_labels,
        output_name=output_name,
    )

    train_labels_df = train_labels_df.with_columns(
        [pl.col(col_name).cast(polars_dtype)]
    )

    valid_labels_df = update_labels_df(
        master_df=valid_labels_df,
        new_labels=cur_labels.valid_labels,
        output_name=output_name,
    ).with_columns([pl.col(col_name).cast(polars_dtype)])

    return train_labels_df, valid_labels_df


def process_survival_output(
    n_bins: int,
    output_name: str,
    tabular_target_labels_info: dict[str, Any],
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    all_ids: set[str],
    label_transformers: dict[str, Any],
    per_modality_missing_ids: dict[str, set[str]],
    train_labels_df: pl.DataFrame,
    valid_labels_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    tabular_info = tabular_target_labels_info[output_name]

    tabular_info_copy = copy(tabular_info)
    tabular_info_copy.con_columns = []

    cur_labels = set_up_train_and_valid_tabular_data(
        tabular_file_info=tabular_info_copy,
        train_ids=train_ids,
        valid_ids=valid_ids,
        do_transform_labels=True,
    )

    assert len(tabular_info.con_columns) == 1
    event_column = tabular_info.cat_columns[0]
    time_column = tabular_info.con_columns[0]

    df_time = pl.read_csv(
        tabular_info.file_path,
        columns=["ID", time_column],
    ).with_columns([pl.col("ID").cast(pl.Utf8), pl.col(time_column).cast(pl.Float32)])

    df_time_train = df_time.filter(pl.col("ID").is_in(train_ids))
    df_time_valid = df_time.filter(pl.col("ID").is_in(valid_ids))

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

    cur_labels.train_labels = cur_labels.train_labels.with_columns(
        [
            transform_durations_with_nans(
                df=df_time_train,
                time_column=time_column,
                transformer=dur_transformer,
            ).alias(time_column)
        ]
    )

    cur_labels.valid_labels = cur_labels.valid_labels.with_columns(
        [
            transform_durations_with_nans(
                df=df_time_valid,
                time_column=time_column,
                transformer=dur_transformer,
            ).alias(time_column)
        ]
    )

    cur_labels.label_transformers[time_column] = dur_transformer
    label_transformers[output_name] = cur_labels.label_transformers

    all_labels = cur_labels.all_labels
    cur_ids = set(all_labels.get_column("ID").to_list())
    missing_ids = all_ids.difference(cur_ids)
    per_modality_missing_ids[output_name] = missing_ids

    train_labels_df = update_labels_df(
        master_df=train_labels_df,
        new_labels=cur_labels.train_labels,
        output_name=output_name,
    )
    valid_labels_df = update_labels_df(
        master_df=valid_labels_df,
        new_labels=cur_labels.valid_labels,
        output_name=output_name,
    )

    return train_labels_df, valid_labels_df


def synchronize_missing_survival_values(
    df_time: pl.DataFrame,
    df_labels: pl.DataFrame,
    time_column: str,
    event_column: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    na_ids_time = set(
        df_time.filter(pl.col(time_column).is_null()).get_column("ID").to_list()
    )
    na_ids_labels = set(
        df_labels.filter(pl.col(event_column).is_null()).get_column("ID").to_list()
    )

    all_na_ids = list(na_ids_time.union(na_ids_labels))

    df_time_sync = df_time.with_columns(
        [
            pl.when(pl.col("ID").is_in(all_na_ids))
            .then(None)
            .otherwise(pl.col(time_column))
            .alias(time_column)
        ]
    )

    df_labels_sync = df_labels.with_columns(
        [
            pl.when(pl.col("ID").is_in(all_na_ids))
            .then(None)
            .otherwise(pl.col(event_column))
            .alias(event_column)
        ]
    )

    return df_time_sync, df_labels_sync


def _streamline_duration_transformer_input(
    df_time_train: pl.DataFrame,
    time_column: str,
) -> np.ndarray:
    values = (
        df_time_train.filter(pl.col(time_column).is_not_null())
        .get_column(time_column)
        .to_numpy()
    )

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
    df: pl.DataFrame,
    time_column: str,
    transformer: KBinsDiscretizer | IdentityTransformer,
) -> pl.Series:
    values = df.get_column(time_column).to_numpy()
    nan_mask = np.isnan(values)

    if not nan_mask.any():
        return pl.Series(transformer.transform(values.reshape(-1, 1)).flatten())

    non_null_values = values[~nan_mask]
    transformed = transformer.transform(non_null_values.reshape(-1, 1)).flatten()

    result = np.full(len(df), np.nan)
    result[~nan_mask] = transformed

    return pl.Series(result)


def set_up_delayed_target_labels(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_name: str,
) -> Labels:
    df_train = pl.DataFrame(
        {
            "ID": list(train_ids),
            f"{output_name}": ["DELAYED"] * len(train_ids),
        },
        schema_overrides={f"{output_name}": pl.Categorical},
    )

    df_valid = pl.DataFrame(
        {
            "ID": list(valid_ids),
            f"{output_name}": ["DELAYED"] * len(valid_ids),
        },
        schema_overrides={f"{output_name}": pl.Categorical},
    )

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
    Note we have the None here because we want to be able to
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

    train_values = [id_to_data_pointer_mapping.get(id_, None) for id_ in train_ids]
    df_train = pl.DataFrame(
        {
            "ID": list(train_ids),
            f"{output_name}": train_values,
        }
    )

    valid_values = [id_to_data_pointer_mapping.get(id_, None) for id_ in valid_ids]
    df_valid = pl.DataFrame(
        {
            "ID": list(valid_ids),
            f"{output_name}": valid_values,
        }
    )

    df_train = df_train.with_columns([pl.col(f"{output_name}").cast(pl.Utf8)])
    df_valid = df_valid.with_columns([pl.col(f"{output_name}").cast(pl.Utf8)])

    return Labels(
        train_labels=df_train,
        valid_labels=df_valid,
        label_transformers={},
    )


def gather_torch_null_missing_ids(labels: pl.DataFrame, output_name: str) -> set[str]:
    missing_ids = (
        labels.filter(pl.col(output_name).is_null()).get_column("ID").to_list()
    )

    return {str(id_) for id_ in missing_ids}


def gather_data_pointers_from_data_source(
    data_source: Path,
    validate: bool = True,
    output_inner_key: str | None = None,
) -> dict[str, str | int]:
    """
    Disk: ID -> file path
    Deeplake: ID -> integer index
    """
    iterator: Generator[tuple[str, str]] | Generator[tuple[str, int]]
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
        iterator = ((f.stem, str(f)) for f in iterator_base)

    logger.debug("Gathering data pointers from %s.", data_source)
    id_to_pointer_mapping = {}
    for id_, pointer in tqdm(iterator, desc="Progress"):
        if id_ in id_to_pointer_mapping:
            raise ValueError(f"Duplicate ID: {id_}")

        id_to_pointer_mapping[id_] = pointer

    return id_to_pointer_mapping


def build_deeplake_available_pointer_iterator(
    data_source: Path, inner_key: str
) -> Generator[tuple[str, int]]:
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

        yield id_, int(int_pointer)  # type: ignore


def gather_all_ids_from_output_configs(
    output_configs: Sequence[schemas.OutputConfig],
) -> tuple[str, ...]:
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
                f"Only csv and directory data sources are supported. Got: {cur_source}"
            )
        all_ids.update(cur_ids)

    return tuple(all_ids)


def read_manual_ids_if_exist(
    manual_valid_ids_file: None | str,
) -> Sequence[str] | None:
    if not manual_valid_ids_file:
        return None

    with open(manual_valid_ids_file) as infile:
        manual_ids = tuple(line.strip() for line in infile)

    return manual_ids


def get_tabular_target_file_infos(
    output_configs: Iterable[schemas.OutputConfig],
) -> dict[str, TabularFileInfo]:
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
