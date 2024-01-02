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
    supervised_target_labels = set_up_all_target_labels_wrapper(
        output_configs=output_configs,
        custom_label_ops=custom_ops,
        train_ids=train_ids,
        valid_ids=valid_ids,
    )
    save_transformer_set(
        transformers_per_source=supervised_target_labels.label_transformers,
        run_folder=run_folder,
    )

    return supervised_target_labels


@dataclass
class MergedTargetLabels:
    train_labels: al_target_label_dict
    valid_labels: al_target_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return {**self.train_labels, **self.valid_labels}


def set_up_all_target_labels_wrapper(
    output_configs: Sequence[schemas.OutputConfig],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> MergedTargetLabels:
    df_labels_train = pd.DataFrame(index=list(train_ids))
    df_labels_valid = pd.DataFrame(index=list(valid_ids))
    label_transformers = {}

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
            case "sequence":
                cur_labels = set_up_delayed_target_labels(
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_name=output_name,
                )

            case "array":
                cur_labels = set_up_file_target_labels(
                    train_ids=train_ids,
                    valid_ids=valid_ids,
                    output_config=output_config,
                )

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

    labels_data_object = MergedTargetLabels(
        train_labels=train_labels_dict,
        valid_labels=valid_labels_dict,
        label_transformers=label_transformers,
    )

    return labels_data_object


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


def set_up_file_target_labels(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    output_config: schemas.OutputConfig,
) -> Labels:
    output_name = output_config.output_info.output_name
    output_source = output_config.output_info.output_source

    id_to_data_pointer_mapping = gather_data_pointers_from_data_source(
        data_source=Path(output_source),
        validate=True,
    )

    train_ids_set = set(train_ids)
    train_labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping[id_]} for id_ in train_ids_set
    }

    valid_ids_set = set(valid_ids)
    valid_labels: al_label_dict = {
        id_: {output_name: id_to_data_pointer_mapping[id_]} for id_ in valid_ids_set
    }

    return Labels(
        train_labels=train_labels,
        valid_labels=valid_labels,
        label_transformers={},
    )


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
) -> dict[str, dict[str, dict[str, float | int | Path]]]:
    """
    The df has a 2-level multi index, like so ['ID', output_name]

    We want to convert it to a nested dict like so:

        {'ID': {output_name: {target_output_column: target_column_value}}}
    """
    index_dict = df.to_dict(orient="index")

    parsed_dict: dict[str, dict[str, dict[str, float | int | Path]]] = {}
    for key_tuple, value in index_dict.items():
        cur_id, cur_output_name = key_tuple

        if cur_id not in parsed_dict:
            parsed_dict[cur_id] = {}

        parsed_dict[cur_id][cur_output_name] = value

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
