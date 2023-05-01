from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union, Iterable

import pandas as pd

from eir.data_load.label_setup import (
    al_target_label_dict,
    al_label_transformers,
    TabularFileInfo,
    al_all_column_ops,
    set_up_train_and_valid_tabular_data,
    gather_ids_from_tabular_file,
    gather_ids_from_data_source,
)
from eir.setup import schemas


@dataclass
class MergedTargetLabels:
    train_labels: al_target_label_dict
    valid_labels: al_target_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return {**self.train_labels, **self.valid_labels}


def set_up_tabular_target_labels_wrapper(
    tabular_target_file_infos: Dict[str, TabularFileInfo],
    custom_label_ops: al_all_column_ops,
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
) -> MergedTargetLabels:
    df_labels_train = pd.DataFrame(index=train_ids)
    df_labels_valid = pd.DataFrame(index=valid_ids)
    label_transformers = {}

    for output_name, tabular_info in tabular_target_file_infos.items():
        cur_labels = set_up_train_and_valid_tabular_data(
            tabular_file_info=tabular_info,
            custom_label_ops=custom_label_ops,
            train_ids=train_ids,
            valid_ids=valid_ids,
        )

        df_train_cur = pd.DataFrame.from_dict(cur_labels.train_labels, orient="index")
        df_valid_cur = pd.DataFrame.from_dict(cur_labels.valid_labels, orient="index")

        df_train_cur["Output Name"] = output_name
        df_valid_cur["Output Name"] = output_name

        df_labels_train = pd.concat((df_labels_train, df_train_cur))
        df_labels_valid = pd.concat((df_labels_valid, df_valid_cur))

        cur_transformers = cur_labels.label_transformers
        label_transformers[output_name] = cur_transformers

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


def df_to_nested_dict(df: pd.DataFrame) -> Dict:
    """
    The df has a 2-level multi index, like so ['ID', output_name]

    We want to convert it to a nested dict like so:

        {'ID': {output_name: {target_output_column: target_column_value}}}
    """
    index_dict = df.to_dict(orient="index")

    parsed_dict = {}
    for key_tuple, value in index_dict.items():
        cur_id, cur_output_name = key_tuple

        if cur_id not in parsed_dict:
            parsed_dict[cur_id] = {}

        parsed_dict[cur_id][cur_output_name] = value

    return parsed_dict


def gather_all_ids_from_output_configs(
    output_configs: Sequence[schemas.OutputConfig],
) -> Tuple[str, ...]:
    all_ids = set()
    for config in output_configs:
        cur_source = Path(config.output_info.output_source)
        if cur_source.suffix == ".csv":
            cur_ids = gather_ids_from_tabular_file(file_path=cur_source)
        elif cur_source.is_dir():
            cur_ids = gather_ids_from_data_source(data_source=cur_source)
        else:
            raise NotImplementedError()
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
    tabular_infos = {}

    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            raise NotImplementedError()

        output_name = output_config.output_info.output_name
        tabular_info = TabularFileInfo(
            file_path=Path(output_config.output_info.output_source),
            con_columns=output_config.output_type_info.target_con_columns,
            cat_columns=output_config.output_type_info.target_cat_columns,
            parsing_chunk_size=output_config.output_type_info.label_parsing_chunk_size,
        )
        tabular_infos[output_name] = tabular_info

    return tabular_infos
