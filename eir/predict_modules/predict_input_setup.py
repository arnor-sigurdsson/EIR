from dataclasses import dataclass
from functools import partial
from typing import Sequence, Union, Callable, Dict

from eir.data_load import label_setup
from eir.data_load.label_setup import (
    al_label_dict,
    al_label_transformers,
    TabularFileInfo,
    al_all_column_ops,
    transform_label_df,
)
from eir.experiment_io.experiment_io import (
    load_transformers,
    load_serialized_input_object,
)
from eir.setup import schemas, input_setup
from eir.setup.input_setup import (
    al_input_objects_as_dict,
    SequenceInputInfo,
    BytesInputInfo,
    ImageInputInfo,
)
from eir.train import Hooks


@dataclass
class PredictTabularInputInfo:
    labels: "PredictInputLabels"
    input_config: schemas.InputConfig


@dataclass
class PredictInputLabels:
    label_dict: al_label_dict
    label_transformers: al_label_transformers

    @property
    def all_labels(self):
        return self.label_dict


def setup_tabular_input_for_testing(
    input_config: schemas.InputConfig,
    ids: Sequence[str],
    hooks: Union["Hooks", None],
    output_folder: str,
) -> PredictTabularInputInfo:

    tabular_file_info = input_setup.get_tabular_input_file_info(
        input_source=input_config.input_info.input_source,
        tabular_data_type_config=input_config.input_type_info,
    )

    custom_ops = hooks.custom_column_label_parsing_ops if hooks else None
    predict_labels = get_input_labels_for_predict(
        tabular_file_info=tabular_file_info,
        input_name=input_config.input_info.input_name,
        custom_label_ops=custom_ops,
        ids=ids,
        output_folder=output_folder,
    )

    predict_input_info = PredictTabularInputInfo(
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

    label_setup._pre_check_label_df(df=df_labels_test, name="Testing DataFrame")

    all_columns = list(tabular_file_info.con_columns) + list(
        tabular_file_info.cat_columns
    )
    label_transformers_with_input_name = load_transformers(
        transformers_to_load={input_name: all_columns}, output_folder=output_folder
    )
    loaded_fit_label_transformers = label_transformers_with_input_name[input_name]

    con_transformers = {
        k: v
        for k, v in loaded_fit_label_transformers.items()
        if k in tabular_file_info.con_columns
    }
    train_con_column_means = prep_missing_con_dict(con_transformers=con_transformers)

    df_labels_test_no_na = label_setup.handle_missing_label_values_in_df(
        df=df_labels_test,
        cat_label_columns=tabular_file_info.cat_columns,
        con_label_columns=tabular_file_info.con_columns,
        con_manual_values=train_con_column_means,
        name="test_df",
    )

    df_labels_test_final = transform_label_df(
        df_labels=df_labels_test_no_na, label_transformers=loaded_fit_label_transformers
    )

    labels_dict = df_labels_test_final.to_dict("index")

    labels_data_object = PredictInputLabels(
        label_dict=labels_dict,
        label_transformers=loaded_fit_label_transformers,
    )

    return labels_data_object


def set_up_inputs_for_predict(
    test_inputs_configs: schemas.al_input_configs,
    ids: Sequence[str],
    hooks: Union["Hooks", None],
    output_folder: str,
) -> al_input_objects_as_dict:

    train_input_setup_kwargs = {
        "ids": ids,
        "output_folder": output_folder,
    }
    all_inputs = input_setup.set_up_inputs_general(
        inputs_configs=test_inputs_configs,
        hooks=hooks,
        setup_func_getter=get_input_setup_function_for_predict,
        setup_func_kwargs=train_input_setup_kwargs,
    )

    return all_inputs


def get_input_setup_function_for_predict(
    input_config: schemas.InputConfig,
) -> Callable[..., input_setup.al_input_objects]:
    mapping = get_input_setup_function_map_for_predict()
    input_type = input_config.input_info.input_type

    return mapping[input_type]


def get_input_setup_function_map_for_predict() -> Dict[
    str, Callable[..., input_setup.al_input_objects]
]:
    setup_mapping = {
        "omics": input_setup.set_up_omics_input,
        "tabular": setup_tabular_input_for_testing,
        "sequence": partial(
            load_serialized_input_object, input_class=SequenceInputInfo
        ),
        "bytes": partial(load_serialized_input_object, input_class=BytesInputInfo),
        "image": partial(load_serialized_input_object, input_class=ImageInputInfo),
    }

    return setup_mapping


def prep_missing_con_dict(con_transformers: al_label_transformers) -> Dict[str, float]:

    train_means = {
        column: transformer.mean_[0] for column, transformer in con_transformers.items()
    }

    return train_means
