import json
from argparse import Namespace
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import sample
from typing import (
    Union,
    Dict,
    Sequence,
    Callable,
    Iterable,
    Tuple,
    Generator,
    Literal,
)

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

import eir.visualization.visualization_funcs as vf
from eir.data_load import datasets, label_setup
from eir.data_load.data_utils import get_output_info_generator
from eir.data_load.datasets import (
    al_datasets,
)
from eir.data_load.label_setup import (
    al_label_dict,
    al_label_transformers_object,
    al_label_transformers,
    al_all_column_ops,
    transform_label_df,
    TabularFileInfo,
)
from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    LoadedTrainExperiment,
    load_serialized_train_experiment,
    load_serialized_input_object,
    load_transformers,
)
from eir.interpretation.interpretation import (
    activation_analysis_wrapper,
)
from eir.models import al_fusion_models
from eir.models.model_setup import (
    get_meta_model_class_and_kwargs_from_configs,
    load_model,
    get_default_model_registry_per_input_type,
    get_default_meta_class,
)
from eir.models.model_training_utils import gather_pred_outputs_from_dloader
from eir.setup import config
from eir.setup import input_setup
from eir.setup import schemas
from eir.setup.config import (
    Configs,
    get_main_parser,
    recursive_dict_replace,
    object_to_primitives,
)
from eir.setup.input_setup import (
    al_input_objects_as_dict,
    SequenceInputInfo,
    BytesInputInfo,
    ImageInputInfo,
)
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train import (
    prepare_base_batch_default,
    Hooks,
    get_tabular_target_file_infos,
    gather_all_ids_from_output_configs,
    check_dataset_and_batch_size_compatiblity,
    df_to_nested_dict,
)
from eir.train_utils.evaluation import PerformancePlotConfig
from eir.train_utils.metrics import (
    al_metric_record_dict,
    calculate_batch_metrics,
    al_step_metric_dict,
)

al_named_dict_configs = Dict[
    Literal["global_configs", "fusion_configs", "input_configs", "output_configs"],
    Iterable[Dict],
]


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class PredictSpecificCLArgs:
    model_path: str
    evaluate: bool
    output_folder: str
    act_background_source: Union[Literal["train"], Literal["predict"]]


def main():
    main_parser = get_main_parser()

    main_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model to use for predictions.",
    )

    main_parser.add_argument("--evaluate", dest="evaluate", action="store_true")

    main_parser.add_argument(
        "--output_folder",
        type=str,
        help="Where to save prediction results.",
        required=True,
    )

    main_parser.add_argument(
        "--act_background_source",
        type=str,
        help="For activation analysis, whether to load backgrounds from the data used "
        "for training or to use the current data passed to the predict module.",
        choices=["train", "predict"],
        default="train",
    )

    predict_cl_args = main_parser.parse_args()

    run_predict(predict_cl_args=predict_cl_args)


def run_predict(predict_cl_args: Namespace):

    run_folder = get_run_folder_from_model_path(model_path=predict_cl_args.model_path)
    loaded_train_experiment = load_serialized_train_experiment(run_folder=run_folder)

    predict_config = get_default_predict_config(
        loaded_train_experiment=loaded_train_experiment,
        predict_cl_args=predict_cl_args,
    )

    predict(predict_config=predict_config, predict_cl_args=predict_cl_args)

    if predict_config.train_configs_overloaded.global_config.get_acts:
        _compute_predict_activations(
            loaded_train_experiment=loaded_train_experiment,
            predict_config=predict_config,
        )


def predict(
    predict_config: "PredictConfig",
    predict_cl_args: Namespace,
) -> None:

    all_preds, all_labels, all_ids = gather_pred_outputs_from_dloader(
        data_loader=predict_config.test_dataloader,
        batch_prep_hook=predict_config.hooks.predict_stages.base_prepare_batch,
        batch_prep_hook_kwargs={"predict_config": predict_config},
        model=predict_config.model,
        with_labels=predict_cl_args.evaluate,
    )

    if predict_cl_args.evaluate:
        metrics = calculate_batch_metrics(
            outputs_as_dict=predict_config.outputs,
            outputs=all_preds,
            labels=all_labels,
            mode="val",
            metric_record_dict=predict_config.metrics,
        )
        serialize_prediction_metrics(
            output_folder=Path(predict_cl_args.output_folder), metrics=metrics
        )

    target_columns_gen = get_output_info_generator(
        outputs_as_dict=predict_config.outputs
    )

    for output_name, target_column_type, target_column_name in target_columns_gen:

        target_preds = all_preds[output_name][target_column_name]
        predictions = _parse_predictions(target_preds=target_preds)

        target_labels = None
        if all_labels:
            target_labels = all_labels[output_name][target_column_name].cpu().numpy()

        target_transformers = predict_config.outputs[output_name].target_transformers
        cur_target_transformer = target_transformers[target_column_name]
        classes = _get_target_class_names(
            transformer=cur_target_transformer, target_column=target_column_name
        )

        output_folder = Path(
            predict_cl_args.output_folder, output_name, target_column_name
        )
        ensure_path_exists(path=output_folder, is_folder=True)

        merged_predictions = _merge_ids_predictions_and_labels(
            ids=all_ids,
            predictions=predictions,
            labels=target_labels,
            prediction_classes=classes,
        )

        _save_predictions(
            df_predictions=merged_predictions,
            outfolder=output_folder,
        )

        if predict_cl_args.evaluate:
            cur_labels = all_labels[output_name][target_column_name].cpu().numpy()

            plot_config = PerformancePlotConfig(
                val_outputs=predictions,
                val_labels=cur_labels,
                val_ids=all_ids,
                iteration=0,
                column_name=target_column_name,
                column_type=target_column_type,
                target_transformer=cur_target_transformer,
                output_folder=output_folder,
            )

            vf.gen_eval_graphs(plot_config=plot_config)


def _merge_ids_predictions_and_labels(
    ids: Sequence[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    prediction_classes: Union[Sequence[str], None] = None,
    label_column_name: str = "True Label",
) -> pd.DataFrame:
    df = pd.DataFrame()

    df["ID"] = ids
    df = df.set_index("ID")

    df[label_column_name] = labels

    if prediction_classes is None:
        prediction_classes = [f"Score Class {i}" for i in range(predictions.shape[1])]

    df[prediction_classes] = predictions

    return df


def serialize_prediction_metrics(output_folder: Path, metrics: al_step_metric_dict):
    with open(str(output_folder / "calculated_metrics.json"), "w") as outfile:
        parsed = _convert_dict_values_to_python_objects(object_=metrics)
        json.dump(obj=parsed, fp=outfile)


def _convert_dict_values_to_python_objects(object_):
    """
    Needed since json does not serialize numpy dtypes natively.
    """

    if isinstance(object_, np.number):
        return object_.item()
    elif isinstance(object_, dict):
        for key, value in object_.items():
            object_[key] = _convert_dict_values_to_python_objects(object_=value)

    return object_


@dataclass
class PredictConfig:
    train_configs_overloaded: Configs
    inputs: al_input_objects_as_dict
    outputs: al_output_objects_as_dict
    predict_specific_cl_args: PredictSpecificCLArgs
    test_dataset: datasets.DiskDataset
    test_dataloader: DataLoader
    model: al_fusion_models
    hooks: "PredictHooks"
    metrics: al_metric_record_dict


@dataclass
class PredictHooks:

    predict_stages: "PredictHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None


@dataclass
class PredictHookStages:

    al_hook = Callable[..., Dict]
    al_hooks = [Iterable[al_hook]]

    base_prepare_batch: al_hooks
    model_forward: al_hooks


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


@dataclass
class PredictTargetLabels:
    label_dict: al_label_dict
    label_transformers: Dict[str, al_label_transformers]

    @property
    def all_labels(self):
        return self.label_dict


def get_default_predict_config(
    loaded_train_experiment: "LoadedTrainExperiment",
    predict_cl_args: Namespace,
) -> PredictConfig:
    configs_overloaded_for_predict = _converge_train_and_predict_configs(
        train_configs=loaded_train_experiment.configs, predict_cl_args=predict_cl_args
    )

    default_train_hooks = loaded_train_experiment.hooks

    if predict_cl_args.evaluate:
        test_ids = gather_all_ids_from_output_configs(
            output_configs=configs_overloaded_for_predict.output_configs
        )

        label_ops = default_train_hooks.custom_column_label_parsing_ops
        target_labels = get_target_labels_for_testing(
            configs_overloaded_for_predict=configs_overloaded_for_predict,
            custom_column_label_parsing_ops=label_ops,
            ids=test_ids,
        )
    else:
        test_ids = label_setup.gather_all_ids_from_all_inputs(
            input_configs=configs_overloaded_for_predict.input_configs
        )
        target_labels = None

    test_inputs = set_up_inputs_for_predict(
        test_inputs_configs=configs_overloaded_for_predict.input_configs,
        ids=test_ids,
        hooks=default_train_hooks,
        output_folder=loaded_train_experiment.configs.global_config.output_folder,
    )

    label_dict = target_labels.label_dict if target_labels else {}
    test_dataset = _set_up_default_dataset(
        configs=configs_overloaded_for_predict,
        target_labels_dict=label_dict,
        inputs_as_dict=test_inputs,
        outputs_as_dict=loaded_train_experiment.outputs,
    )

    check_dataset_and_batch_size_compatiblity(
        dataset=test_dataset,
        batch_size=configs_overloaded_for_predict.global_config.batch_size,
        name="Test",
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=configs_overloaded_for_predict.global_config.batch_size,
        shuffle=False,
        num_workers=configs_overloaded_for_predict.global_config.dataloader_workers,
    )

    default_model_registry = get_default_model_registry_per_input_type()

    func = get_meta_model_class_and_kwargs_from_configs
    fusion_model_class, fusion_model_kwargs = func(
        global_config=configs_overloaded_for_predict.global_config,
        fusion_config=configs_overloaded_for_predict.fusion_config,
        inputs_as_dict=test_inputs,
        outputs_as_dict=loaded_train_experiment.outputs,
        model_registry_per_input_type=default_model_registry,
        model_registry_per_output_type={},
        meta_class_getter=get_default_meta_class,
    )

    model = load_model(
        model_path=Path(predict_cl_args.model_path),
        model_class=fusion_model_class,
        model_init_kwargs=fusion_model_kwargs,
        device=configs_overloaded_for_predict.global_config.device,
        test_mode=True,
        strict_shapes=True,
    )
    assert not model.training

    predict_specific_cl_args = extract_predict_specific_cl_args(
        all_predict_cl_args=predict_cl_args
    )

    default_predict_hooks = _get_default_predict_hooks(train_hooks=default_train_hooks)
    test_config = PredictConfig(
        train_configs_overloaded=configs_overloaded_for_predict,
        inputs=test_inputs,
        outputs=loaded_train_experiment.outputs,
        predict_specific_cl_args=predict_specific_cl_args,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        model=model,
        hooks=default_predict_hooks,
        metrics=loaded_train_experiment.metrics,
    )
    return test_config


@dataclass
class LoadedTrainExperimentMixedWithPredict(LoadedTrainExperiment):
    model: nn.Module
    inputs: al_input_objects_as_dict


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
    train_con_column_means = _prep_missing_con_dict(con_transformers=con_transformers)

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


def _get_default_predict_hooks(train_hooks: "Hooks") -> PredictHooks:
    stages = PredictHookStages(
        base_prepare_batch=[_hook_default_predict_prepare_batch],
        model_forward=[train_hooks.step_func_hooks.model_forward],
    )
    predict_hooks = PredictHooks(
        predict_stages=stages,
        custom_column_label_parsing_ops=train_hooks.custom_column_label_parsing_ops,
    )

    return predict_hooks


def _hook_default_predict_prepare_batch(
    predict_config: "PredictConfig",
    loader_batch,
    *args,
    **kwargs,
):
    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        input_objects=predict_config.inputs,
        output_objects=predict_config.outputs,
        model=predict_config.model,
        device=predict_config.train_configs_overloaded.global_config.device,
    )

    state_updates = {"batch": batch}

    return state_updates


def extract_predict_specific_cl_args(
    all_predict_cl_args: Namespace,
) -> PredictSpecificCLArgs:
    predict_module_specific_keys = PredictSpecificCLArgs.__dataclass_fields__.keys()

    dataclass_kwargs = {
        k: v
        for k, v in all_predict_cl_args.__dict__.items()
        if k in predict_module_specific_keys
    }
    predict_specific_cl_args_object = PredictSpecificCLArgs(**dataclass_kwargs)

    return predict_specific_cl_args_object


def _parse_predictions(target_preds: torch.Tensor) -> np.ndarray:

    predictions = target_preds.cpu().numpy()
    return predictions


def _get_target_class_names(
    transformer: al_label_transformers_object, target_column: str
):
    if isinstance(transformer, LabelEncoder):
        return transformer.classes_
    return [target_column]


def _converge_train_and_predict_configs(
    train_configs: Configs, predict_cl_args: Namespace
) -> Configs:

    train_configs_copy = deepcopy(train_configs)

    named_dict_iterators = get_named_pred_dict_iterators(
        predict_cl_args=predict_cl_args
    )

    matched_dict_iterator = get_train_predict_matched_config_generator(
        train_configs=train_configs_copy, named_dict_iterators=named_dict_iterators
    )

    predict_configs_overloaded = overload_train_configs_for_predict(
        matched_dict_iterator=matched_dict_iterator
    )

    return predict_configs_overloaded


def get_named_pred_dict_iterators(
    predict_cl_args: Namespace,
) -> al_named_dict_configs:
    target_keys = {
        "global_configs",
        "fusion_configs",
        "input_configs",
        "output_configs",
    }

    dict_of_generators = {}
    for key, value in predict_cl_args.__dict__.items():

        if key in target_keys:

            if not value:
                value = ()
            cur_gen = config.get_yaml_to_dict_iterator(yaml_config_files=value)
            dict_of_generators[key] = tuple(cur_gen)
    return dict_of_generators


def get_train_predict_matched_config_generator(
    train_configs: Configs,
    named_dict_iterators: al_named_dict_configs,
) -> Generator[Tuple[str, Dict, Dict], None, None]:
    train_keys = set(train_configs.__dict__.keys())

    single_configs = {
        "global_configs": "global_config",
        "fusion_configs": "fusion_config",
    }

    sequence_configs = {"input_configs", "output_configs"}

    for predict_argument_name, predict_dict_iter in named_dict_iterators.items():
        name_in_configs_object = single_configs.get(
            predict_argument_name, predict_argument_name
        )
        assert name_in_configs_object in train_keys

        if predict_dict_iter is None:
            predict_dict_iter = []

        # If not a sequence we can yield directly
        if predict_argument_name in single_configs.keys():
            train_config = getattr(train_configs, name_in_configs_object)
            train_config_as_dict = object_to_primitives(obj=train_config)

            predict_config_as_dict = config.combine_dicts(dicts=predict_dict_iter)

            yield (
                name_in_configs_object,
                train_config_as_dict,
                predict_config_as_dict,
            )

        # Otherwise we have to match the respective ones with each other
        elif predict_argument_name in sequence_configs:
            train_config_sequence = getattr(train_configs, name_in_configs_object)

            for cur_config in train_config_sequence:
                matching_func = get_config_sequence_matching_func(
                    name=name_in_configs_object
                )
                pred_dict_match_from_iter = matching_func(
                    train_config=cur_config, predict_dict_iterator=predict_dict_iter
                )

                cur_train_config_as_dict = object_to_primitives(obj=cur_config)

                yield (
                    name_in_configs_object,
                    cur_train_config_as_dict,
                    pred_dict_match_from_iter,
                )


def get_config_sequence_matching_func(
    name: Literal["input_configs", "output_configs"]
) -> Callable:
    assert name in ("input_configs", "output_configs")

    def _input_configs(
        train_config: schemas.InputConfig,
        predict_dict_iterator: Iterable[Dict],
    ):
        matches = []

        train_input_info = train_config.input_info
        for predict_input_config_dict in predict_dict_iterator:
            feature_extractor_info = predict_input_config_dict["input_info"]
            cond_1 = feature_extractor_info["input_name"] == train_input_info.input_name
            cond_2 = feature_extractor_info["input_type"] = train_input_info.input_type

            if all((cond_1, cond_2)):
                matches.append(predict_input_config_dict)

        assert len(matches) == 1, matches
        return matches[0]

    def _output_configs(
        train_config: schemas.OutputConfig,
        predict_dict_iterator: Iterable[Dict],
    ):
        matches = []

        train_cat_columns = train_config.output_type_info.target_cat_columns
        train_con_columns = train_config.output_type_info.target_con_columns

        for predict_output_config_dict in predict_dict_iterator:
            cat_cols = predict_output_config_dict.get("output_type_info", {}).get(
                "target_cat_columns", []
            )
            con_cols = predict_output_config_dict.get("output_type_info", {}).get(
                "target_con_columns", []
            )

            cond_1 = train_cat_columns == cat_cols
            cond_2 = train_con_columns == con_cols

            if all((cond_1, cond_2)):
                matches.append(predict_output_config_dict)

        assert len(matches) == 1, matches
        return matches[0]

    if name == "input_configs":
        return _input_configs
    return _output_configs


def overload_train_configs_for_predict(
    matched_dict_iterator: Generator[Tuple[str, Dict, Dict], None, None],
) -> Configs:

    main_overloaded_kwargs = {}

    for name, train_config_dict, predict_config_dict_to_inject in matched_dict_iterator:

        _maybe_warn_about_output_folder_overload_from_predict(
            name=name,
            predict_config_dict_to_inject=predict_config_dict_to_inject,
            train_config_dict=train_config_dict,
        )

        overloaded_dict = recursive_dict_replace(
            dict_=train_config_dict, dict_to_inject=predict_config_dict_to_inject
        )
        if name in ("global_config", "fusion_config"):
            main_overloaded_kwargs[name] = overloaded_dict
        elif name in ("input_configs", "output_configs"):
            main_overloaded_kwargs.setdefault(name, [])
            main_overloaded_kwargs.get(name).append(overloaded_dict)

    global_config_overloaded = config.get_global_config(
        global_configs=[main_overloaded_kwargs.get("global_config")]
    )
    input_configs_overloaded = config.get_input_configs(
        input_configs=main_overloaded_kwargs.get("input_configs")
    )
    fusion_config_overloaded = config.load_fusion_configs(
        [main_overloaded_kwargs.get("fusion_config")]
    )

    tabular_output_setup = config.DynamicOutputSetup(
        output_types_schema_map=config.get_outputs_types_schema_map(),
        output_module_config_class_getter=config.get_output_module_config_class,
        output_module_init_class_map=config.get_output_config_type_init_callable_map(),
    )

    output_configs_overloaded = config.load_output_configs(
        output_configs=main_overloaded_kwargs.get("output_configs"),
        dynamic_output_setup=tabular_output_setup,
    )

    train_configs_overloaded = config.Configs(
        global_config=global_config_overloaded,
        input_configs=input_configs_overloaded,
        fusion_config=fusion_config_overloaded,
        output_configs=output_configs_overloaded,
    )

    return train_configs_overloaded


def _maybe_warn_about_output_folder_overload_from_predict(
    name: str, predict_config_dict_to_inject: Dict, train_config_dict: Dict
) -> None:
    if name == "global_config" and "output_folder" in predict_config_dict_to_inject:

        output_folder_from_predict = predict_config_dict_to_inject["output_folder"]
        output_folder_from_train = train_config_dict["output_folder"]

        if output_folder_from_predict == output_folder_from_train:
            return

        logger.warning(
            "output_folder in '%s' will be replaced with '%s' from given prediction "
            "output configuration ('%s'). If this is intentional, you can ignore this "
            "message but most likely this is a mistake and will cause an error. "
            "Resolution: Remove 'output_folder' from "
            "the global output config as it will "
            "be automatically looked up based on the experiment.",
            train_config_dict,
            output_folder_from_predict,
            predict_config_dict_to_inject,
        )


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
    train_con_column_means = _prep_missing_con_dict(con_transformers=con_transformers)

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


def _prep_missing_con_dict(con_transformers: al_label_transformers) -> Dict[str, float]:

    train_means = {
        column: transformer.mean_[0] for column, transformer in con_transformers.items()
    }

    return train_means


def _set_up_default_dataset(
    configs: Configs,
    target_labels_dict: Union[None, al_label_dict],
    inputs_as_dict: al_input_objects_as_dict,
    outputs_as_dict: al_output_objects_as_dict,
) -> al_datasets:

    test_dataset_kwargs = datasets.construct_default_dataset_kwargs_from_cl_args(
        target_labels_dict=target_labels_dict,
        outputs=outputs_as_dict,
        inputs=inputs_as_dict,
        test_mode=True,
    )

    test_dataset = datasets.DiskDataset(**test_dataset_kwargs)

    return test_dataset


def _save_predictions(df_predictions: pd.DataFrame, outfolder: Path) -> None:
    df_predictions.to_csv(path_or_buf=str(outfolder / "predictions.csv"))


def _compute_predict_activations(
    loaded_train_experiment: "LoadedTrainExperiment",
    predict_config: PredictConfig,
) -> None:

    gc = predict_config.train_configs_overloaded.global_config

    background_source = predict_config.predict_specific_cl_args.act_background_source
    background_source_config = get_background_source_config(
        background_source_in_predict_cl_args=background_source,
        train_configs=loaded_train_experiment.configs,
        predict_configs=predict_config.train_configs_overloaded,
    )
    background_dataloader = _get_predict_background_loader(
        batch_size=gc.batch_size,
        num_act_background_samples=gc.act_background_samples,
        outputs_as_dict=loaded_train_experiment.outputs,
        configs=background_source_config,
        dataloader_workers=gc.dataloader_workers,
        loaded_hooks=loaded_train_experiment.hooks,
    )

    overloaded_train_experiment = _overload_train_experiment_for_predict_activations(
        train_config=loaded_train_experiment,
        predict_config=predict_config,
    )

    activation_outfolder_callable = partial(
        _get_predict_activation_outfolder_target,
        predict_outfolder=Path(predict_config.predict_specific_cl_args.output_folder),
    )

    activation_analysis_wrapper(
        model=predict_config.model,
        experiment=overloaded_train_experiment,
        outfolder_target_callable=activation_outfolder_callable,
        dataset_to_interpret=predict_config.test_dataset,
        background_loader=background_dataloader,
    )


def get_background_source_config(
    background_source_in_predict_cl_args: Literal["train", "predict"],
    train_configs: Configs,
    predict_configs: Configs,
) -> Configs:
    """
    TODO:   In the case of predict, make sure background and samples analysed are
            separated.
    """
    if background_source_in_predict_cl_args == "predict":
        logger.info(
            "Background for activation analysis will be loaded from sources "
            "passed to predict.py."
        )
        return predict_configs

    elif background_source_in_predict_cl_args == "train":
        logger.info(
            "Background for activation analysis will be loaded from sources "
            "previously used for training run with name '%s'.",
            train_configs.global_config.output_folder,
        )
        return train_configs

    raise ValueError()


def _overload_train_experiment_for_predict_activations(
    train_config: LoadedTrainExperiment,
    predict_config: PredictConfig,
) -> LoadedTrainExperimentMixedWithPredict:
    """
    TODO:   Possibly set inputs=None as a field in LoadedTrainExperiment that then gets
            filled with test_inputs. When we do not need the weird monkey-patching here
            of the batch_prep_hooks, as the LoadedTrainExperiment will have the
            inputs attribute.
    """
    train_experiment_copy = copy(train_config)

    mixed_experiment_kwargs = train_experiment_copy.__dict__
    mixed_experiment_kwargs["model"] = predict_config.model
    mixed_experiment_kwargs["configs"] = predict_config.train_configs_overloaded
    mixed_experiment_kwargs["inputs"] = predict_config.inputs

    mixed_experiment = LoadedTrainExperimentMixedWithPredict(**mixed_experiment_kwargs)

    return mixed_experiment


def _get_predict_background_loader(
    batch_size: int,
    num_act_background_samples: int,
    dataloader_workers: int,
    configs: Configs,
    outputs_as_dict: al_output_objects_as_dict,
    loaded_hooks: Union["Hooks", None],
):
    """
    TODO: Add option to choose whether to reuse train data as background,
          to use the data passed to the predict.py module, or possibly just
          an option to serialize the explainer from a training run and reuse
          that if passed as an option here.
    """

    background_ids_pool = label_setup.gather_all_ids_from_all_inputs(
        input_configs=configs.input_configs
    )
    background_ids_sampled = sample(
        population=background_ids_pool,
        k=num_act_background_samples,
    )

    target_labels = get_target_labels_for_testing(
        configs_overloaded_for_predict=configs,
        custom_column_label_parsing_ops=loaded_hooks.custom_column_label_parsing_ops,
        ids=background_ids_sampled,
    )

    background_inputs_as_dict = set_up_inputs_for_predict(
        test_inputs_configs=configs.input_configs,
        ids=background_ids_sampled,
        hooks=loaded_hooks,
        output_folder=configs.global_config.output_folder,
    )
    background_dataset = _set_up_default_dataset(
        configs=configs,
        outputs_as_dict=outputs_as_dict,
        target_labels_dict=target_labels.label_dict,
        inputs_as_dict=background_inputs_as_dict,
    )

    check_dataset_and_batch_size_compatiblity(
        dataset=background_dataset,
        batch_size=batch_size,
        name="Test activation background",
    )
    background_loader = DataLoader(
        dataset=background_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
    )

    return background_loader


def _get_predict_activation_outfolder_target(
    predict_outfolder: Path, output_name: str, column_name: str, input_name: str
) -> Path:
    activation_outfolder = (
        predict_outfolder / output_name / column_name / "activations" / input_name
    )
    ensure_path_exists(path=activation_outfolder, is_folder=True)

    return activation_outfolder


if __name__ == "__main__":
    main()
