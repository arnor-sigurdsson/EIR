import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Union,
    Dict,
    Sequence,
    Callable,
    Iterable,
    Literal,
)

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

import eir.visualization.visualization_funcs as vf
from eir.data_load import datasets, label_setup
from eir.data_load.data_utils import get_output_info_generator
from eir.data_load.label_setup import (
    al_label_transformers_object,
    al_all_column_ops,
)
from eir.experiment_io.experiment_io import (
    get_run_folder_from_model_path,
    LoadedTrainExperiment,
    load_serialized_train_experiment,
)
from eir.models import al_fusion_models
from eir.models.model_setup import (
    get_meta_model_class_and_kwargs_from_configs,
    load_model,
    get_default_model_registry_per_input_type,
    get_default_meta_class,
)
from eir.models.model_training_utils import gather_pred_outputs_from_dloader
from eir.predict_modules.predict_activations import compute_predict_activations
from eir.predict_modules.predict_config import converge_train_and_predict_configs
from eir.predict_modules.predict_data import set_up_default_dataset
from eir.predict_modules.predict_input_setup import (
    set_up_inputs_for_predict,
)
from eir.predict_modules.predict_target_setup import get_target_labels_for_testing
from eir.setup.config import (
    Configs,
    get_main_parser,
)
from eir.setup.input_setup import (
    al_input_objects_as_dict,
)
from eir.setup.output_setup import al_output_objects_as_dict
from eir.train import (
    prepare_base_batch_default,
    Hooks,
    gather_all_ids_from_output_configs,
    check_dataset_and_batch_size_compatiblity,
)
from eir.train_utils.evaluation import PerformancePlotConfig
from eir.train_utils.metrics import (
    al_metric_record_dict,
    calculate_batch_metrics,
    al_step_metric_dict,
)


logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class PredictSpecificCLArgs:
    model_path: str
    evaluate: bool
    output_folder: str
    act_background_source: Union[Literal["train"], Literal["predict"]]


def main():
    main_parser = get_main_parser(output_nargs="*")

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

    _verify_predict_cl_args(predict_cl_args=predict_cl_args)

    run_predict(predict_cl_args=predict_cl_args)


def _verify_predict_cl_args(predict_cl_args: Namespace):
    if predict_cl_args.evaluate:
        if len(predict_cl_args.output_configs) == 0:
            raise ValueError(
                "If you want to evaluate, you must specify at least one output config."
                "This is needed to know the target column(s) and values to compute"
                "metrics for (i.e., to compare predicted vs. true values)."
            )


def run_predict(predict_cl_args: Namespace):

    run_folder = get_run_folder_from_model_path(model_path=predict_cl_args.model_path)
    loaded_train_experiment = load_serialized_train_experiment(run_folder=run_folder)

    predict_config = get_default_predict_config(
        loaded_train_experiment=loaded_train_experiment,
        predict_cl_args=predict_cl_args,
    )

    predict(predict_config=predict_config, predict_cl_args=predict_cl_args)

    if predict_config.train_configs_overloaded.global_config.get_acts:
        compute_predict_activations(
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


def get_default_predict_config(
    loaded_train_experiment: "LoadedTrainExperiment",
    predict_cl_args: Namespace,
) -> PredictConfig:
    configs_overloaded_for_predict = converge_train_and_predict_configs(
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
    test_dataset = set_up_default_dataset(
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


def _save_predictions(df_predictions: pd.DataFrame, outfolder: Path) -> None:
    df_predictions.to_csv(path_or_buf=str(outfolder / "predictions.csv"))


if __name__ == "__main__":
    main()
