import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Union

import numpy as np
import torchtext
from aislib.misc_utils import ensure_path_exists
from torch.utils.data import DataLoader

torchtext.disable_torchtext_deprecation_warning()

from eir import train
from eir.data_load import datasets, label_setup
from eir.data_load.label_setup import al_all_column_ops
from eir.experiment_io.experiment_io import (
    LoadedTrainExperiment,
    check_version,
    get_run_folder_from_model_path,
    load_serialized_train_experiment,
)
from eir.models.model_setup_modules.meta_setup import (
    al_meta_model,
    get_default_meta_class,
    get_meta_model_class_and_kwargs_from_configs,
)
from eir.models.model_setup_modules.model_io import load_model
from eir.models.model_training_utils import get_prediction_outputs_generator
from eir.predict_modules.predict_attributions import compute_predict_attributions
from eir.predict_modules.predict_config import converge_train_and_predict_configs
from eir.predict_modules.predict_data import set_up_default_dataset
from eir.predict_modules.predict_input_setup import set_up_inputs_for_predict
from eir.predict_modules.predict_output_modules.predict_array_output import (
    predict_array_wrapper,
)
from eir.predict_modules.predict_output_modules.predict_sequence_output import (
    predict_sequence_wrapper,
)
from eir.predict_modules.predict_output_modules.predict_tabular_output import (
    predict_tabular_wrapper_no_labels,
    predict_tabular_wrapper_with_labels,
)
from eir.predict_modules.predict_target_setup import (
    MissingTargetsInfo,
    get_target_labels_for_testing,
)
from eir.setup.config import Configs, get_main_parser
from eir.setup.input_setup import al_input_objects_as_dict
from eir.setup.output_setup import al_output_objects_as_dict
from eir.target_setup.target_label_setup import gather_all_ids_from_output_configs
from eir.train import check_dataset_and_batch_size_compatibility
from eir.train_utils.evaluation import (
    deregister_pre_evaluation_hooks,
    register_pre_evaluation_hooks,
    run_all_eval_hook_analysis,
    run_split_evaluation,
)
from eir.train_utils.metrics import al_metric_record_dict, al_step_metric_dict
from eir.train_utils.step_logic import Hooks, prepare_base_batch_default
from eir.train_utils.utils import set_log_level_for_eir_loggers
from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)

al_predict_hook = Callable[..., dict]
al_predict_hooks = Iterable[al_predict_hook]


@dataclass()
class PredictSpecificCLArgs:
    model_path: str
    evaluate: bool
    output_folder: str
    attribution_background_source: Union[Literal["train"], Literal["predict"]]


def main():
    main_parser = get_main_parser(output_nargs="*", global_nargs="*")

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
        "--attribution_background_source",
        type=str,
        help="For attribution analysis, whether to load backgrounds from the data used "
        "for training or to use the current data passed to the predict module.",
        choices=["train", "predict"],
        default="train",
    )

    predict_cl_args = main_parser.parse_args()

    _verify_predict_cl_args(predict_cl_args=predict_cl_args)
    run_folder = get_run_folder_from_model_path(model_path=predict_cl_args.model_path)
    check_version(run_folder=run_folder)

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

    set_log_level_for_eir_loggers(
        log_level=loaded_train_experiment.configs.global_config.log_level
    )

    predict_experiment = get_default_predict_experiment(
        loaded_train_experiment=loaded_train_experiment,
        predict_cl_args=predict_cl_args,
    )

    predict(
        predict_experiment=predict_experiment,
        predict_cl_args=predict_cl_args,
    )

    if predict_experiment.configs.global_config.compute_attributions:
        compute_predict_attributions(
            loaded_train_experiment=loaded_train_experiment,
            predict_config=predict_experiment,
        )


def predict(
    predict_experiment: "PredictExperiment",
    predict_cl_args: Namespace,
) -> None:
    hook_finalizers = register_pre_evaluation_hooks(
        model=predict_experiment.model,
        global_config=predict_experiment.configs.global_config,
    )

    output_generator = get_prediction_outputs_generator(
        data_loader=predict_experiment.test_dataloader,
        batch_prep_hook=predict_experiment.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"experiment": predict_experiment},
        model=predict_experiment.model,
        with_labels=predict_cl_args.evaluate,
    )

    criteria = train.get_criteria(outputs_as_dict=predict_experiment.outputs)
    loss_func = train.get_loss_callable(criteria=criteria)

    predict_results = run_split_evaluation(
        output_generator=output_generator,
        output_objects=predict_experiment.outputs,
        experiment_metrics=predict_experiment.metrics,
        loss_function=loss_func,
        device=predict_experiment.configs.global_config.device,
        with_labels=predict_cl_args.evaluate,
        missing_ids_per_output=predict_experiment.test_dataset.missing_ids_per_output,
    )

    hook_outputs = deregister_pre_evaluation_hooks(
        hook_finalizers=hook_finalizers,
        evaluation_results=predict_results,
    )

    run_all_eval_hook_analysis(
        hook_outputs=hook_outputs,
        run_folder=Path(predict_cl_args.output_folder),
        iteration="predict",
    )

    if predict_cl_args.evaluate:
        serialize_prediction_metrics(
            output_folder=Path(predict_cl_args.output_folder),
            metrics=predict_results.metrics_with_averages,
        )

        predict_tabular_wrapper_with_labels(
            predict_config=predict_experiment,
            all_predictions=predict_results.gathered_outputs,
            all_labels=predict_results.gathered_labels,
            all_ids=predict_results.ids_per_output,
            predict_cl_args=predict_cl_args,
        )
    else:
        predict_tabular_wrapper_no_labels(
            predict_config=predict_experiment,
            all_predictions=predict_results.gathered_outputs,
            all_ids=predict_results.ids_per_output,
            predict_cl_args=predict_cl_args,
        )

    predict_sequence_wrapper(
        predict_experiment=predict_experiment,
        output_folder=predict_cl_args.output_folder,
    )

    predict_array_wrapper(
        predict_experiment=predict_experiment,
        output_folder=predict_cl_args.output_folder,
    )


def serialize_prediction_metrics(output_folder: Path, metrics: al_step_metric_dict):
    ensure_path_exists(path=output_folder, is_folder=True)
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
class PredictExperiment:
    configs: Configs
    inputs: al_input_objects_as_dict
    outputs: al_output_objects_as_dict
    predict_specific_cl_args: PredictSpecificCLArgs
    test_dataset: datasets.DiskDataset | datasets.MemoryDataset
    test_dataloader: DataLoader
    model: al_meta_model
    hooks: "PredictHooks"
    metrics: al_metric_record_dict


@dataclass
class PredictHooks:
    step_func_hooks: "PredictStepFunctionHookStages"
    custom_column_label_parsing_ops: al_all_column_ops = None


@dataclass
class PredictStepFunctionHookStages:
    base_prepare_batch: al_predict_hooks
    model_forward: al_predict_hooks


def get_default_predict_experiment(
    loaded_train_experiment: "LoadedTrainExperiment",
    predict_cl_args: Namespace,
) -> PredictExperiment:
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
        missing_ids_per_output = target_labels.missing_ids_per_output
    else:
        test_ids = label_setup.gather_all_ids_from_all_inputs(
            input_configs=configs_overloaded_for_predict.input_configs
        )
        target_labels = None
        missing_ids_per_output = MissingTargetsInfo(
            missing_ids_per_modality={},
            missing_ids_within_modality={},
            precomputed_missing_ids={},
        )

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
        missing_ids_per_output=missing_ids_per_output,
    )
    predict_batch_size = _auto_set_test_batch_size(
        batch_size=configs_overloaded_for_predict.global_config.batch_size,
        test_set_size=len(test_dataset),
    )

    check_dataset_and_batch_size_compatibility(
        dataset=test_dataset,
        batch_size=predict_batch_size,
        name="Test",
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=configs_overloaded_for_predict.global_config.dataloader_workers,
    )

    func = get_meta_model_class_and_kwargs_from_configs
    fusion_model_class, fusion_model_kwargs = func(
        global_config=configs_overloaded_for_predict.global_config,
        fusion_config=configs_overloaded_for_predict.fusion_config,
        inputs_as_dict=test_inputs,
        outputs_as_dict=loaded_train_experiment.outputs,
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
    predict_experiment = PredictExperiment(
        configs=configs_overloaded_for_predict,
        inputs=test_inputs,
        outputs=loaded_train_experiment.outputs,
        predict_specific_cl_args=predict_specific_cl_args,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        model=model,
        hooks=default_predict_hooks,
        metrics=loaded_train_experiment.metrics,
    )

    return predict_experiment


def _auto_set_test_batch_size(batch_size: int, test_set_size: int) -> int:
    if test_set_size < batch_size:
        logger.warning(
            f"Test set size ({test_set_size}) is smaller than "
            f"batch size ({batch_size}). "
            f"Setting batch size to test set size."
        )
        batch_size = test_set_size
    return batch_size


def _get_default_predict_hooks(train_hooks: "Hooks") -> PredictHooks:
    stages = PredictStepFunctionHookStages(
        base_prepare_batch=[_hook_default_predict_prepare_batch],
        model_forward=train_hooks.step_func_hooks.model_forward,
    )
    predict_hooks = PredictHooks(
        step_func_hooks=stages,
        custom_column_label_parsing_ops=train_hooks.custom_column_label_parsing_ops,
    )

    return predict_hooks


def _hook_default_predict_prepare_batch(
    experiment: "PredictExperiment",
    loader_batch,
    *args,
    **kwargs,
):
    batch = prepare_base_batch_default(
        loader_batch=loader_batch,
        input_objects=experiment.inputs,
        output_objects=experiment.outputs,
        model=experiment.model,
        device=experiment.configs.global_config.device,
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


if __name__ == "__main__":
    main()
