from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.data_load.label_setup import al_label_transformers_object
from eir.models import model_training_utils
from eir.setup.output_setup import (
    ComputedArrayOutputInfo,
    ComputedImageOutputInfo,
    ComputedSequenceOutputInfo,
    ComputedSurvivalOutputInfo,
    ComputedTabularOutputInfo,
    al_output_objects_as_dict,
)
from eir.setup.schemas import GlobalConfig
from eir.target_setup.target_label_setup import MissingTargetsInfo
from eir.train_utils import metrics, utils
from eir.train_utils.evaluation_modules.evaluation_output_survival import (
    save_survival_evaluation_results_wrapper,
)
from eir.train_utils.evaluation_modules.train_handlers_array_output import (
    array_out_single_sample_evaluation_wrapper,
)
from eir.train_utils.evaluation_modules.train_handlers_sequence_output import (
    sequence_out_single_sample_evaluation_wrapper,
)
from eir.train_utils.ignite_port.engine import Engine
from eir.train_utils.latent_analysis import (
    LatentHookOutput,
    latent_analysis_wrapper,
    register_latent_hook,
)
from eir.utils.logging import get_logger
from eir.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from eir.train import Experiment
    from eir.train_utils.step_logic import al_training_labels_target
    from eir.train_utils.train_handlers import HandlerConfig

logger = get_logger(name=__name__, tqdm_compatible=True)


def validation_handler(engine: Engine, handler_config: "HandlerConfig") -> None:
    """
    A bit hacky how we manually attach metrics here, but that's because we
    don't want to evaluate as a running average (i.e. do it in the step
    function), but rather run over the whole validation dataset as we do
    in this function.
    """
    exp = handler_config.experiment
    gc = exp.configs.global_config
    iteration = engine.state.iteration

    exp.model.eval()

    hook_finalizers = register_pre_evaluation_hooks(
        model=exp.model,
        global_config=gc,
        run_folder=handler_config.run_folder,
        iteration=iteration,
    )

    output_generator = model_training_utils.get_prediction_outputs_generator(
        data_loader=exp.valid_loader,
        batch_prep_hook=exp.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"experiment": exp},
        model=exp.model,
        with_labels=True,
    )
    evaluation_results = run_split_evaluation(
        output_generator=output_generator,
        output_objects=exp.outputs,
        experiment_metrics=exp.metrics,
        loss_function=exp.loss_function,
        device=gc.be.device,
        missing_ids_per_output=exp.valid_dataset.missing_ids_per_output,
        with_labels=True,
    )

    hook_outputs = deregister_pre_evaluation_hooks(
        hook_finalizers=hook_finalizers,
        evaluation_results=evaluation_results,
    )

    max_samples_for_viz: None | int = None
    if gc.latent_sampling is not None:
        max_samples_for_viz = gc.latent_sampling.max_samples_for_viz
    run_all_eval_hook_analysis(
        hook_outputs=hook_outputs,
        max_samples_for_viz=max_samples_for_viz,
    )

    write_eval_header = iteration == gc.ec.sample_interval
    metrics.persist_metrics(
        handler_config=handler_config,
        metrics_dict=evaluation_results.metrics_with_averages,
        iteration=iteration,
        write_header=write_eval_header,
        prefixes={"metrics": "validation_", "writer": "validation"},
    )

    if gc.ec.saved_result_detail_level >= 5:
        save_tabular_evaluation_results_wrapper(
            val_outputs=evaluation_results.gathered_outputs,
            val_labels=evaluation_results.gathered_labels,
            val_ids=evaluation_results.ids_per_output,
            iteration=iteration,
            experiment=handler_config.experiment,
        )

        sequence_out_single_sample_evaluation_wrapper(
            input_objects=exp.inputs,
            experiment=exp,
            iteration=iteration,
            auto_dataset_to_load_from=exp.valid_dataset,
            output_folder=gc.be.output_folder,
        )

        array_out_single_sample_evaluation_wrapper(
            input_objects=exp.inputs,
            experiment=exp,
            iteration=iteration,
            auto_dataset_to_load_from=exp.valid_dataset,
            output_folder=gc.be.output_folder,
        )

        save_survival_evaluation_results_wrapper(
            val_outputs=evaluation_results.gathered_outputs,
            val_labels=evaluation_results.gathered_labels,
            val_ids=evaluation_results.ids_per_output,
            iteration=iteration,
            experiment=handler_config.experiment,
            evaluation_metrics=evaluation_results.metrics_with_averages,
        )

    exp.model.train()


def expand_binary_logits(binary_logits: torch.Tensor) -> torch.Tensor:
    neg_logits = -binary_logits / 2
    pos_logits = binary_logits / 2
    return torch.cat([neg_logits, pos_logits], dim=-1)


def run_all_eval_hook_analysis(
    hook_outputs: dict[str, LatentHookOutput],
    max_samples_for_viz: int | None = None,
) -> None:
    for _hook_name, hook_output in hook_outputs.items():
        match hook_output:
            case LatentHookOutput():
                latent_analysis_wrapper(
                    latent_outputs=hook_output,
                    max_samples_for_viz=max_samples_for_viz,
                )
            case _:
                raise NotImplementedError()


def register_pre_evaluation_hooks(
    model: torch.nn.Module,
    global_config: GlobalConfig,
    run_folder: Path,
    iteration: int | str,
) -> dict[str, Callable]:
    model_hook_finalizers: dict[str, Callable] = {}

    latent_config = global_config.latent_sampling
    if latent_config is not None:
        for layer_path in latent_config.layers_to_sample:
            model_hook_finalizers[f"latent_{layer_path}"] = register_latent_hook(
                model=model,
                layer_path=layer_path,
                batch_size_for_saving=latent_config.batch_size_for_saving,
                run_folder=run_folder,
                iteration=iteration,
            )

    return model_hook_finalizers


def deregister_pre_evaluation_hooks(
    hook_finalizers: dict[str, Callable],
    evaluation_results: "EvaluationResults",
) -> dict[str, LatentHookOutput]:
    hook_outputs = {}
    for hook_name, hook_finalizer in hook_finalizers.items():
        hook_outputs[hook_name] = hook_finalizer(evaluation_results=evaluation_results)

    return hook_outputs


@dataclass()
class SplitModelOutputs:
    output_compute: dict[str, dict[str, torch.Tensor]]
    target_compute: dict[str, dict[str, torch.Tensor]]
    output_gather: dict[str, dict[str, torch.Tensor]]
    target_gather: dict[str, dict[str, torch.Tensor]]
    ids: list[str]


def get_split_output_generator(
    output_generator: Generator[
        model_training_utils.al_dataloader_gathered_predictions
    ],
    output_objects: al_output_objects_as_dict,
) -> Generator[SplitModelOutputs]:
    output_objects_by_type = split_output_objects_by_eval_type(
        output_objects=output_objects
    )

    for model_outputs, target_labels, ids in output_generator:
        output_gather = {
            k: v for k, v in model_outputs.items() if k in output_objects_by_type.gather
        }

        if target_labels is None:
            target_gather = {}
        else:
            target_gather = {
                k: v
                for k, v in target_labels.items()
                if k in output_objects_by_type.gather
            }

        output_compute = {
            k: v
            for k, v in model_outputs.items()
            if k in output_objects_by_type.compute
        }

        if target_labels is None:
            target_compute = {}
        else:
            target_compute = {
                k: v
                for k, v in target_labels.items()
                if k in output_objects_by_type.compute
            }

        yield SplitModelOutputs(
            output_compute=output_compute,
            target_compute=target_compute,
            output_gather=output_gather,
            target_gather=target_gather,
            ids=ids,
        )


@dataclass()
class EvaluationResults:
    metrics_with_averages: metrics.al_step_metric_dict
    gathered_outputs: dict[str, dict[str, torch.Tensor]]
    gathered_labels: dict[str, dict[str, torch.Tensor]]
    ids_per_output: dict[str, dict[str, list[str]]]
    all_ids: list[str]


def run_split_evaluation(
    output_generator: Generator[
        model_training_utils.al_dataloader_gathered_predictions
    ],
    output_objects: al_output_objects_as_dict,
    experiment_metrics: metrics.al_metric_record_dict,
    loss_function: Callable,
    device: str,
    missing_ids_per_output: MissingTargetsInfo,
    with_labels: bool = True,
) -> EvaluationResults:
    """
    This function is implemented in a way that allows for the evaluation to be
    split into two parts, evaluation that is performed:

        (1) On the full validation/predict dataset, for example to compute tabular
            metrics such as ROC-AUC and R2 that require the full dataset.
        (2) On a per-batch basis, then averaged across batches, for example
            loss for sequence outputs.

    It would be optimal to just use (1) for all output types, but this becomes
    a (GPU) RAM issue when the e.g. evaluation/prediction dataset is large,
    long sequences, and large embedding sizes are used.

    Note that applying (2) assumes that the same result is obtained when
    averaging across batches as when evaluating on the full dataset. Otherwise,
    the result will be an approximation, or possibly incorrect.
    """
    full_gathered_output_batches_total = []
    full_gathered_label_batches_total = []
    full_gathered_ids_total = []

    batch_computed_metrics = []
    batch_sizes = []

    split_output_objects = split_output_objects_by_eval_type(
        output_objects=output_objects
    )
    split_output_generator = get_split_output_generator(
        output_generator=output_generator,
        output_objects=output_objects,
    )

    for split_outputs in split_output_generator:
        output_compute = split_outputs.output_compute
        target_compute = split_outputs.target_compute
        output_gather = split_outputs.output_gather
        target_gather = split_outputs.target_gather

        full_gathered_output_batches_total.append(output_gather)

        if with_labels:
            full_gathered_label_batches_total.append(target_gather)

        full_gathered_ids_total += split_outputs.ids

        if with_labels and len(split_output_objects.compute) > 0:
            filtered = metrics.filter_missing_outputs_and_labels(
                batch_ids=split_outputs.ids,
                model_outputs=output_compute,
                target_labels=target_compute,
                missing_ids_info=missing_ids_per_output,
                with_labels=with_labels,
            )

            cur_compute_metrics = compute_eval_metrics_wrapper(
                val_outputs=filtered.model_outputs,
                val_target_labels=filtered.target_labels,
                output_objects=split_output_objects.compute,
                experiment_metrics=experiment_metrics,
                loss_function=loss_function,
                device=device,
                val_outputs_for_loss=output_compute,
                val_targets_for_loss=target_compute,
            )
            batch_computed_metrics.append(cur_compute_metrics)
            batch_sizes.append(len(split_outputs.ids))

    computed_metrics_averaged = average_nested_dict_values(
        data=batch_computed_metrics, weights=batch_sizes
    )

    stack_func = model_training_utils.stack_list_of_output_target_dicts
    full_gathered_label_batches_stacked = {}
    if with_labels:
        full_gathered_label_batches_stacked = stack_func(
            list_of_target_batch_dicts=full_gathered_label_batches_total
        )
    full_gathered_output_batches_stacked = stack_func(
        list_of_target_batch_dicts=full_gathered_output_batches_total
    )

    filtered = metrics.filter_missing_outputs_and_labels(
        batch_ids=full_gathered_ids_total,
        model_outputs=full_gathered_output_batches_stacked,
        target_labels=full_gathered_label_batches_stacked,
        missing_ids_info=missing_ids_per_output,
        with_labels=with_labels,
    )

    full_gathered_metrics = {}
    if with_labels and len(split_output_objects.gather) > 0:
        full_gathered_metrics = compute_eval_metrics_wrapper(
            val_outputs=filtered.model_outputs,
            val_target_labels=filtered.target_labels,
            output_objects=split_output_objects.gather,
            experiment_metrics=experiment_metrics,
            loss_function=loss_function,
            device=device,
            val_outputs_for_loss=full_gathered_output_batches_stacked,
            val_targets_for_loss=full_gathered_label_batches_stacked,
        )

    all_metrics = {**computed_metrics_averaged, **full_gathered_metrics}

    ids_per_output = _build_ids_per_output(filtered_values=filtered)

    evaluation_results = EvaluationResults(
        metrics_with_averages=all_metrics,
        gathered_outputs=filtered.model_outputs,
        gathered_labels=filtered.target_labels,
        ids_per_output=ids_per_output,
        all_ids=full_gathered_ids_total,
    )

    return evaluation_results


def _build_ids_per_output(
    filtered_values: metrics.FilteredOutputsAndLabels,
) -> dict[str, dict[str, list[str]]]:
    fv = filtered_values
    ids_per_output = fv.ids
    if ids_per_output is None:
        assert fv.common_ids is not None
        ids_per_output = {}
        for output_name, inner_dict in fv.model_outputs.items():
            if output_name not in ids_per_output:
                ids_per_output[output_name] = {}

            for inner_target_name in inner_dict:
                ids_per_output[output_name][inner_target_name] = fv.common_ids
    else:
        assert fv.common_ids is None
        return ids_per_output

    return ids_per_output


def average_nested_dict_values(data: list[dict[str, Any]], weights: list[int]) -> dict:
    df = pd.json_normalize(data=data)

    weighted_sums = df.multiply(weights, axis=0).sum().to_dict()
    total_weight = sum(weights)

    weighted_averages_flat = {
        key: value / total_weight for key, value in weighted_sums.items()
    }
    averages_nested = unflatten_dict(flat_dict=weighted_averages_flat, separator=".")

    return averages_nested


def unflatten_dict(flat_dict: dict[str, Any], separator=".") -> dict[str, Any]:
    unflattened_dict: dict[str, Any] = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(separator)
        d = unflattened_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    return unflattened_dict


@dataclass()
class SplitOutputObjectsByType:
    compute: al_output_objects_as_dict
    gather: al_output_objects_as_dict


def split_output_objects_by_eval_type(
    output_objects: al_output_objects_as_dict,
) -> SplitOutputObjectsByType:
    split_map = get_split_outputs_map(output_objects=output_objects)
    output_objects_compute = {
        k: i for k, i in output_objects.items() if split_map[k] == "compute"
    }
    output_objects_gather = {
        k: i for k, i in output_objects.items() if split_map[k] == "gather"
    }

    return SplitOutputObjectsByType(
        compute=output_objects_compute,
        gather=output_objects_gather,
    )


def get_split_outputs_map(output_objects: al_output_objects_as_dict) -> dict[str, str]:
    mapping = {}

    for output_name, output_object in output_objects.items():
        match output_object:
            case ComputedTabularOutputInfo() | ComputedSurvivalOutputInfo():
                mapping[output_name] = "gather"
            case (
                ComputedSequenceOutputInfo()
                | ComputedArrayOutputInfo()
                | ComputedImageOutputInfo()
            ):
                mapping[output_name] = "compute"
            case _:
                raise ValueError(f"Unknown output type: {type(output_object)}")

    return mapping


def compute_eval_metrics_wrapper(
    val_outputs: dict[str, dict[str, torch.Tensor]],
    val_target_labels: "al_training_labels_target",
    output_objects: al_output_objects_as_dict,
    experiment_metrics: metrics.al_metric_record_dict,
    loss_function: Callable,
    device: str,
    val_outputs_for_loss: dict[str, dict[str, torch.Tensor]] | None = None,
    val_targets_for_loss: Optional["al_training_labels_target"] = None,
) -> metrics.al_step_metric_dict:
    """
    We have these specific optional values `val_outputs_for_loss` and
    `val_targets_for_loss` as the loss computations assume we have equally
    sized batches and they handle the NaNs themselves.

    Later if we add NaN handling directly to metric calculations, we
    can simplify this again to only accept `val_outputs` and `val_target_labels`.
    """
    val_target_labels = model_training_utils.parse_tabular_target_labels(
        output_objects=output_objects, device=device, labels=val_target_labels
    )

    eval_metrics_dict = metrics.calculate_batch_metrics(
        outputs_as_dict=output_objects,
        outputs=val_outputs,
        labels=val_target_labels,
        mode="val",
        metric_record_dict=experiment_metrics,
    )

    outputs_for_loss = val_outputs_for_loss or val_outputs
    targets_for_loss = val_targets_for_loss or val_target_labels

    val_losses = loss_function(inputs=outputs_for_loss, targets=targets_for_loss)
    val_loss_avg = metrics.aggregate_losses(losses_dict=val_losses)
    eval_metrics_dict_w_loss = metrics.add_loss_to_metrics(
        outputs_as_dict=output_objects,
        losses=val_losses,
        metric_dict=eval_metrics_dict,
    )

    averaging_functions = experiment_metrics["averaging_functions"]
    assert isinstance(averaging_functions, dict)
    eval_metrics_dict_w_averages = metrics.add_multi_task_average_metrics(
        batch_metrics_dict=eval_metrics_dict_w_loss,
        outputs_as_dict=output_objects,
        loss=val_loss_avg.item(),
        performance_average_functions=averaging_functions,
    )

    return eval_metrics_dict_w_averages


def save_tabular_evaluation_results_wrapper(
    val_outputs: dict[str, dict[str, torch.Tensor]],
    val_labels: dict[str, dict[str, torch.Tensor]],
    val_ids: dict[str, dict[str, list[str]]],
    iteration: int,
    experiment: "Experiment",
) -> None:
    for output_name, output_object in experiment.outputs.items():
        output_type = output_object.output_config.output_info.output_type
        if output_type != "tabular":
            continue

        assert isinstance(output_object, ComputedTabularOutputInfo)
        target_columns = output_object.target_columns
        for column_type, list_of_cols_of_this_type in target_columns.items():
            for column_name in list_of_cols_of_this_type:
                cur_sample_output_folder = utils.prepare_sample_output_folder(
                    output_folder=experiment.configs.gc.be.output_folder,
                    column_name=column_name,
                    output_name=output_name,
                    iteration=iteration,
                )

                cur_val_outputs = val_outputs[output_name][column_name]

                if column_type == "cat" and cur_val_outputs.shape[1] == 1:
                    cur_val_outputs = expand_binary_logits(
                        binary_logits=cur_val_outputs
                    )

                cur_val_ids = val_ids[output_name][column_name]

                cur_val_labels = val_labels[output_name][column_name]
                filtered = metrics.filter_tabular_missing_targets(
                    outputs=cur_val_outputs,
                    target_labels=cur_val_labels,
                    ids=cur_val_ids,
                    target_type=column_type,
                )

                to_npy = metrics.general_torch_to_numpy
                cur_val_outputs_np = to_npy(tensor=filtered.model_outputs)

                cur_val_labels_np = to_npy(tensor=filtered.target_labels)
                if column_type == "cat":
                    cur_val_labels_np = cur_val_labels_np.astype(int)

                cur_val_ids = filtered.ids

                target_transformers = output_object.target_transformers

                plot_config = PerformancePlotConfig(
                    val_outputs=cur_val_outputs_np,
                    val_labels=cur_val_labels_np,
                    val_ids=cur_val_ids,
                    iteration=iteration,
                    column_name=column_name,
                    column_type=column_type,
                    target_transformer=target_transformers[column_name],
                    output_folder=cur_sample_output_folder,
                )

                save_evaluation_results(plot_config=plot_config)


@dataclass
class PerformancePlotConfig:
    val_outputs: np.ndarray
    val_labels: np.ndarray
    val_ids: list[str]
    iteration: int
    column_name: str
    column_type: str
    target_transformer: al_label_transformers_object
    output_folder: Path


def save_evaluation_results(
    plot_config: PerformancePlotConfig,
) -> None:
    pc = plot_config

    vf.gen_eval_graphs(plot_config=pc)

    if pc.column_type == "cat":
        save_classification_predictions(
            val_labels=pc.val_labels,
            val_outputs=pc.val_outputs,
            val_ids=pc.val_ids,
            transformer=pc.target_transformer,
            output_folder=pc.output_folder,
        )
    elif pc.column_type == "con":
        scale_and_save_regression_predictions(
            val_labels=pc.val_labels,
            val_outputs=pc.val_outputs,
            val_ids=pc.val_ids,
            transformer=pc.target_transformer,
            output_folder=pc.output_folder,
        )


def save_classification_predictions(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: list[str],
    transformer: LabelEncoder,
    output_folder: Path,
) -> None:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # BCEWithLogitsLoss case
    if val_outputs.shape[1] == 1:
        validation_predictions_total = sigmoid(val_outputs).round().astype(int)
    else:
        validation_predictions_total = val_outputs.argmax(axis=1)

    df_predictions = _parse_valid_classification_predictions(
        val_true=val_labels,
        val_outputs=val_outputs,
        val_classes=transformer.classes_,
        val_predictions=validation_predictions_total,
        ids=np.array(val_ids),
    )

    df_predictions = _inverse_numerical_labels_hook(
        df=df_predictions, target_transformer=transformer
    )
    df_predictions.to_csv(output_folder / "predictions.csv", index=False)


def _parse_valid_classification_predictions(
    val_true: np.ndarray,
    val_predictions: np.ndarray,
    val_outputs: np.ndarray,
    val_classes: Sequence[str],
    ids: np.ndarray,
) -> pd.DataFrame:
    assert len(val_classes) == val_outputs.shape[1]

    columns = ["ID", "True_Label", "Predicted"]
    prediction_classes = [f"Score Class {i}" for i in val_classes]
    columns += prediction_classes

    column_values = [
        ids,
        val_true,
        val_predictions,
    ]

    for i in range(len(prediction_classes)):
        column_values.append(val_outputs[:, i])

    df = pd.DataFrame(columns=columns)

    for col_name, data in zip(columns, column_values, strict=False):
        df[col_name] = data

    return df


def _inverse_numerical_labels_hook(
    df: pd.DataFrame, target_transformer: LabelEncoder
) -> pd.DataFrame:
    for column in ["True_Label", "Predicted"]:
        df[column] = target_transformer.inverse_transform(df[column])

    return df


def scale_and_save_regression_predictions(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: list[str],
    transformer: StandardScaler,
    output_folder: Path,
) -> None:
    val_labels_2d = val_labels.reshape(-1, 1)
    val_outputs_2d = val_outputs.reshape(-1, 1)

    val_labels = transformer.inverse_transform(val_labels_2d).squeeze()
    val_outputs = transformer.inverse_transform(val_outputs_2d).squeeze()

    data = np.array([val_ids, val_labels, val_outputs]).T
    df = pd.DataFrame(data=data, columns=["ID", "Actual", "Predicted"])
    df = df.set_index("ID")

    df.to_csv(output_folder / "regression_predictions.csv", index=True)
