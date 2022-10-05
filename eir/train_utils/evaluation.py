from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir.data_load.data_utils import get_output_info_generator
from eir.data_load.label_setup import al_label_transformers_object
from eir.models import model_training_utils
from eir.train_utils import metrics
from eir.train_utils import utils
from eir.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from eir.train_utils.train_handlers import HandlerConfig
    from eir.train import Experiment

logger = get_logger(name=__name__, tqdm_compatible=True)


def validation_handler(engine: Engine, handler_config: "HandlerConfig") -> None:
    """
    A bit hacky how we manually attach metrics here, but that's because we
    don't want to evaluate as a running average (i.e. do it in the step
    function), but rather run over the whole validation dataset as we do
    in this function.

    TODO: Streamline with hooks.
    """
    exp = handler_config.experiment
    gc = exp.configs.global_config
    iteration = engine.state.iteration

    exp.model.eval()
    gather_preds = model_training_utils.gather_pred_outputs_from_dloader

    val_outputs_total, val_target_labels, val_ids_total = gather_preds(
        data_loader=exp.valid_loader,
        batch_prep_hook=exp.hooks.step_func_hooks.base_prepare_batch,
        batch_prep_hook_kwargs={"experiment": exp},
        model=exp.model,
        with_labels=True,
    )
    exp.model.train()

    val_target_labels = model_training_utils.parse_target_labels(
        output_objects=exp.outputs, device=gc.device, labels=val_target_labels
    )

    val_losses = exp.loss_function(inputs=val_outputs_total, targets=val_target_labels)
    val_loss_avg = metrics.aggregate_losses(losses_dict=val_losses)

    eval_metrics_dict = metrics.calculate_batch_metrics(
        outputs_as_dict=exp.outputs,
        outputs=val_outputs_total,
        labels=val_target_labels,
        mode="val",
        metric_record_dict=exp.metrics,
    )

    eval_metrics_dict_w_loss = metrics.add_loss_to_metrics(
        outputs_as_dict=exp.outputs,
        losses=val_losses,
        metric_dict=eval_metrics_dict,
    )

    eval_metrics_dict_w_avgs = metrics.add_multi_task_average_metrics(
        batch_metrics_dict=eval_metrics_dict_w_loss,
        outputs_as_dict=exp.outputs,
        loss=val_loss_avg.item(),
        performance_average_functions=exp.metrics["averaging_functions"],
    )

    write_eval_header = True if iteration == gc.sample_interval else False
    metrics.persist_metrics(
        handler_config=handler_config,
        metrics_dict=eval_metrics_dict_w_avgs,
        iteration=iteration,
        write_header=write_eval_header,
        prefixes={"metrics": "validation_", "writer": "validation"},
    )

    if gc.save_evaluation_sample_results:
        save_evaluation_results_wrapper(
            val_outputs=val_outputs_total,
            val_labels=val_target_labels,
            val_ids=val_ids_total,
            iteration=iteration,
            experiment=handler_config.experiment,
        )


def save_evaluation_results_wrapper(
    val_outputs: Dict[str, torch.Tensor],
    val_labels: Dict[str, Dict[str, torch.Tensor]],
    val_ids: List[str],
    iteration: int,
    experiment: "Experiment",
):

    target_columns_gen = get_output_info_generator(outputs_as_dict=experiment.outputs)

    for output_name, column_type, column_name in target_columns_gen:
        cur_sample_outfolder = utils.prep_sample_outfolder(
            output_folder=experiment.configs.global_config.output_folder,
            column_name=column_name,
            output_name=output_name,
            iteration=iteration,
        )

        cur_val_outputs = val_outputs[output_name][column_name]
        cur_val_outputs = cur_val_outputs.cpu().numpy()

        cur_val_labels = val_labels[output_name][column_name]
        cur_val_labels = cur_val_labels.cpu().numpy()

        target_transformers = experiment.outputs[output_name].target_transformers
        plot_config = PerformancePlotConfig(
            val_outputs=cur_val_outputs,
            val_labels=cur_val_labels,
            val_ids=val_ids,
            iteration=iteration,
            column_name=column_name,
            column_type=column_type,
            target_transformer=target_transformers[column_name],
            output_folder=cur_sample_outfolder,
        )

        save_evaluation_results(plot_config=plot_config)


@dataclass
class PerformancePlotConfig:
    val_outputs: np.ndarray
    val_labels: np.ndarray
    val_ids: List[str]
    iteration: int
    column_name: str
    column_type: str
    target_transformer: al_label_transformers_object
    output_folder: Path


def save_evaluation_results(
    plot_config: PerformancePlotConfig,
) -> None:

    pc = plot_config

    common_args = {
        "val_outputs": pc.val_outputs,
        "val_labels": pc.val_labels,
        "val_ids": pc.val_ids,
        "outfolder": pc.output_folder,
        "transformer": pc.target_transformer,
    }

    vf.gen_eval_graphs(plot_config=pc)

    if pc.column_type == "cat":
        save_classification_preds(**common_args)
    elif pc.column_type == "con":
        scale_and_save_regression_preds(**common_args)


def save_classification_preds(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: List[str],
    transformer: LabelEncoder,
    outfolder: Path,
) -> None:
    val_preds_total = val_outputs.argmax(axis=1)

    df_predictions = _parse_valid_classification_preds(
        val_true=val_labels,
        val_outputs=val_outputs,
        val_classes=transformer.classes_,
        val_preds=val_preds_total,
        ids=np.array(val_ids),
    )

    df_predictions = _inverse_numerical_labels_hook(
        df=df_predictions, target_transformer=transformer
    )
    df_predictions.to_csv(outfolder / "predictions.csv", index=False)


def _parse_valid_classification_preds(
    val_true: np.ndarray,
    val_preds: np.ndarray,
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
        val_preds,
    ]

    for i in range(len(prediction_classes)):
        column_values.append(val_outputs[:, i])

    df = pd.DataFrame(columns=columns)

    for col_name, data in zip(columns, column_values):
        df[col_name] = data

    return df


def _inverse_numerical_labels_hook(
    df: pd.DataFrame, target_transformer: LabelEncoder
) -> pd.DataFrame:
    for column in ["True_Label", "Predicted"]:
        df[column] = target_transformer.inverse_transform(df[column])

    return df


def scale_and_save_regression_preds(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: List[str],
    transformer: StandardScaler,
    outfolder: Path,
) -> None:

    val_labels_2d = val_labels.reshape(-1, 1)
    val_outputs_2d = val_outputs.reshape(-1, 1)

    val_labels = transformer.inverse_transform(val_labels_2d).squeeze()
    val_outputs = transformer.inverse_transform(val_outputs_2d).squeeze()

    data = np.array([val_ids, val_labels, val_outputs]).T
    df = pd.DataFrame(data=data, columns=["ID", "Actual", "Predicted"])

    df.to_csv(outfolder / "regression_predictions.csv", index=["ID"])
