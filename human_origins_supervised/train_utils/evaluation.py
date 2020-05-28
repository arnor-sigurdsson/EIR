from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import get_logger
from ignite.engine import Engine
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, StandardScaler

from human_origins_supervised.data_load.data_utils import get_target_columns_generator
from human_origins_supervised.data_load.datasets import al_label_transformers_object
from human_origins_supervised.models import model_utils
from human_origins_supervised.train_utils import metrics
from human_origins_supervised.train_utils import utils
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.train import Config

logger = get_logger(name=__name__, tqdm_compatible=True)


def validation_handler(engine: Engine, handler_config: "HandlerConfig") -> None:
    """
    A bit hacky how we manually attach metrics here, but that's because we
    don't want to evaluate as a running average (i.e. do it in the step
    function), but rather run over the whole validation dataset as we do
    in this function.
    """
    c = handler_config.config
    cl_args = c.cl_args
    iteration = engine.state.iteration

    c.model.eval()
    gather_preds = model_utils.gather_pred_outputs_from_dloader
    val_outputs_total, val_target_labels, val_ids_total = gather_preds(
        data_loader=c.valid_loader,
        cl_args=c.cl_args,
        model=c.model,
        device=cl_args.device,
        with_labels=True,
    )
    c.model.train()

    val_target_labels = model_utils.parse_target_labels(
        target_columns=c.target_columns, device=cl_args.device, labels=val_target_labels
    )

    val_losses = metrics.calculate_losses(
        criterions=c.criterions, labels=val_target_labels, outputs=val_outputs_total
    )
    val_loss_avg = metrics.aggregate_losses(val_losses)

    eval_metrics_dict = metrics.calculate_batch_metrics(
        target_columns=c.target_columns,
        losses=val_losses,
        outputs=val_outputs_total,
        labels=val_target_labels,
        prefix="v_",
        metric_record_dict=c.metrics,
    )

    eval_metrics_dict_w_avgs = metrics.add_multi_task_average_metrics(
        batch_metrics_dict=eval_metrics_dict,
        target_columns=c.target_columns,
        prefix="v_",
        loss=val_loss_avg.item(),
        average_targets=c.metrics["average_targets"],
    )

    write_eval_header = True if iteration == cl_args.sample_interval else False
    metrics.persist_metrics(
        handler_config=handler_config,
        metrics_dict=eval_metrics_dict_w_avgs,
        iteration=iteration,
        write_header=write_eval_header,
        prefixes={"metrics": "v_", "writer": "validation"},
    )

    save_evaluation_results_wrapper(
        val_outputs=val_outputs_total,
        val_labels=val_target_labels,
        val_ids=val_ids_total,
        iteration=iteration,
        config=handler_config.config,
    )


def save_evaluation_results_wrapper(
    val_outputs: Dict[str, torch.Tensor],
    val_labels: Dict[str, torch.Tensor],
    val_ids: List[str],
    iteration: int,
    config: "Config",
):

    target_columns_gen = get_target_columns_generator(config.target_columns)
    transformers = config.target_transformers

    for column_type, column_name in target_columns_gen:
        cur_sample_outfolder = utils.prep_sample_outfolder(
            run_name=config.cl_args.run_name,
            column_name=column_name,
            iteration=iteration,
        )

        cur_val_outputs = val_outputs[column_name].cpu().numpy()
        cur_val_labels = val_labels[column_name].cpu().numpy()

        plot_config = PerformancePlotConfig(
            val_outputs=cur_val_outputs,
            val_labels=cur_val_labels,
            val_ids=val_ids,
            iteration=iteration,
            column_name=column_name,
            column_type=column_type,
            target_transformer=transformers[column_name],
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


def save_evaluation_results(plot_config: PerformancePlotConfig,) -> None:

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
        get_most_wrong_wrapper(**common_args)
    elif pc.column_type == "con":
        scale_and_save_regression_preds(**common_args)


def get_most_wrong_wrapper(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: List[str],
    transformer: LabelEncoder,
    outfolder: Path,
) -> None:
    val_preds_total = val_outputs.argmax(axis=1)

    some_are_incorrect = (val_labels != val_preds_total).sum() > 0
    if some_are_incorrect:
        df_most_wrong = get_most_wrong_cls_preds(
            val_true=val_labels,
            val_preds=val_preds_total,
            val_outputs=val_outputs,
            ids=np.array(val_ids),
        )

        df_most_wrong = _inverse_numerical_labels_hook(
            df=df_most_wrong, target_transformer=transformer
        )
        _check_df_most_wrong(df=df_most_wrong, outfolder=outfolder)
        df_most_wrong.to_csv(outfolder / "wrong_preds.csv")


def _check_df_most_wrong(df: pd.DataFrame, outfolder: Path) -> None:
    try:
        assert_1_string = "True label equal predicted labels."
        assert not (df["True_Label"] == df["Wrong_Label"]).any(), assert_1_string

        assert (df["True_Prob"] < 0.5).all(), "True predicted over 0.5."

    except AssertionError as e:
        logger.error(
            "Got AssertionError ('%s') when checking for probabilities of wrong "
            "predictions. Something might be weird, or a rare event where probabilities"
            "are exactly equal happened. The file is wrong_preds.csv in %s.",
            e,
            outfolder,
        )


def get_most_wrong_cls_preds(
    val_true: np.ndarray,
    val_preds: np.ndarray,
    val_outputs: np.ndarray,
    ids: np.ndarray,
) -> pd.DataFrame:
    wrong_mask = val_preds != val_true
    wrong_indices = np.where(wrong_mask)[0]
    all_wrong_probs = softmax(val_outputs[wrong_indices], axis=1)

    correct_labels_for_misclassified = val_true[wrong_indices]

    correct_label_prob = all_wrong_probs[
        np.arange(wrong_indices.shape[0]), correct_labels_for_misclassified
    ]
    assert correct_label_prob.max() < 0.5

    wrong_pred_labels = val_preds[wrong_indices]
    wrong_label_pred_prob = all_wrong_probs[
        np.arange(wrong_indices.shape[0]), wrong_pred_labels
    ]

    columns = ["Sample_ID", "True_Label", "True_Prob", "Wrong_Label", "Wrong_Prob"]
    column_values = [
        ids[wrong_indices],
        correct_labels_for_misclassified,
        correct_label_prob,
        wrong_pred_labels,
        wrong_label_pred_prob,
    ]

    df = pd.DataFrame(columns=columns)

    for col_name, data in zip(columns, column_values):
        df[col_name] = data

    df = df.sort_values(by=["True_Prob"])
    return df


def _inverse_numerical_labels_hook(
    df: pd.DataFrame, target_transformer: LabelEncoder
) -> pd.DataFrame:
    for column in ["True_Label", "Wrong_Label"]:
        df[column] = target_transformer.inverse_transform(df[column])

    return df


def scale_and_save_regression_preds(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: List[str],
    transformer: StandardScaler,
    outfolder: Path,
) -> None:
    val_labels = transformer.inverse_transform(val_labels).squeeze()
    val_outputs = transformer.inverse_transform(val_outputs).squeeze()

    data = np.array([val_ids, val_labels, val_outputs]).T
    df = pd.DataFrame(data=data, columns=["ID", "Actual", "Predicted"])

    df.to_csv(outfolder / "regression_predictions.csv", index=["ID"])
