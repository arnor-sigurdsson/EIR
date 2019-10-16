from pathlib import Path
from typing import List, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from ignite.engine import Engine
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, StandardScaler

from human_origins_supervised.models import model_utils
from human_origins_supervised.train_utils.metric_funcs import (
    get_train_metrics,
    select_metric_func,
)
from human_origins_supervised.train_utils.utils import check_if_iteration_sample
from human_origins_supervised.visualization import visualization_funcs as vf

if TYPE_CHECKING:
    from human_origins_supervised.train_utils.train_handlers import HandlerConfig
    from human_origins_supervised.train import Config

logger = get_logger(__name__)


def evaluation_handler(engine: Engine, handler_config: "HandlerConfig") -> None:
    """
    A bit hacky how we manually attach metrics here, but that's because we
    don't want to evaluate as a running average (i.e. do it in the step
    function), but rather run over the whole validation dataset as we do
    in this function.
    """
    c = handler_config.config
    args = c.cl_args
    iteration = engine.state.iteration

    n_iters_per_epoch = len(c.train_loader)
    do_eval = check_if_iteration_sample(
        iteration, args.sample_interval, n_iters_per_epoch, args.n_epochs
    )
    train_metrics = get_train_metrics(args.model_task, prefix="v")

    # we update here with NaNs as the metrics are written out in each iteration to
    # training_history.log, so the iterations match when plotting later
    if not do_eval:
        placeholder_metrics = {metric: np.nan for metric in train_metrics}
        placeholder_metrics["v_loss"] = np.nan
        engine.state.metrics.update(placeholder_metrics)
        return

    metric_func = select_metric_func(args.model_task, c.label_encoder)

    c.model.eval()
    gather_preds = model_utils.gather_pred_outputs_from_dloader
    val_outputs_total, val_labels_total, val_ids_total = gather_preds(
        c.valid_loader, c.cl_args, c.model, args.device, c.valid_dataset.labels_dict
    )
    c.model.train()

    val_labels_total = model_utils.cast_labels(args.model_task, val_labels_total)

    val_loss = c.criterion(val_outputs_total, val_labels_total)
    metric_dict = metric_func(val_outputs_total, val_labels_total, "v")
    metric_dict["v_loss"] = val_loss.item()

    engine.state.metrics.update(metric_dict)

    save_evaluation_results(
        val_outputs=val_outputs_total,
        val_labels=val_labels_total,
        val_ids=val_ids_total,
        iteration=iteration,
        run_folder=handler_config.run_folder,
        config=handler_config.config,
    )


def get_most_wrong_cls_preds(
    val_true: np.ndarray,
    val_preds: np.ndarray,
    val_outputs: np.ndarray,
    ids: np.ndarray,
) -> pd.DataFrame:
    wrong_mask = val_preds != val_true
    wrong_indices = np.where(wrong_mask)[0]
    all_probs = softmax(val_outputs[wrong_indices], axis=1)

    correct_labels_for_misclassified = val_true[wrong_indices]
    assert (correct_labels_for_misclassified == val_preds[wrong_indices]).sum() == 0

    # select prob model gave for correct class
    correct_label_prob = all_probs[
        np.arange(wrong_indices.shape[0]), correct_labels_for_misclassified
    ]
    assert correct_label_prob.max() < 0.5

    wrong_pred_labels = val_preds[wrong_indices]
    wrong_label_pred_prob = all_probs[
        np.arange(wrong_indices.shape[0]), wrong_pred_labels
    ]
    assert (wrong_label_pred_prob > (1 / len(np.unique(val_true)))).all()

    columns = ["Sample_ID", "True_Label", "True_Prob", "Wrong_Label", "Wrong_Prob"]
    df = pd.DataFrame(columns=columns)

    for col_name, data in zip(
        columns,
        [
            ids[wrong_indices],
            correct_labels_for_misclassified,
            correct_label_prob,
            wrong_pred_labels,
            wrong_label_pred_prob,
        ],
    ):
        df[col_name] = data

    df = df.sort_values(by=["True_Prob"])
    return df


def get_most_wrong_wrapper(
    val_labels, val_outputs, val_ids, encoder, data_folder, outfolder
):
    val_preds_total = val_outputs.argmax(axis=1)

    if (val_labels != val_preds_total).sum() > 0:
        df_most_wrong = get_most_wrong_cls_preds(
            val_labels, val_preds_total, val_outputs, np.array(val_ids)
        )

        df_most_wrong = inverse_numerical_labels_hook(df_most_wrong, encoder)
        df_most_wrong = anno_meta_hook(df_most_wrong, Path(data_folder))
        df_most_wrong.to_csv(outfolder / "wrong_preds.csv")


def inverse_numerical_labels_hook(
    df: pd.DataFrame, label_encoder: LabelEncoder
) -> pd.DataFrame:
    for column in ["True_Label", "Wrong_Label"]:
        df[column] = label_encoder.inverse_transform(df[column])

    return df


def scale_and_save_regression_preds(
    val_labels: np.ndarray,
    val_outputs: np.ndarray,
    val_ids: List[str],
    encoder: StandardScaler,
    outfolder: Path,
) -> None:
    val_labels = encoder.inverse_transform(val_labels).squeeze()
    val_outputs = encoder.inverse_transform(val_outputs).squeeze()

    data = np.array([val_ids, val_labels, val_outputs]).T
    df = pd.DataFrame(data=data, columns=["ID", "Actual", "Predicted"])

    df.to_csv(outfolder / "regression_predictions.csv", index=["ID"])


def anno_meta_hook(
    df: pd.DataFrame,
    data_folder: Union[None, Path] = None,
    anno_fpath: Union[Path, str] = "infer",
) -> pd.DataFrame:
    df["Sample_ID"] = df["Sample_ID"].map(lambda x: x.split("_-_")[0])

    type_ = "command line argument"
    if anno_fpath == "infer":
        if not data_folder:
            raise ValueError(
                "Expected data folder to be passed in if inferring about"
                "anno filepath."
            )

        data_folder_name = data_folder.parts[1]
        anno_fpath = Path(f"data/{data_folder_name}/raw/data.anno")
        type_ = "inferred"

    if not anno_fpath.exists():
        logger.error(
            "Could not find %s anno file at %s. Skipping meta info hook in wrong "
            "prediction analysis.",
            type_,
            anno_fpath,
        )
        return df

    anno_columns = ["Instance ID", "Group Label", "Location", "Country"]
    try:
        df_anno = pd.read_csv(anno_fpath, usecols=anno_columns, sep="\t")
    except ValueError:
        logger.error(f"Could not read column names in {anno_fpath}, skipping.")
        return df

    df_merged = pd.merge(
        df, df_anno, how="left", left_on="Sample_ID", right_on="Instance ID"
    )
    df_merged = df_merged.drop("Instance ID", axis=1)
    df_merged.set_index("Sample_ID")

    return df_merged


def save_evaluation_results(
    val_outputs: torch.Tensor,
    val_labels: torch.Tensor,
    val_ids: List[str],
    iteration: int,
    run_folder: Path,
    config: "Config",
) -> None:
    args = config.cl_args

    sample_outfolder = Path(run_folder, "samples", str(iteration))
    ensure_path_exists(sample_outfolder, is_folder=True)

    val_outputs = val_outputs.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    common_args = {
        "val_outputs": val_outputs,
        "val_labels": val_labels,
        "val_ids": val_ids,
        "outfolder": sample_outfolder,
        "encoder": config.label_encoder,
    }

    vf.gen_eval_graphs(model_task=args.model_task, **common_args)

    if args.model_task == "cls":
        get_most_wrong_wrapper(data_folder=args.data_folder, **common_args)
    elif args.model_task == "reg":
        scale_and_save_regression_preds(**common_args)
