import copy
import csv
import json
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import List, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from human_origins_supervised.models import model_utils
from human_origins_supervised.train_utils.benchmark import benchmark
from human_origins_supervised.train_utils.metric_funcs import (
    select_metric_func,
    get_train_metrics,
)
from human_origins_supervised.visualization import (
    visualization_funcs as vf,
    model_visualization as mv,
)

if TYPE_CHECKING:
    from human_origins_supervised.train import Config

logger = get_logger(__name__)


def check_if_sample_or_end_epoch(epoch: int, sample_interval: int, n_epochs: int):
    if sample_interval:
        condition_1 = epoch % sample_interval == 0
    else:
        condition_1 = False
    condition_2 = epoch == n_epochs

    return condition_1 or condition_2


def get_most_wrong_preds(
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
    val_labels_total,
    val_outputs_total,
    ids_total,
    label_encoder,
    data_folder,
    outfolder,
):
    val_preds_total = val_outputs_total.argmax(axis=1)

    if (val_labels_total != val_preds_total).sum() > 0:
        df_most_wrong = get_most_wrong_preds(
            val_labels_total, val_preds_total, val_outputs_total, np.array(ids_total)
        )

        df_most_wrong = inverse_numerical_labels_hook(df_most_wrong, label_encoder)
        df_most_wrong = anno_meta_hook(df_most_wrong, Path(data_folder))
        df_most_wrong.to_csv(outfolder / "wrong_preds.csv")


def inverse_numerical_labels_hook(
    df: pd.DataFrame, label_encoder: LabelEncoder
) -> pd.DataFrame:

    for column in ["True_Label", "Wrong_Label"]:
        df[column] = label_encoder.inverse_transform(df[column])

    return df


def scale_and_save_regression_preds(
    y_true: np.ndarray,
    y_outp: np.ndarray,
    ids: List[str],
    scaler: StandardScaler,
    outfolder: Path,
) -> None:

    y_true = scaler.inverse_transform(y_true).squeeze()
    y_outp = scaler.inverse_transform(y_outp).squeeze()

    data = np.array([ids, y_true, y_outp]).T
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


def evaluate(engine: Engine, config: "Config", run_folder: Path) -> None:
    """
    A bit hacky how we manually attach metrics here, but that's because we
    don't want to evaluate as a running average (i.e. do it in the step
    function), but rather run over the whole validation dataset as we do
    in this function.
    """
    c = config

    metric_func = select_metric_func(c.cl_args.model_task, c.label_encoder)

    c.model.eval()
    gather_preds = model_utils.gather_pred_outputs_from_dloader
    val_outputs_total, val_labels_total, val_ids_total = gather_preds(
        c.valid_loader, c.model, c.cl_args.device
    )
    c.model.train()

    val_labels_total = model_utils.cast_labels(c.cl_args.model_task, val_labels_total)

    val_loss = c.criterion(val_outputs_total, val_labels_total)
    metric_dict = metric_func(val_outputs_total, val_labels_total, "v")
    metric_dict["v_loss"] = val_loss.item()

    engine.state.metrics.update(metric_dict)

    epoch = engine.state.epoch
    save_eval_results = check_if_sample_or_end_epoch(
        epoch, c.cl_args.sample_interval, c.cl_args.n_epochs
    )
    if save_eval_results:

        sample_outfolder = Path(run_folder, "samples", str(epoch))
        ensure_path_exists(sample_outfolder, is_folder=True)

        val_outputs_total = val_outputs_total.cpu().numpy()
        val_labels_total = val_labels_total.cpu().numpy()

        vf.gen_eval_graphs(
            val_labels=val_labels_total,
            val_outputs=val_outputs_total,
            val_ids_total=val_ids_total,
            outfolder=sample_outfolder,
            encoder=c.label_encoder,
            model_task=c.cl_args.model_task,
        )

        if c.cl_args.model_task == "cls":
            get_most_wrong_wrapper(
                val_labels_total,
                val_outputs_total,
                val_ids_total,
                c.label_encoder,
                c.cl_args.data_folder,
                sample_outfolder,
            )
        else:
            scale_and_save_regression_preds(
                val_labels_total,
                val_outputs_total,
                val_ids_total,
                c.label_encoder,
                sample_outfolder,
            )


def sample(engine: Engine, config: "Config", run_folder: Path) -> None:
    """
    We need to copy the model to avoid affecting the actual model during
    training (e.g. zero-ing out gradients).

    TODO: Refactor this function further – reuse for parts for benchmarking.
    """

    c = config
    args = c.cl_args

    def pre_transform(single_sample, sample_label):
        single_sample = single_sample.to(device=args.device, dtype=torch.float32)

        sample_label = sample_label.to(device=args.device)
        sample_label = model_utils.cast_labels(args.model_task, sample_label)

        return single_sample, sample_label

    epoch = engine.state.epoch
    do_sample = check_if_sample_or_end_epoch(epoch, args.sample_interval, args.n_epochs)
    do_acts = args.get_acts

    if do_sample:
        sample_outfolder = Path(run_folder, "samples", str(epoch))
        ensure_path_exists(sample_outfolder, is_folder=True)

        if do_acts:
            model_copy = copy.deepcopy(c.model)

            no_explainer_background_samples = np.max([int(args.batch_size / 8), 16])
            explainer = mv.get_shap_object(
                model_copy, args.device, c.train_loader, no_explainer_background_samples
            )

            proc_funcs = {"pre": (pre_transform,)}
            act_func = partial(
                mv.get_shap_sample_acts_deep,
                explainer=explainer,
                model_task=args.model_task,
            )

            mv.analyze_activations(config, act_func, proc_funcs, sample_outfolder)


def attach_metrics(engine: Engine, monitoring_metrics: List[str]) -> None:
    """
    For each metric, we crate an output_transform function that grabs the
    target variable from the output of the step function (which is a dict).

    Basically what we attach to the trainer operates on the output of the
    update / step function, that we pass to the Engine definition.

    We use a partial so each lambda has it's own metric variable (otherwise
    they all reference the same object as it gets overwritten).
    """
    for metric in monitoring_metrics:
        partial_func = partial(lambda x, metric_: x[metric_], metric_=metric)
        RunningAverage(output_transform=partial_func).attach(engine, metric)


def log_stats(
    engine: Engine, pbar: ProgressBar, run_folder: Path, run_name: str = None
) -> None:
    log_string = f"[Epoch {engine.state.epoch}/{engine.state.max_epochs}]"

    for name, value in engine.state.metrics.items():
        log_string += f" | {name}: {value:.4f}"

    pbar.log_message(log_string)

    if run_name:
        with open(str(run_folder) + "/training_history.log", "a") as logfile:
            fieldnames = sorted(engine.state.metrics.keys())
            writer = csv.DictWriter(logfile, fieldnames=fieldnames)
            if engine.state.epoch == 1:
                writer.writeheader()
            writer.writerow(engine.state.metrics)


def save_progress(
    engine: Engine, cl_args: Namespace, run_folder: Path, model: nn.Module
) -> None:
    hook_funcs = []

    def plot_benchmark_hook(ax):
        benchmark_file = Path(run_folder, "benchmark/benchmark_metrics.txt")
        with open(str(benchmark_file), "r") as bfile:
            lines = [i.strip() for i in bfile if i.startswith("VAL MCC")]
            value = float(lines[0].split(": ")[-1])

        benchm_line = ax.axhline(
            y=value, linewidth=0.5, color="gray", linestyle="dashed"
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(benchm_line)
        labels.append("LR Benchmark")
        ax.legend(handles, labels)

    if cl_args.benchmark:
        hook_funcs.append(plot_benchmark_hook)

    if check_if_sample_or_end_epoch(
        engine.state.epoch, cl_args.sample_interval, cl_args.n_epochs
    ):
        metrics_file = run_folder + "/training_history.log"
        vf.generate_all_plots(metrics_file, hook_funcs=hook_funcs)

        with open(Path(run_folder, "model_info.txt"), "w") as mfile:
            mfile.write(str(model))


def configure_trainer(trainer: Engine, config: "Config") -> Engine:
    """
    NOTE:
        **Important** the evaluate handler must be attached before the
        ``save_progress`` function, as it manually adds validation metrics
        to the engine state. I.e. we need to make sure they have been
        calculated before calling ``save_progress`` during training.

    TODO:
        Check if there is a better way to address tohe above, e.g. reordering
        the handlers in this func in the end?
    """
    args = config.cl_args

    run_folder = "runs/" + args.run_name

    for handler in evaluate, sample:
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, handler, config=config, run_folder=run_folder
        )

    monitoring_metrics = ["t_loss"] + get_train_metrics(model_task=args.model_task)
    attach_metrics(trainer, monitoring_metrics)

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_stats,
        pbar=pbar,
        run_folder=run_folder,
        run_name=args.run_name,
    )

    if args.run_name:
        checkpoint_handler = ModelCheckpoint(
            Path(run_folder, "saved_models"),
            args.run_name,
            create_dir=True,
            n_saved=100,
            save_interval=args.checkpoint_interval,
            save_as_state_dict=True,
        )

        if args.model_task == "cls":
            np.save(
                Path(run_folder, "saved_models", "classes.npy"),
                config.label_encoder.classes_,
            )

        with open(run_folder + "/run_config.json", "w") as config_file:
            config_dict = vars(args)
            json.dump(config_dict, config_file, sort_keys=True, indent=4)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint_handler, to_save={"model": config.model}
        )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            save_progress,
            cl_args=args,
            run_folder=run_folder,
            model=config.model,
        )

        if args.benchmark:
            trainer.add_event_handler(
                Events.STARTED, benchmark, config=config, run_folder=run_folder
            )

    return trainer
