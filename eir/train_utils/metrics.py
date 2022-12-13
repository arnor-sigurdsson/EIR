import csv
import warnings
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Dict,
    TYPE_CHECKING,
    List,
    Tuple,
    Callable,
    Union,
    Generator,
    Sequence,
)

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists, get_logger
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.linalg import vector_norm
from torch.utils.tensorboard import SummaryWriter

from eir.data_load.data_utils import get_output_info_generator
from eir.data_load.label_setup import al_label_transformers
from eir.setup.schemas import OutputConfig

if TYPE_CHECKING:
    from eir.train import (
        al_criteria,
        Experiment,
        al_training_labels_target,
    )  # noqa: F401
    from eir.models.omics.omics_models import al_omics_models  # noqa: F401
    from eir.train_utils.train_handlers import HandlerConfig
    from eir.data_load.label_setup import (  # noqa: F401
        al_target_columns,
        al_label_transformers_object,
    )
    from eir.setup.output_setup import al_output_objects_as_dict

# aliases
# output_name -> target_name -> metric name: value
al_step_metric_dict = Dict[str, Dict[str, Dict[str, float]]]
al_metric_record_dict = Dict[
    str, Union[Tuple["MetricRecord", ...], "al_averaging_functions_dict"]
]
al_averaging_functions_dict = Dict[
    str, Callable[["al_step_metric_dict", str, str], float]
]

logger = get_logger(name=__name__, tqdm_compatible=True)


@dataclass()
class MetricRecord:
    name: str
    function: Callable
    only_val: bool = False
    minimize_goal: bool = False


def calculate_batch_metrics(
    outputs_as_dict: "al_output_objects_as_dict",
    outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: "al_training_labels_target",
    mode: str,
    metric_record_dict: al_metric_record_dict,
) -> al_step_metric_dict:
    assert mode in ["val", "train"]

    target_columns_gen = get_output_info_generator(outputs_as_dict=outputs_as_dict)

    master_metric_dict = {}

    for output_name, output_target_type, target_name in target_columns_gen:

        cur_metric_dict = {}

        if output_name not in master_metric_dict:
            master_metric_dict[output_name] = {}

        if output_target_type == "general":
            master_metric_dict[output_name][target_name] = cur_metric_dict

        elif output_target_type in ("con", "cat"):
            cur_output_object = outputs_as_dict[output_name]
            cur_output_type = cur_output_object.output_config.output_info.output_type
            assert cur_output_type == "tabular"

            al_record = Tuple[MetricRecord, ...]
            cur_metric_records: al_record = metric_record_dict[output_target_type]

            cur_outputs = outputs[output_name][target_name]
            cur_outputs = cur_outputs.detach().cpu().to(dtype=torch.float32).numpy()

            cur_labels = labels[output_name][target_name]
            cur_labels = cur_labels.cpu().to(dtype=torch.float32).numpy()

            for metric_record in cur_metric_records:

                if metric_record.only_val and mode == "train":
                    continue

                cur_key = f"{output_name}_{target_name}_{metric_record.name}"
                cur_metric_dict[cur_key] = metric_record.function(
                    outputs=cur_outputs,
                    labels=cur_labels,
                    column_name=target_name,
                    output_name=output_name,
                )

            master_metric_dict[output_name][target_name] = cur_metric_dict
        else:
            raise NotImplementedError()

    return master_metric_dict


def add_loss_to_metrics(
    outputs_as_dict: "al_output_objects_as_dict",
    losses: Dict[str, Dict[str, torch.Tensor]],
    metric_dict: al_step_metric_dict,
) -> al_step_metric_dict:

    target_columns_gen = get_output_info_generator(outputs_as_dict=outputs_as_dict)
    metric_dict_copy = copy(metric_dict)

    for output_name, column_type, column_name in target_columns_gen:
        cur_metric_dict = metric_dict_copy[output_name][column_name]
        cur_key = f"{output_name}_{column_name}_loss"
        cur_metric_dict[cur_key] = losses[output_name][column_name].item()

    return metric_dict_copy


def add_multi_task_average_metrics(
    batch_metrics_dict: al_step_metric_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    loss: float,
    performance_average_functions: Dict[str, Callable[[al_step_metric_dict], float]],
) -> al_step_metric_dict:
    average_performance = average_performances_across_tasks(
        metric_dict=batch_metrics_dict,
        outputs_as_dict=outputs_as_dict,
        performance_calculation_functions=performance_average_functions,
    )
    batch_metrics_dict["average"] = {
        "average": {
            "loss-average": loss,
            "perf-average": average_performance,
        }
    }

    return batch_metrics_dict


def average_performances_across_tasks(
    metric_dict: al_step_metric_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    performance_calculation_functions: Dict[
        str, Callable[[al_step_metric_dict], float]
    ],
) -> float:
    target_columns_gen = get_output_info_generator(outputs_as_dict)

    all_metrics = []

    for output_name, column_type, column_name in target_columns_gen:
        cur_metric_func = performance_calculation_functions.get(column_type)

        metric_func_args = {
            "metric_dict": metric_dict,
            "output_name": output_name,
            "column_name": column_name,
        }
        cur_value = cur_metric_func(**metric_func_args)
        all_metrics.append(cur_value)

        all_metrics.append(cur_value)

    average = np.array(all_metrics).mean()

    return average


def calc_mcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(a=outputs, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mcc = matthews_corrcoef(labels, pred)

    return mcc


def calc_roc_auc_ovo(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:
    """
    TODO:   In rare scenarios, we might run into the issue of not having all labels
            represented in the labels array (i.e. labels were in train, but not in
            valid). This is not a problem for metrics like MCC / accuracy, but we
            will have to account for this here and in the AP calculation, possibly
            by ignoring columns in outputs and label_binarize outputs where the columns
            returned from label_binarize are all 0.
    """

    assert average in ["micro", "macro"]

    if outputs.shape[1] > 2:
        outputs = softmax(x=outputs, axis=1)
    else:
        outputs = outputs[:, 1]

    roc_auc = roc_auc_score(
        y_true=labels, y_score=outputs, average=average, multi_class="ovo"
    )
    return roc_auc


def calc_average_precision(
    outputs: np.ndarray, labels: np.ndarray, average: str = "macro", *args, **kwargs
) -> float:

    assert average in ["micro", "macro"]

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    else:
        outputs = outputs[:, 1]

    average_precision = average_precision_score(
        y_true=labels, y_score=outputs, average=average
    )

    return average_precision


def calc_acc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(outputs, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return accuracy


def calc_pcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    pcc = pearsonr(x=labels.squeeze(), y=outputs.squeeze())[0]
    return pcc


def calc_r2(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:

    if len(outputs) < 2:
        return 0.0

    r2 = r2_score(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return r2


def calc_rmse(
    outputs: np.ndarray,
    labels: np.ndarray,
    target_transformers: Dict[str, Dict[str, StandardScaler]],
    output_name: str,
    column_name: str,
    *args,
    **kwargs,
) -> float:
    cur_target_transformer = target_transformers[output_name][column_name]

    labels_2d = labels.reshape(-1, 1)
    outputs_2d = outputs.reshape(-1, 1)

    labels = cur_target_transformer.inverse_transform(labels_2d).squeeze()
    preds = cur_target_transformer.inverse_transform(outputs_2d).squeeze()

    rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=preds))
    return rmse


def calculate_prediction_losses(
    criteria: "al_criteria",
    inputs: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Inputs here refers to the input to the loss functions, which is the output
    from the actual models.
    """
    losses_dict = {}

    for output_name, target_criterion_dict in criteria.items():

        for target_column, criterion in target_criterion_dict.items():
            cur_target_col_labels = targets[output_name][target_column]
            cur_target_col_outputs = inputs[output_name][target_column]

            if output_name not in losses_dict:
                losses_dict[output_name] = {}

            losses_dict[output_name][target_column] = criterion(
                input=cur_target_col_outputs, target=cur_target_col_labels
            )

    return losses_dict


def aggregate_losses(losses_dict: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    losses_values = []
    for output_name, targets_for_output_dict in losses_dict.items():
        losses_values += list(targets_for_output_dict.values())
    average_loss = torch.mean(torch.stack(losses_values))

    return average_loss


def get_uncertainty_loss_hook(
    output_configs: Sequence[OutputConfig],
    device: str,
):
    uncertainty_loss_modules = {}
    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            continue

        if not output_config.output_type_info.uncertainty_weighted_mt_loss:
            continue

        logger.info(
            f"Adding uncertainty loss for {output_config.output_info.output_name}."
        )

        output_type_info = output_config.output_type_info
        cur_target_cat_columns = list(output_type_info.target_cat_columns)
        cur_target_con_columns = list(output_type_info.target_con_columns)
        loss_module = UncertaintyMultiTaskLoss(
            target_cat_columns=cur_target_cat_columns,
            target_con_columns=cur_target_con_columns,
            device=device,
        )
        uncertainty_loss_modules[output_config.output_info.output_name] = loss_module

    if len(uncertainty_loss_modules) == 0:
        raise ValueError("Expected at least one uncertainty loss module.")

    hook = partial(
        hook_add_uncertainty_loss, uncertainty_modules=uncertainty_loss_modules
    )

    return hook


def hook_add_uncertainty_loss(
    state: Dict,
    uncertainty_modules: Dict[str, "UncertaintyMultiTaskLoss"],
    loss_key: str = "per_target_train_losses",
    *args,
    **kwargs,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Note that we only update the relevant losses in the base dict.
    """

    base_losses_dict = state[loss_key]
    updated_losses = copy(base_losses_dict)

    for output_name, module in uncertainty_modules.items():
        cur_module = uncertainty_modules[output_name]
        cur_loss_dict = base_losses_dict[output_name]
        cur_uncertainty_losses = cur_module(losses_dict=cur_loss_dict)
        updated_losses[output_name] = cur_uncertainty_losses

    state_updates = {loss_key: updated_losses}

    return state_updates


class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(
        self,
        target_cat_columns: List[str],
        target_con_columns: List[str],
        device: str,
    ):
        super().__init__()

        self.target_cat_columns = target_cat_columns
        self.target_con_columns = target_con_columns
        self.device = device

        self.log_vars = self._construct_params(
            cur_target_columns=self.target_cat_columns + self.target_con_columns,
            device=self.device,
        )

    @staticmethod
    def _construct_params(
        cur_target_columns: List[str], device: str
    ) -> Dict[str, nn.Parameter]:

        param_dict = {}
        for column_name in cur_target_columns:
            cur_param = nn.Parameter(torch.zeros(1), requires_grad=True).to(
                device=device
            )
            param_dict[column_name] = cur_param

        return param_dict

    def _calc_uncertainty_loss(
        self, name: str, loss_value: torch.Tensor
    ) -> torch.Tensor:
        log_var = self.log_vars[name]
        scalar = 2.0 if name in self.target_cat_columns else 1.0

        precision = torch.exp(-log_var)
        loss = scalar * torch.sum(precision * loss_value + log_var)

        return loss

    def forward(self, losses_dict: Dict) -> Dict[str, torch.Tensor]:
        losses_uncertain = {}

        for target_column, loss_value_base in losses_dict.items():
            loss_value_uncertain = self._calc_uncertainty_loss(
                name=target_column, loss_value=loss_value_base
            )
            losses_uncertain[target_column] = loss_value_uncertain

        return losses_uncertain


def hook_add_l1_loss(
    experiment: "Experiment",
    state: Dict,
    loss_key: str = "loss",
    *args,
    **kwargs,
) -> Dict:
    """
    TODO: Do the validation outside of the actual hook.
    TODO: Use a combined iterator here for the fusion module and modules_to_fuse.
    TODO: Convert all fusion modules to have their own L1 penalized weights.
    """
    model_configs = experiment.inputs

    l1_loss = torch.tensor(0.0, device=experiment.configs.global_config.device)
    for input_name, input_module in experiment.model.input_modules.items():

        cur_model_config = model_configs[input_name].input_config.model_config
        cur_model_init_config = cur_model_config.model_init_config

        current_l1 = getattr(cur_model_init_config, "l1", None)
        has_l1_weights = hasattr(input_module, "l1_penalized_weights")

        if current_l1 and not has_l1_weights:
            raise AttributeError(
                f"Module {input_module} for input name {input_name} does not have"
                f"l1_penalized_weights attribute."
            )

        if has_l1_weights and current_l1:
            cur_l1_loss = get_model_l1_loss(model=input_module, l1_weight=current_l1)
            l1_loss += cur_l1_loss

    fus_l1 = getattr(experiment.model.fusion_module.model_config, "l1", None)
    fus_has_l1_weights = hasattr(experiment.model, "l1_penalized_weights")

    if fus_l1 and fus_has_l1_weights:
        fusion_l1_loss = get_model_l1_loss(model=experiment.model, l1_weight=fus_l1)
        l1_loss += fusion_l1_loss

    updated_loss = state[loss_key] + l1_loss

    state_updates = {loss_key: updated_loss}

    return state_updates


def get_model_l1_loss(model: nn.Module, l1_weight: float) -> torch.Tensor:
    l1_loss = calc_l1_loss(
        weight_tensor=model.l1_penalized_weights, l1_weight=l1_weight
    )
    return l1_loss


def calc_l1_loss(weight_tensor: torch.Tensor, l1_weight: float):
    l1_loss = vector_norm(weight_tensor, ord=1, dim=None) * l1_weight
    return l1_loss


def add_extra_losses(total_loss: torch.Tensor, extra_loss_functions: List[Callable]):
    """
    TODO: Possibly add inputs and labels as arguments here if needed later.
    """
    for loss_func in extra_loss_functions:
        total_loss += loss_func()

    return total_loss


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: Dict[str, str],
    writer_funcs: Union[None, Dict[str, Dict[str, Callable]]] = None,
):

    hc = handler_config
    exp = handler_config.experiment
    gc = exp.configs.global_config

    target_generator = get_output_info_generator(outputs_as_dict=exp.outputs)

    metrics_files = get_metrics_files(
        target_generator=target_generator,
        run_folder=hc.run_folder,
        train_or_val_target_prefix=f"{prefixes['metrics']}",
    )

    if write_header:
        _ensure_metrics_paths_exists(metrics_files=metrics_files)

    for output_name, target_and_file_dict in metrics_files.items():
        for target_name, target_history_file in target_and_file_dict.items():

            cur_metric_dict = metrics_dict[output_name][target_name]

            _add_metrics_to_writer(
                name=f"{prefixes['writer']}/{target_name}",
                metric_dict=cur_metric_dict,
                iteration=iteration,
                writer=exp.writer,
                plot_skip_steps=gc.plot_skip_steps,
            )

            cur_func = get_buffered_metrics_writer(buffer_interval=1)
            if writer_funcs:
                cur_func = writer_funcs[output_name][target_name]

            cur_func(
                filepath=target_history_file,
                metrics=cur_metric_dict,
                iteration=iteration,
                write_header=write_header,
            )


def get_metrics_files(
    target_generator: Generator[Tuple[str, str, str], None, None],
    run_folder: Path,
    train_or_val_target_prefix: str,
) -> Dict[str, Dict[str, Path]]:
    assert train_or_val_target_prefix in ["validation_", "train_"]

    path_dict = {}
    for output_name, column_type, target_column in target_generator:
        if output_name not in path_dict:
            path_dict[output_name] = {}

        cur_file_name = train_or_val_target_prefix + target_column + "_history.log"
        cur_path = Path(
            run_folder, "results", output_name, target_column, cur_file_name
        )
        path_dict[output_name][target_column] = cur_path

    average_loss_training_metrics_file = get_average_history_filepath(
        run_folder=run_folder, train_or_val_target_prefix=train_or_val_target_prefix
    )
    path_dict["average"] = {"average": average_loss_training_metrics_file}

    return path_dict


def get_average_history_filepath(
    run_folder: Path, train_or_val_target_prefix: str
) -> Path:
    assert train_or_val_target_prefix in ["validation_", "train_"]
    metrics_file_path = run_folder / f"{train_or_val_target_prefix}average_history.log"
    return metrics_file_path


def _ensure_metrics_paths_exists(metrics_files: Dict[str, Dict[str, Path]]) -> None:
    for output_name, target_and_file_dict in metrics_files.items():
        for path in target_and_file_dict.values():
            ensure_path_exists(path=path)


def _add_metrics_to_writer(
    name: str,
    metric_dict: Dict[str, float],
    iteration: int,
    writer: SummaryWriter,
    plot_skip_steps: int,
) -> None:
    """
    We do %10 to reduce the amount of training data going to tensorboard, otherwise
    it slows down with many large experiments.
    """
    if iteration >= plot_skip_steps and iteration % 10 == 0:
        for metric_name, metric_value in metric_dict.items():
            cur_name = name + f"/{metric_name}"
            writer.add_scalar(
                tag=cur_name, scalar_value=metric_value, global_step=iteration
            )


def get_buffered_metrics_writer(buffer_interval: int):
    buffer = []

    def append_metrics_to_file(
        filepath: Path, metrics: Dict[str, float], iteration: int, write_header=False
    ):

        nonlocal buffer

        dict_to_write = {**{"iteration": iteration}, **metrics}

        if write_header:
            with open(str(filepath), "a") as logfile:
                fieldnames = ["iteration"] + sorted(metrics.keys())
                writer = csv.DictWriter(logfile, fieldnames=fieldnames)

                if write_header:
                    writer.writeheader()

        if iteration % buffer_interval == 0:

            buffer.append(dict_to_write)

            with open(str(filepath), "a") as logfile:
                fieldnames = ["iteration"] + sorted(metrics.keys())
                writer = csv.DictWriter(logfile, fieldnames=fieldnames)

                source = buffer if buffer_interval != 1 else [dict_to_write]
                for row in source:
                    writer.writerow(row)

            buffer = []

        else:
            buffer.append(dict_to_write)

    return append_metrics_to_file


def read_metrics_history_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="iteration")

    return df


def get_metrics_dataframes(
    results_dir: Path, target_string: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_history_path = read_metrics_history_file(
        results_dir / f"train_{target_string}_history.log"
    )
    valid_history_path = read_metrics_history_file(
        results_dir / f"validation_{target_string}_history.log"
    )

    return train_history_path, valid_history_path


def get_default_metrics(
    target_transformers: Dict[str, al_label_transformers],
) -> "al_metric_record_dict":
    mcc = MetricRecord(name="mcc", function=calc_mcc)
    acc = MetricRecord(name="acc", function=calc_acc)

    rmse = MetricRecord(
        name="rmse",
        function=partial(calc_rmse, target_transformers=target_transformers),
        minimize_goal=True,
    )

    roc_auc_macro = MetricRecord(
        name="roc-auc-macro", function=calc_roc_auc_ovo, only_val=True
    )
    ap_macro = MetricRecord(
        name="ap-macro", function=calc_average_precision, only_val=True
    )
    r2 = MetricRecord(name="r2", function=calc_r2, only_val=True)
    pcc = MetricRecord(name="pcc", function=calc_pcc, only_val=True)

    averaging_functions = get_default_performance_averaging_functions(
        cat_metric_name="mcc", con_metric_name="loss"
    )
    default_metrics = {
        "cat": (mcc, acc, roc_auc_macro, ap_macro),
        "con": (rmse, r2, pcc),
        "averaging_functions": averaging_functions,
    }
    return default_metrics


def get_default_performance_averaging_functions(
    cat_metric_name: str, con_metric_name: str
) -> al_averaging_functions_dict:

    logger.info(
        "Default performance averaging functions across tasks set to %s for "
        "categorical targets and %s for continuous targets.",
        cat_metric_name.upper(),
        con_metric_name.upper(),
    )

    def _calc_cat_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_name: str,
    ) -> float:
        combined_key = f"{output_name}_{column_name}_{metric_name}"
        value = metric_dict[output_name][column_name][combined_key]
        return value

    def _calc_con_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_name: str,
    ) -> float:
        combined_key = f"{output_name}_{column_name}_{metric_name}"
        value = 1.0 - metric_dict[output_name][column_name][combined_key]
        return value

    performance_averaging_functions = {
        "cat": partial(_calc_cat_averaging_value, metric_name=cat_metric_name),
        "con": partial(_calc_con_averaging_value, metric_name=con_metric_name),
        "general": partial(_calc_con_averaging_value, metric_name="loss"),
    }

    return performance_averaging_functions
