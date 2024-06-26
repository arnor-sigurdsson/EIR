import csv
import warnings
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from statistics import mean
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.linalg import vector_norm

from eir.data_load.data_utils import get_output_info_generator
from eir.setup.schema_modules.output_schemas_tabular import TabularOutputTypeConfig
from eir.setup.schemas import OutputConfig
from eir.target_setup.target_label_setup import MissingTargetsInfo
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.data_load.label_setup import (  # noqa: F401
        al_label_transformers,
        al_label_transformers_object,
        al_target_columns,
    )
    from eir.models.input.omics.omics_models import al_omics_models  # noqa: F401
    from eir.models.meta.meta_utils import FeatureExtractorProtocolWithL1
    from eir.setup.output_setup import al_output_objects_as_dict
    from eir.train import Experiment, al_criteria_dict  # noqa: F401
    from eir.train_utils.step_logic import al_training_labels_target
    from eir.train_utils.train_handlers import HandlerConfig

# aliases
# output_name -> target_name -> metric name: value
al_step_metric_dict = Dict[str, Dict[str, Dict[str, float]]]

logger = get_logger(name=__name__)


class MetricFunctionProtocol(Protocol):
    def __call__(
        self,
        outputs: np.ndarray,
        labels: np.ndarray,
        column_name: str,
        output_name: str,
    ) -> float: ...


class AverageMetricFunctionProtocol(Protocol):
    def __call__(
        self,
        metric_dict: al_step_metric_dict,
        output_name: str,
        column_name: str,
    ) -> float: ...


al_averaging_functions_dict = Dict[str, AverageMetricFunctionProtocol]
al_metric_record_dict = Dict[
    str, Tuple["MetricRecord", ...] | "al_averaging_functions_dict"
]

al_cat_averaging_metric_choices = Sequence[
    Literal["mcc", "acc", "roc-auc-macro", "ap-macro"]
]
al_con_averaging_metric_choices = Sequence[Literal["r2", "pcc", "loss"]]


@dataclass()
class MetricRecord:
    name: str
    function: MetricFunctionProtocol
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

    master_metric_dict: al_step_metric_dict = {}

    for output_name, output_target_type, target_name in target_columns_gen:
        cur_metric_dict: dict[str, float] = {}

        if output_name not in master_metric_dict:
            master_metric_dict[output_name] = {}

        if output_target_type == "general":
            master_metric_dict[output_name][target_name] = cur_metric_dict

        elif output_target_type in ("con", "cat"):
            cur_output_object = outputs_as_dict[output_name]
            cur_output_type = cur_output_object.output_config.output_info.output_type
            assert cur_output_type == "tabular"

            al_record = Tuple[MetricRecord, ...]
            cur_record = metric_record_dict[output_target_type]
            assert isinstance(cur_record, tuple)
            cur_metric_records: al_record = cur_record

            cur_outputs = outputs[output_name][target_name]
            cur_outputs_np = cur_outputs.detach().cpu().to(dtype=torch.float32).numpy()

            cur_labels = labels[output_name][target_name]
            cur_labels_np = cur_labels.cpu().to(dtype=torch.float32).numpy()

            for metric_record in cur_metric_records:
                if metric_record.only_val and mode == "train":
                    continue

                cur_key = f"{output_name}_{target_name}_{metric_record.name}"
                cur_value = metric_record.function(
                    outputs=cur_outputs_np,
                    labels=cur_labels_np,
                    column_name=target_name,
                    output_name=output_name,
                )

                cur_metric_dict[cur_key] = cur_value

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
    performance_average_functions: Optional["al_averaging_functions_dict"],
) -> al_step_metric_dict:
    average = {
        "average": {
            "loss-average": loss,
        }
    }

    if performance_average_functions is not None:
        average_performance = average_performances_across_tasks(
            metric_dict=batch_metrics_dict,
            outputs_as_dict=outputs_as_dict,
            performance_calculation_functions=performance_average_functions,
        )
        average["average"]["perf-average"] = average_performance

    batch_metrics_dict["average"] = average

    return batch_metrics_dict


def average_performances_across_tasks(
    metric_dict: al_step_metric_dict,
    outputs_as_dict: "al_output_objects_as_dict",
    performance_calculation_functions: "al_averaging_functions_dict",
) -> float:
    target_columns_gen = get_output_info_generator(
        outputs_as_dict=outputs_as_dict,
    )

    all_metrics = []

    for output_name, column_type, column_name in target_columns_gen:
        cur_metric_func = performance_calculation_functions[column_type]

        cur_value = cur_metric_func(
            metric_dict=metric_dict, output_name=output_name, column_name=column_name
        )

        if not np.isnan(cur_value):
            all_metrics.append(cur_value)

    average = np.array(all_metrics).mean()

    return average


def handle_empty(default_value: float, metric_name: Optional[str] = None):
    """
    This can happen when modelling on multiple outputs, which can vary in their
    sparsity, and by chance some outputs are empty in a batch.
    """

    def decorator(func):
        logged = False

        @wraps(func)
        def wrapper(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs):
            nonlocal logged
            if (outputs.size == 0 or labels.size == 0) and not logged:
                logger.info(
                    f"Empty inputs encountered in "
                    f"{metric_name or func.__name__}, "
                    f"returning default value: {default_value}"
                )
                logged = True
            if outputs.size == 0 or labels.size == 0:
                return default_value
            return func(outputs, labels, *args, **kwargs)

        return wrapper

    return decorator


def handle_class_mismatch(default_value: float, metric_name: Optional[str] = None):
    """
    Decorator to handle cases where the number of unique classes in 'labels'
    does not match the number of columns in 'outputs'. This scenario can occur
    in multi-class classification tasks with sparse or imbalanced outputs.
    """

    def decorator(func):
        logged = False

        @wraps(func)
        def wrapper(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs):
            nonlocal logged
            unique_classes = np.unique(labels)
            # For binary classification we have this special case
            if outputs.shape[1] == 2 and len(unique_classes) == 1:
                pass
            elif len(unique_classes) != outputs.shape[1]:
                if not logged:
                    logger.info(
                        f"Class mismatch encountered in "
                        f"{metric_name or func.__name__}, "
                        f"expected number of classes: {outputs.shape[1]}, "
                        f"found unique classes in labels: {len(unique_classes)}. "
                        f"Returning default value: {default_value}"
                    )
                    logged = True
                return default_value
            return func(outputs, labels, *args, **kwargs)

        return wrapper

    return decorator


@handle_empty(default_value=np.nan, metric_name="MCC")
def calc_mcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    prediction = np.argmax(a=outputs, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        mcc = matthews_corrcoef(labels, prediction)

    return mcc


@handle_class_mismatch(default_value=np.nan, metric_name="ROC-AUC")
@handle_empty(default_value=np.nan, metric_name="ROC-AUC")
def calc_roc_auc_ovo(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    """
    TODO:   In rare scenarios, we might run into the issue of not having all labels
            represented in the labels array (i.e. labels were in train, but not in
            valid). This is not a problem for metrics like MCC / accuracy, but we
            will have to account for this here and in the AP calculation, possibly
            by ignoring columns in outputs and label_binarize outputs where the columns
            returned from label_binarize are all 0.
    """

    average = "macro"

    if outputs.shape[1] > 2:
        outputs = softmax(x=outputs, axis=1)
    else:
        outputs = outputs[:, 1]

    roc_auc = roc_auc_score(
        y_true=labels,
        y_score=outputs,
        average=average,
        multi_class="ovo",
    )
    return roc_auc


@handle_class_mismatch(default_value=np.nan, metric_name="AP")
@handle_empty(default_value=np.nan, metric_name="AP")
def calc_average_precision(
    outputs: np.ndarray, labels: np.ndarray, *args, **kwargs
) -> float:
    average = "macro"

    if outputs.shape[1] > 2:
        labels = label_binarize(y=labels, classes=sorted(np.unique(labels)))
    else:
        outputs = outputs[:, 1]

    average_precision = average_precision_score(
        y_true=labels,
        y_score=outputs,
        average=average,
    )

    return average_precision


@handle_empty(default_value=np.nan, metric_name="ACC")
def calc_acc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    pred = np.argmax(outputs, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return accuracy


@handle_empty(default_value=np.nan, metric_name="PCC")
def calc_pcc(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    if len(outputs) < 2:
        return 0.0

    pcc = pearsonr(x=labels.squeeze(), y=outputs.squeeze())[0]
    return pcc


@handle_empty(default_value=np.nan, metric_name="R2")
def calc_r2(outputs: np.ndarray, labels: np.ndarray, *args, **kwargs) -> float:
    if len(outputs) < 2:
        return 0.0

    r2 = r2_score(y_true=labels.squeeze(), y_pred=outputs.squeeze())
    return r2


@handle_empty(default_value=np.nan, metric_name="RMSE")
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

    mean_ = cur_target_transformer.mean_
    scale_ = cur_target_transformer.scale_

    labels = (labels_2d * scale_ + mean_).squeeze()
    predictions = (outputs_2d * scale_ + mean_).squeeze()

    if np.shape(labels):
        rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    else:
        rmse = np.sqrt((labels - predictions) ** 2)

    return rmse


class LogEmptyLossProtocol(Protocol):
    def __call__(self, output_name: str, output_head_name: str) -> None: ...


def log_empty_loss_once() -> LogEmptyLossProtocol:
    logged_combinations = set()

    def log(output_name: str, output_head_name: str) -> None:
        if (output_name, output_head_name) not in logged_combinations:
            logger.info(
                f"Empty output batch encountered for {output_name},"
                f" {output_head_name}; "
                f"setting loss to NaN for this batch and future empty batches. "
                f"Empty batches will not be used for training."
            )
            logged_combinations.add((output_name, output_head_name))

    return log


def calculate_prediction_losses(
    criteria: "Dict[str, Dict[str, torch.nn.Module]]",
    inputs: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, Dict[str, torch.Tensor]],
    log_empty_loss_callable: LogEmptyLossProtocol,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    We check for empty output tensors and log a warning if we encounter them.
    This can happen when modelling on multiple outputs, which can vary in their
    sparsity, and by chance some outputs are empty in a batch.
    """
    losses_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    for output_name, target_dict in targets.items():
        for output_head_name, cur_inner_target in target_dict.items():
            if output_head_name in inputs[output_name]:
                input_tensor = inputs[output_name][output_head_name]
                criterion = criteria[output_name][output_head_name]

                if output_name not in losses_dict:
                    losses_dict[output_name] = {}

                if input_tensor.nelement() > 0 and cur_inner_target.nelement() > 0:
                    computed_loss = criterion(
                        input=input_tensor,
                        target=cur_inner_target,
                    )
                else:
                    log_empty_loss_callable(
                        output_name=output_name, output_head_name=output_head_name
                    )
                    computed_loss = torch.tensor(np.nan, requires_grad=True)

                losses_dict[output_name][output_head_name] = computed_loss
            else:
                raise KeyError(f"Missing input for {output_name} {output_head_name}")

    return losses_dict


def aggregate_losses(losses_dict: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
    losses_values = []
    for output_name, targets_for_output_dict in losses_dict.items():
        for loss in targets_for_output_dict.values():
            if not torch.isnan(loss).any():
                losses_values.append(loss)

    if not losses_values:
        return torch.tensor(np.nan, requires_grad=True)

    average_loss = torch.mean(torch.stack(losses_values))
    return average_loss


def get_uncertainty_loss_hook(
    output_configs: Sequence[OutputConfig],
    device: str,
) -> Callable:
    uncertainty_loss_modules = {}
    for output_config in output_configs:
        if output_config.output_info.output_type != "tabular":
            continue

        output_type_info = output_config.output_type_info
        assert isinstance(output_type_info, TabularOutputTypeConfig)

        if not output_type_info.uncertainty_weighted_mt_loss:
            continue

        logger.info(
            f"Adding uncertainty loss for {output_config.output_info.output_name}."
        )

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
    ) -> dict[str, torch.Tensor]:
        param_dict: dict[str, torch.Tensor] = {}
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
            input_module_with_l1 = cast("FeatureExtractorProtocolWithL1", input_module)
            cur_l1_loss = get_model_l1_loss(
                model=input_module_with_l1, l1_weight=current_l1
            )
            l1_loss += cur_l1_loss

    updated_loss = state[loss_key] + l1_loss

    state_updates = {loss_key: updated_loss}

    return state_updates


def get_model_l1_loss(
    model: "FeatureExtractorProtocolWithL1", l1_weight: float
) -> torch.Tensor:
    l1_loss = calc_l1_loss(
        weight_tensor=model.l1_penalized_weights, l1_weight=l1_weight
    )
    return l1_loss


def calc_l1_loss(weight_tensor: torch.Tensor, l1_weight: float):
    l1_loss = vector_norm(weight_tensor, ord=1, dim=None) * l1_weight
    return l1_loss


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: Dict[str, str],
    writer_funcs: Union[None, Dict[str, Dict[str, Callable]]] = None,
) -> None:

    hc = handler_config
    exp = handler_config.experiment

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

            if writer_funcs:
                cur_func = writer_funcs[output_name][target_name]
            else:
                cur_func = get_buffered_metrics_writer(buffer_interval=1)

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
) -> dict[str, dict[str, Path]]:
    assert train_or_val_target_prefix in ["validation_", "train_"]

    path_dict: dict[str, dict[str, Path]] = {}
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


def get_buffered_metrics_writer(buffer_interval: int):
    buffer = []

    def append_metrics_to_file(
        filepath: Path,
        metrics: Dict[str, float],
        iteration: int,
        write_header=False,
    ) -> None:
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
    target_transformers: Dict[str, "al_label_transformers"],
    cat_averaging_metrics: Optional[al_cat_averaging_metric_choices],
    con_averaging_metrics: Optional[al_con_averaging_metric_choices],
) -> "al_metric_record_dict":
    mcc = MetricRecord(
        name="mcc",
        function=calc_mcc,
    )
    acc = MetricRecord(
        name="acc",
        function=calc_acc,
    )
    roc_auc_macro = MetricRecord(
        name="roc-auc-macro",
        function=calc_roc_auc_ovo,
        only_val=True,
    )
    ap_macro = MetricRecord(
        name="ap-macro",
        function=calc_average_precision,
        only_val=True,
    )

    rmse = MetricRecord(
        name="rmse",
        function=partial(calc_rmse, target_transformers=target_transformers),
        minimize_goal=True,
    )
    r2 = MetricRecord(
        name="r2",
        function=calc_r2,
        only_val=True,
    )
    pcc = MetricRecord(
        name="pcc",
        function=calc_pcc,
        only_val=True,
    )

    avg_metrics = parse_averaging_metrics(
        cat_averaging_metrics=cat_averaging_metrics,
        con_averaging_metrics=con_averaging_metrics,
    )
    averaging_functions = get_performance_averaging_functions(
        cat_metric_names=avg_metrics["cat_metric_names"],
        con_metric_names=avg_metrics["con_metric_names"],
    )

    default_metrics: al_metric_record_dict = {
        "cat": (mcc, acc, roc_auc_macro, ap_macro),
        "con": (rmse, r2, pcc),
        "averaging_functions": averaging_functions,
    }
    return default_metrics


def parse_averaging_metrics(
    cat_averaging_metrics: Optional[al_cat_averaging_metric_choices],
    con_averaging_metrics: Optional[al_con_averaging_metric_choices],
) -> Dict[str, Sequence[str]]:
    base = _get_default_averaging_metrics()

    if cat_averaging_metrics:
        _validate_metrics(
            passed_in_metrics=cat_averaging_metrics,
            expected_metrics=["loss", "acc", "mcc", "roc-auc-macro", "ap-macro"],
            target_type="categorical",
        )
        base["cat_metric_names"] = cat_averaging_metrics
    if con_averaging_metrics:
        _validate_metrics(
            passed_in_metrics=con_averaging_metrics,
            expected_metrics=["loss", "rmse", "pcc", "r2"],
            target_type="continuous",
        )
        base["con_metric_names"] = con_averaging_metrics

    return base


def _validate_metrics(
    passed_in_metrics: Sequence[str], expected_metrics: Sequence[str], target_type: str
) -> None:
    for metric in passed_in_metrics:
        if metric not in expected_metrics:
            raise ValueError(
                f"Metric {metric} not found in expected metrics {expected_metrics} for "
                f" {target_type} targets."
            )


def _get_default_averaging_metrics() -> Dict[str, Sequence[str]]:
    return {
        "cat_metric_names": ["mcc", "roc-auc-macro", "ap-macro"],
        "con_metric_names": ["loss", "pcc", "r2"],
    }


def get_performance_averaging_functions(
    cat_metric_names: Sequence[str],
    con_metric_names: Sequence[str],
) -> al_averaging_functions_dict:
    """
    Note that we have the mean(values) else 0.0 to account for some values not being
    computed on the training batches, e.g. ROC-AUC, due some metrics possibly
    raising errors if e.g. there are only negative labels in a batch.
    """

    logger.info(
        "Tabular output performance averaging functions across tasks set to averages "
        "of %s for categorical targets and %s for continuous targets. These "
        "values are used to determine overall performance (using the validation set), "
        "which is used to control factors such as early stopping and LR scheduling. "
        "Other output cases use 1.0-LOSS by default.",
        [i.upper() for i in cat_metric_names],
        [i.upper().replace("LOSS", "1.0-LOSS") for i in con_metric_names],
    )

    def _calc_cat_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_names: al_cat_averaging_metric_choices,
    ) -> float:
        values = []
        for metric_name in metric_names:
            combined_key = f"{output_name}_{column_name}_{metric_name}"
            value = metric_dict[output_name][column_name].get(combined_key, None)

            if value is None:
                continue

            values.append(value)

        return mean(values) if values else 0.0

    def _calc_con_averaging_value(
        metric_dict: "al_step_metric_dict",
        output_name: str,
        column_name: str,
        metric_names: al_con_averaging_metric_choices,
    ) -> float:
        values = []
        for metric_name in metric_names:
            combined_key = f"{output_name}_{column_name}_{metric_name}"
            value = metric_dict[output_name][column_name].get(combined_key, None)

            if value is None:
                continue

            if metric_name == "loss":
                value = 1.0 - value

            values.append(value)

        return mean(values) if values else 0.0

    performance_averaging_functions: al_averaging_functions_dict = {
        "cat": partial(_calc_cat_averaging_value, metric_names=cat_metric_names),
        "con": partial(_calc_con_averaging_value, metric_names=con_metric_names),
        "general": partial(_calc_con_averaging_value, metric_names=["loss"]),
    }

    return performance_averaging_functions


@dataclass
class FilteredOutputsAndLabels:
    model_outputs: Dict[str, Dict[str, torch.Tensor]]
    target_labels: Dict[str, Dict[str, torch.Tensor]]
    ids: Dict[str, Dict[str, List[str]]]


def filter_missing_outputs_and_labels(
    batch_ids: List[str],
    model_outputs: Dict[str, Dict[str, torch.Tensor]],
    target_labels: Dict[str, Dict[str, torch.Tensor]],
    missing_ids_info: MissingTargetsInfo,
    with_labels: bool = True,
) -> FilteredOutputsAndLabels:
    filtered_outputs = {}
    filtered_labels = {}
    filtered_ids: Dict[str, Dict[str, List[str]]] = {}

    precomputed = missing_ids_info.precomputed_missing_ids

    for output_name, output_inner_dict in model_outputs.items():
        filtered_inner_dict = {}
        filtered_inner_labels = {}
        filtered_inner_ids = {}

        output_missing_info = precomputed.get(output_name, {})
        if not output_missing_info:
            filtered_outputs[output_name] = output_inner_dict
            if with_labels:
                filtered_labels[output_name] = target_labels[output_name]
            else:
                filtered_labels[output_name] = {
                    inner_key: torch.tensor([]) for inner_key in output_inner_dict
                }
            filtered_ids[output_name] = {
                inner_key: batch_ids for inner_key in output_inner_dict
            }
            continue

        for inner_key, modality_output_tensor in output_inner_dict.items():
            cur_missing_ids = output_missing_info.get(inner_key, set())

            if not cur_missing_ids:
                filtered_inner_dict[inner_key] = modality_output_tensor
                if with_labels:
                    filtered_inner_labels[inner_key] = target_labels[output_name][
                        inner_key
                    ]
                else:
                    filtered_inner_labels[inner_key] = torch.tensor([])
                filtered_inner_ids[inner_key] = batch_ids
                continue

            valid_indices = [
                i for i, id_ in enumerate(batch_ids) if id_ not in cur_missing_ids
            ]
            valid_indices_tensor = torch.tensor(
                valid_indices,
                device=modality_output_tensor.device,
                dtype=torch.long,
            )

            output_tensor = modality_output_tensor.index_select(
                dim=0, index=valid_indices_tensor
            )
            filtered_inner_dict[inner_key] = output_tensor

            if with_labels:
                cur_targets = target_labels[output_name][inner_key].index_select(
                    dim=0, index=valid_indices_tensor
                )
                filtered_inner_labels[inner_key] = cur_targets
            else:
                filtered_inner_labels[inner_key] = torch.tensor([])

            ids = [batch_ids[i] for i in valid_indices]
            filtered_inner_ids[inner_key] = ids

        filtered_outputs[output_name] = filtered_inner_dict
        filtered_labels[output_name] = filtered_inner_labels
        filtered_ids[output_name] = filtered_inner_ids

    return FilteredOutputsAndLabels(
        model_outputs=filtered_outputs,
        target_labels=filtered_labels,
        ids=filtered_ids,
    )
