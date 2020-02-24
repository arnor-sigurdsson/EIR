import csv
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from aislib.misc_utils import get_logger, ensure_path_exists

logger = get_logger(name=__name__, tqdm_compatible=True)

if TYPE_CHECKING:
    from human_origins_supervised.data_load.label_setup import (
        al_label_dict,
        al_target_columns,
    )
    from human_origins_supervised.train_utils.train_handlers import (
        HandlerConfig,
        al_step_metric_dict,
    )


def import_custom_module_as_package(module_path, module_name):
    """
    We need to make sure sys.modules[spec.name] = module is called when importing
    a package from an absolute path.

    See: https://docs.python.org/3/reference/import.html#loading
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        logger.debug("Imported custom module %s at %s", module_name, module_path)
    except ImportError:
        logger.error(
            "Could not import custom module %s at %s.", module_name, module_path
        )
        raise

    return module


def get_custom_module_submodule(custom_lib: str, submodule_name: str):
    module_path = custom_lib + "/__init__.py"
    module_name = Path(custom_lib).name

    custom_module = import_custom_module_as_package(module_path, module_name)

    if not hasattr(custom_module, submodule_name):
        logger.debug(
            f"Could not find function {submodule_name} in {module_path}."
            f"Either it is not defined (which is fine) or something went"
            f"wrong. Please check that it is in the correct location"
            f"and that {module_path} actually imports it."
        )
        return None

    return getattr(custom_module, submodule_name)


def get_extra_labels_from_ids(
    labels_dict: "al_label_dict", cur_ids: List[str], target_columns: List[str]
) -> List[Dict[str, str]]:
    """
    Returns a batch in same order as cur_ids.
    """
    extra_labels = []
    for sample_id in cur_ids:
        cur_labels_all = labels_dict.get(sample_id)
        cur_labels_extra = {
            k: v for k, v in cur_labels_all.items() if k in target_columns
        }
        extra_labels.append(cur_labels_extra)

    return extra_labels


def get_run_folder(run_name: str) -> Path:
    return Path("runs", run_name)


def prep_sample_outfolder(run_name: str, column_name: str, iteration: int) -> Path:
    sample_outfolder = (
        get_run_folder(run_name) / "results" / column_name / "samples" / str(iteration)
    )
    ensure_path_exists(sample_outfolder, is_folder=True)

    return sample_outfolder


def persist_metrics(
    handler_config: "HandlerConfig",
    metrics_dict: "al_step_metric_dict",
    iteration: int,
    write_header: bool,
    prefixes: Dict[str, str],
):

    hc = handler_config
    c = handler_config.config
    cl_args = c.cl_args

    metrics_files = get_metrics_files(
        target_columns=c.target_columns,
        run_folder=hc.run_folder,
        target_prefix=f"{prefixes['metrics']}",
    )

    if write_header:
        ensure_metrics_paths_exists(metrics_files)

    for metrics_name, metrics_history_file in metrics_files.items():
        cur_metric_dict = metrics_dict[metrics_name]

        add_metrics_to_writer(
            name=f"{prefixes['writer']}/{metrics_name}",
            metric_dict=cur_metric_dict,
            iteration=iteration,
            writer=c.writer,
            plot_skip_steps=cl_args.plot_skip_steps,
        )

        append_metrics_to_file(
            filepath=metrics_history_file,
            metrics=cur_metric_dict,
            iteration=iteration,
            write_header=write_header,
        )


def append_metrics_to_file(
    filepath: Path, metrics: Dict[str, float], iteration: int, write_header=False
):
    with open(str(filepath), "a") as logfile:
        fieldnames = ["iteration"] + sorted(metrics.keys())
        writer = csv.DictWriter(logfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        dict_to_write = {**{"iteration": iteration}, **metrics}
        writer.writerow(dict_to_write)


def read_metrics_history_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col="iteration")

    return df


def get_metrics_files(
    target_columns: "al_target_columns", run_folder: Path, target_prefix: str
) -> Dict[str, Path]:
    all_target_columns = target_columns["con"] + target_columns["cat"]

    path_dict = {}
    for target_column in all_target_columns:
        cur_fname = target_prefix + target_column + "_history.log"
        cur_path = Path(run_folder, "results", target_column, cur_fname)
        path_dict[target_column] = cur_path

    average_loss_training_metrics_file = Path(
        run_folder, f"{target_prefix}average_history.log"
    )
    path_dict[f"{target_prefix}average"] = average_loss_training_metrics_file

    return path_dict


def ensure_metrics_paths_exists(metrics_files: Dict[str, Path]) -> None:
    for path in metrics_files.values():
        ensure_path_exists(path)


def add_metrics_to_writer(
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


def configure_root_logger(run_name: str):

    logfile_path = get_run_folder(run_name=run_name) / "logging_history.log"

    ensure_path_exists(logfile_path)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logging.getLogger("").addHandler(file_handler)
