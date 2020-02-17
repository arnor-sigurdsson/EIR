import csv
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING

import pandas as pd
from aislib.misc_utils import get_logger, ensure_path_exists

logger = get_logger(name=__name__, tqdm_compatible=True)

if TYPE_CHECKING:
    from human_origins_supervised.data_load.label_setup import al_label_dict
    from human_origins_supervised.data_load.label_setup import al_target_columns


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
        run_folder, f"{target_prefix}average-loss_history.log"
    )
    path_dict[f"{target_prefix}loss-average"] = average_loss_training_metrics_file

    return path_dict


def ensure_metrics_paths_exists(metrics_files: Dict[str, Path]) -> None:
    for path in metrics_files.values():
        ensure_path_exists(path)


def filter_items_from_engine_metrics_dict(
    engine_metrics_dict: Dict[str, float], metrics_substring: str
):
    """
    Note that in case of the average loss we are not using a target column as the
    target string – hence we just return it directly.

    Here we are matching the against patterns likes 't_Origin_mcc' but note that this
    could also include names like 't_Country_of_origin_mcc', or
    't_Country-of-origin_mcc' where Country_of_origin is a column name. So we check if
     the target is actually in the metric.
    """

    if metrics_substring.endswith("loss-average"):
        return {metrics_substring: engine_metrics_dict[metrics_substring]}

    output_dict = {}
    for metric_name, metric_value in engine_metrics_dict.items():
        if metrics_substring in metric_name:
            output_dict[metric_name] = metric_value

    return output_dict
