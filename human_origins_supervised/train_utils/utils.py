from typing import List, Dict, TYPE_CHECKING
import importlib

from aislib.misc_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from human_origins_supervised.data_load.label_setup import al_label_dict


def import_custom_module(module_path):
    try:
        module = importlib.import_module(module_path)
        logger.debug("Imported custom module %s", module_path)
    except ImportError:
        logger.debug("Could not find custom module %s. Skipping.", module_path)
        return None

    return module


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


def check_if_iteration_sample(
    iteration: int,
    iter_sample_interval: int,
    n_iterations_per_epochs: int,
    n_epochs: int,
) -> bool:
    if iter_sample_interval:
        condition_1 = iteration % iter_sample_interval == 0
    else:
        condition_1 = False

    condition_2 = iteration == n_iterations_per_epochs * n_epochs

    return condition_1 or condition_2
