import sys
from typing import List, Dict, TYPE_CHECKING
import importlib
import importlib.util
from pathlib import Path

from aislib.misc_utils import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)

if TYPE_CHECKING:
    from human_origins_supervised.data_load.label_setup import al_label_dict


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


def get_run_folder(run_name: str) -> Path:
    return Path("runs", run_name)
