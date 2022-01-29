import logging
import os
import random
from functools import wraps
from pathlib import Path
from typing import (
    List,
    Dict,
    TYPE_CHECKING,
    Sequence,
    Callable,
    Iterable,
    Union,
    Any,
    Tuple,
)

import numpy as np
import torch
from aislib.misc_utils import get_logger, ensure_path_exists
from ignite.engine import Engine

logger = get_logger(name=__name__, tqdm_compatible=True)

if TYPE_CHECKING:
    from eir.data_load.label_setup import al_label_dict


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


def get_run_folder(output_folder: str) -> Path:
    return Path(output_folder)


def prep_sample_outfolder(output_folder: str, column_name: str, iteration: int) -> Path:
    sample_outfolder = (
        get_run_folder(output_folder=output_folder)
        / "results"
        / column_name
        / "samples"
        / str(iteration)
    )
    ensure_path_exists(sample_outfolder, is_folder=True)

    return sample_outfolder


def configure_root_logger(output_folder: str):

    logfile_path = get_run_folder(output_folder=output_folder) / "logging_history.log"

    ensure_path_exists(logfile_path)
    file_handler = logging.FileHandler(str(logfile_path))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logging.getLogger("").addHandler(file_handler)


def validate_handler_dependencies(handler_dependencies: Sequence[Callable]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            argument_iterable = tuple(args) + tuple(kwargs.values())

            engine_objects = [i for i in argument_iterable if isinstance(i, Engine)]
            assert len(engine_objects) == 1

            engine_object = engine_objects[0]

            for dep in handler_dependencies:
                if not engine_object.has_event_handler(dep):
                    logger.warning(
                        f"Dependency '{dep.__name__}' missing from engine. "
                        f"If your are running EIR directly through the CLI, "
                        f"this is likely a bug. If you are customizing "
                        f"EIR (e.g. the validation handler), this can "
                        f"be expected, please ignore this warning in "
                        f"that case."
                    )

            func_output = func(*args, **kwargs)
            return func_output

        return wrapper

    return decorator


class MissingHandlerDependencyError(Exception):
    pass


def call_hooks_stage_iterable(
    hook_iterable: Iterable[Callable],
    common_kwargs: Dict,
    state: Union[None, Dict[str, Any]],
):
    for hook in hook_iterable:
        _, state = state_registered_hook_call(
            hook_func=hook, **common_kwargs, state=state
        )

    return state


def state_registered_hook_call(
    hook_func: Callable,
    state: Union[Dict[str, Any], None],
    *args,
    **kwargs,
) -> Tuple[Any, Dict[str, Any]]:

    if state is None:
        state = {}

    state_updates = hook_func(state=state, *args, **kwargs)

    state = {**state, **state_updates}

    return state_updates, state


def seed_everything(seed: int = 0) -> None:

    seed, from_os = get_seed(default_seed=seed)

    extra_log = " grabbed from environment variable 'EIR_SEED '" if from_os else " "
    logger.debug("Global random seed%sset to %d.", extra_log, seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_seed(default_seed: int = 0) -> Tuple[int, bool]:
    os_seed = os.environ.get("EIR_SEED", None)

    if os_seed:
        seed = int(os_seed)
        from_os = True
    else:
        seed = default_seed
        from_os = False

    return seed, from_os
