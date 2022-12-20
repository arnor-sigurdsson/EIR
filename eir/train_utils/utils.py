import logging
import os
import random
from functools import wraps
from pathlib import Path
from typing import (
    Dict,
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

from eir.train_utils.distributed import (
    only_call_on_master_node,
    in_master_node,
    in_distributed_env,
)

logger = get_logger(name=__name__, tqdm_compatible=True)


def get_run_folder(output_folder: str) -> Path:
    return Path(output_folder)


def prep_sample_outfolder(
    output_folder: str, output_name: str, column_name: str, iteration: int
) -> Path:
    sample_outfolder = (
        get_run_folder(output_folder=output_folder)
        / "results"
        / output_name
        / column_name
        / "samples"
        / str(iteration)
    )
    ensure_path_exists(sample_outfolder, is_folder=True)

    return sample_outfolder


@only_call_on_master_node
def configure_root_logger(output_folder: str) -> None:

    logfile_path = get_run_folder(output_folder=output_folder) / "logging_history.log"

    ensure_path_exists(path=logfile_path)
    file_handler = logging.FileHandler(filename=str(logfile_path))
    file_handler.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(fmt=formatter)

    logging.getLogger(name="").addHandler(hdlr=file_handler)


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
                    if in_master_node() or not in_distributed_env():
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
