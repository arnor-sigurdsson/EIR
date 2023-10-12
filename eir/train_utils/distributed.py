import os
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Tuple, Union

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import al_meta_model
    from eir.setup.config import Configs

logger = get_logger(name=__name__)


def maybe_initialize_distributed_environment(
    configs: "Configs",
) -> Tuple["Configs", Union[int, None]]:
    is_distributed_run = in_distributed_env()
    if is_distributed_run:
        logger.info(
            "'LOCAL_RANK': '%s' environment variable found. Assuming "
            "distributed distributed training.",
            os.environ["LOCAL_RANK"],
        )
    else:
        return configs, None

    dist.init_process_group(backend="gloo")

    configs_copy = copy(configs)

    local_rank = int(os.environ["LOCAL_RANK"])
    if "cuda" in configs.global_config.device:
        configs_copy.global_config.device = f"cuda:{local_rank}"

    return configs_copy, local_rank


def maybe_make_model_distributed(
    device: str, model: "al_meta_model"
) -> "al_meta_model":
    if not in_distributed_env():
        return model

    local_rank = int(os.environ["LOCAL_RANK"])

    device_ids = None
    output_device = None

    if "cuda" in device:
        device_ids = [local_rank]
        output_device = local_rank

    model = AttrDelegatedDistributedDataParallel(
        module=model,
        device_ids=device_ids,
        output_device=output_device,
    )

    logger.info(
        "Initialized distributed model with rank '%d' and arguments "
        "'device_ids': '%s', 'output_device': '%s'.",
        local_rank,
        device_ids,
        output_device,
    )

    return model


def in_distributed_env() -> bool:
    if "LOCAL_RANK" in os.environ:
        return True
    return False


def in_master_node() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    if local_rank == 0 and global_rank == 0:
        return True

    return False


def only_call_on_master_node(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if in_master_node():
            result = func(*args, **kwargs)
            return result

        return

    return wrapper


class AttrDelegatedDistributedDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
