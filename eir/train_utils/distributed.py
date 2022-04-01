import os
from copy import copy
from functools import wraps
from typing import Callable, Any, TYPE_CHECKING, Tuple, Union

from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from aislib.misc_utils import get_logger

if TYPE_CHECKING:
    from eir.setup.config import Configs
    from eir.models.model_setup import al_fusion_models

logger = get_logger(name=__name__)


def maybe_initialize_distributed_environment(
    configs: "Configs",
) -> Tuple["Configs", Union[int, None]]:

    is_distributed_run = in_distributed_env()
    if is_distributed_run:
        logger.info(
            "'LOCAL_RANK' environment variable found. Assuming"
            "distributed distributed training."
        )
    else:
        return configs, None

    dist.init_process_group(backend="gloo")

    configs_copy = copy(configs)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "cuda" in configs.global_config.device:
        configs_copy.global_config.device = f"cuda:{local_rank}"

    return configs_copy, local_rank


def maybe_make_model_distributed(device: str, model: "al_fusion_models"):

    if not in_distributed_env():
        return model

    local_rank = int(os.environ["LOCAL_RANK"])

    ddp_kwargs = {"device_ids": None, "output_device": None}
    if "cuda" in device:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}

    model = DistributedDataParallel(module=model, **ddp_kwargs)

    logger.info(
        "Initialized distributed model with rank '%d' and arguments: '%s'.",
        local_rank,
        ddp_kwargs,
    )

    return model


def in_distributed_env() -> bool:
    if "LOCAL_RANK" in os.environ:
        return True
    return False


def only_call_on_master_node(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if local_rank == 0:
            result = func(*args, **kwargs)
            return result

        return

    return wrapper
