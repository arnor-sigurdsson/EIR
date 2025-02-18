import os
from collections.abc import Callable
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import al_meta_model
    from eir.setup.config import Configs

logger = get_logger(name=__name__)


def maybe_initialize_distributed_environment(
    configs: "Configs",
) -> tuple["Configs", int | None]:
    is_distributed_run = in_distributed_env()
    if not is_distributed_run:
        return configs, None

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    logger.info(
        "'LOCAL_RANK': '%s' environment variable found. Assuming "
        "distributed training. World size: %d",
        local_rank,
        world_size,
    )

    if "cuda" in configs.gc.be.device:
        torch.cuda.set_device(device=local_rank)
        configs_copy = copy(configs)
        configs_copy.gc.be.device = f"cuda:{local_rank}"
        backend = "nccl"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        configs_copy = copy(configs)
        backend = "gloo"
        dist.init_process_group(backend=backend)

    logger.info(
        f"Initialized process group with backend '{backend}' on rank {local_rank}, "
        f"device: {configs_copy.gc.be.device}"
    )

    return configs_copy, local_rank


def maybe_make_model_distributed(
    device: str,
    model: "al_meta_model",
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
        "Initialized DDP distributed model with rank '%d' and arguments "
        "'device_ids': '%s', 'output_device': '%s'.",
        local_rank,
        device_ids,
        output_device,
    )

    return model


def in_distributed_env() -> bool:
    return "LOCAL_RANK" in os.environ


def in_master_node() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    return bool(local_rank == 0 and global_rank == 0)


def only_call_on_master_node(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        if in_master_node():
            result = func(*args, **kwargs)
            return result

        return None

    return wrapper


class AttrDelegatedDistributedDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
