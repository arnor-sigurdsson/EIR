import os
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from torch.nn.parallel import DistributedDataParallel

from eir.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(name=__name__)


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
