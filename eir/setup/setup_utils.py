import torch
from typing import Iterable, Sequence
from tqdm import tqdm

from torch_optimizer import _NAME_OPTIM_MAP


class RunningStatistics:
    """
    Adapted from https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469.
    """

    def __init__(self, n_dims: int = 2):
        self.n_dims = n_dims
        self.n = 0
        self.sum = 0

        self.shape = None
        self.num_var = None

    def update(self, data: torch.Tensor):
        data = data.view(*list(data.shape[: -self.n_dims]) + [-1])

        with torch.no_grad():
            new_n = data.shape[-1]
            new_var = data.var(-1)
            new_sum = data.sum(-1)

            if self.n == 0:
                self.n = new_n
                self.shape = data.shape[:-1]
                self.sum = new_sum
                self.num_var = new_var.mul_(new_n)

            else:

                ratio = self.n / new_n
                t = (self.sum / ratio).sub_(new_sum).pow_(2)

                self.num_var.add_(other=new_var, alpha=new_n)
                self.num_var.add_(other=t, alpha=ratio / (self.n + new_n))

                self.sum += new_sum
                self.n += new_n

    @property
    def mean(self):
        return self.sum / self.n if self.n > 0 else None

    @property
    def var(self):
        return self.num_var / self.n if self.n > 0 else None

    @property
    def std(self):
        return self.var.sqrt() if self.n > 0 else None


def collect_stats(
    tensor_iterable: Iterable[torch.Tensor], n_dims: int = 2
) -> RunningStatistics:
    stats = RunningStatistics(n_dims)
    for it in tqdm(tensor_iterable, desc="Gathering Image Statistics: "):
        if hasattr(it, "data"):
            stats.update(it.data)
        else:
            stats.update(it)
    return stats


def get_base_optimizer_names() -> set:
    base_names = {"sgdm", "adam", "adamw", "adahessian", "adabelief", "adabeliefw"}

    return base_names


def get_all_optimizer_names() -> Sequence[str]:
    external_optimizers = set(_NAME_OPTIM_MAP.keys())
    base_optimizers = get_base_optimizer_names()
    all_optimizers = set.union(base_optimizers, external_optimizers)
    all_optimizers = sorted(list(all_optimizers))

    return all_optimizers
