from typing import Callable

from torch import Tensor, nn


class Residual(nn.Module):
    def __init__(self, func: Callable[[Tensor], Tensor]):
        super().__init__()
        self.fn = func

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
