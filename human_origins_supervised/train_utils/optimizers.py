import argparse
from collections import defaultdict
from functools import partial
from inspect import signature
from typing import Callable, List, Type

from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from torch.optim.adamw import AdamW

from torch_optimizer import get as get_custom_opt

from human_origins_supervised.models.model_utils import get_model_params


def get_optimizer(
    model: nn.Module, loss_callable: Callable, cl_args: argparse.Namespace
) -> Optimizer:

    all_params = _get_all_params_to_optimize(
        model=model, weight_decay=cl_args.wd, loss_callable=loss_callable
    )

    optimizer_class = _get_optimizer_class(optimizer_name=cl_args.optimizer)
    optimizer_args = _get_constructor_arguments(
        params=all_params, cl_args=cl_args, optimizer_class=optimizer_class
    )
    optimizer = optimizer_class(**optimizer_args)

    return optimizer


def _get_all_params_to_optimize(
    model: nn.Module, weight_decay: float, loss_callable: Callable
) -> List:
    model_params = get_model_params(model=model, wd=weight_decay)

    loss_params = []
    if isinstance(loss_callable, nn.Module):
        loss_params = [{"params": p} for p in loss_callable.parameters()]

    return model_params + loss_params


def _get_external_optimizers(optimizer_name: str) -> Type[Optimizer]:

    custom_opt = get_custom_opt(name=optimizer_name)
    return custom_opt


def _get_optimizer_class(optimizer_name: str) -> Optimizer:
    base_optimizers = {"sgdm": SGD, "adam": Adam, "adamw": AdamW}
    optimizer_getter = defaultdict(partial(_get_external_optimizers, optimizer_name))
    optimizer_getter.update(base_optimizers)

    optimizer_class = optimizer_getter[optimizer_name]
    return optimizer_class


def _get_constructor_arguments(
    params: List, cl_args: argparse.Namespace, optimizer_class: Optimizer
):
    base = {"params": params, "lr": cl_args.lr}
    all_extras = {"betas": (cl_args.b1, cl_args.b2), "momentum": 0.9, "amsgrad": True}
    accepted_args = signature(optimizer_class).parameters.keys()
    common_extras = {k: v for k, v in all_extras.items() if k in accepted_args}
    return {**base, **common_extras}
