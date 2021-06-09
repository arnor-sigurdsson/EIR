import argparse
import reprlib
from collections import defaultdict
from functools import partial
from inspect import signature
from typing import Callable, List, Type, Dict

from adabelief_pytorch import AdaBelief
from aislib.misc_utils import get_logger
from aislib.pytorch_modules import AdaHessian
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch_optimizer import get as get_custom_opt

from eir.models.model_training_utils import add_wd_to_model_params

logger = get_logger(name=__name__, tqdm_compatible=True)


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

    logger.info(
        "Optimizer %s created with arguments %s (note weight decay not included as it "
        "is added manually to parameter groups). Parameter groups %s.",
        optimizer_class,
        get_optimizer_defaults(optimizer=optimizer),
        reprlib.repr(optimizer.param_groups),
    )

    return optimizer


def _get_all_params_to_optimize(
    model: nn.Module, weight_decay: float, loss_callable: Callable
) -> List:
    model_params = add_wd_to_model_params(model=model, wd=weight_decay)

    loss_params = []
    if isinstance(loss_callable, nn.Module):
        loss_params = [{"params": p} for p in loss_callable.parameters()]

    return model_params + loss_params


def _get_external_optimizers(optimizer_name: str) -> Type[Optimizer]:
    custom_opt = get_custom_opt(name=optimizer_name)
    return custom_opt


def _get_optimizer_class(optimizer_name: str) -> Type[Optimizer]:

    optimizer_getter = _create_optimizer_class_getter(optimizer_name=optimizer_name)
    optimizer_class = optimizer_getter[optimizer_name]
    return optimizer_class


def _create_optimizer_class_getter(optimizer_name: str) -> Dict[str, Type[Optimizer]]:
    """
    We use an interface with the external optimizer library as a default factory,
    meaning that if we cannot find an optimizer in our base optimizer, we check if
    they exist in the external library.
    """
    default_factory = partial(_get_external_optimizers, optimizer_name)
    optimizer_getter = defaultdict(default_factory)

    base_optimizers = get_base_optimizers_dict()
    optimizer_getter.update(base_optimizers)

    return optimizer_getter


def get_base_optimizers_dict() -> Dict[str, Type[Optimizer]]:
    base_optimizers = {
        "sgdm": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "adahessian": AdaHessian,
        "adabelief": AdaBelief,
        "adabeliefw": partial(AdaBelief, weight_decouple=True),
    }
    return base_optimizers


def _get_constructor_arguments(
    params: List, cl_args: argparse.Namespace, optimizer_class: Type[Optimizer]
):
    base = {"params": params, "lr": cl_args.lr, "weight_decay": cl_args.wd}
    all_extras = {"betas": (cl_args.b1, cl_args.b2), "momentum": 0.9, "amsgrad": False}
    accepted_args = signature(optimizer_class).parameters.keys()
    common_extras = {k: v for k, v in all_extras.items() if k in accepted_args}
    return {**base, **common_extras}


def get_optimizer_backward_kwargs(optimizer_name: str) -> Dict:
    if optimizer_name == "adahessian":
        return {"create_graph": True}
    return {}


def get_optimizer_defaults(optimizer: Optimizer) -> Dict:
    return optimizer.defaults
