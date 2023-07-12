import reprlib
from collections import defaultdict
from functools import partial
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Type,
)

import torch
from adabelief_pytorch import AdaBelief
from aislib.pytorch_modules import AdaHessian
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch_optimizer import get as get_custom_opt

from eir.models.model_training_utils import add_wd_to_model_params
from eir.setup.setup_utils import get_base_optimizer_names
from eir.utils.logging import get_logger

if TYPE_CHECKING:
    from eir.models.model_setup_modules.meta_setup import al_meta_model
    from eir.setup.schemas import GlobalConfig

logger = get_logger(name=__name__, tqdm_compatible=True)


def get_optimizer(
    model: nn.Module, loss_callable: Callable, global_config: "GlobalConfig"
) -> Optimizer:
    all_params = _get_all_params_to_optimize(
        model=model, weight_decay=global_config.wd, loss_callable=loss_callable
    )

    optimizer_class = _get_optimizer_class(optimizer_name=global_config.optimizer)
    optimizer_args = _get_constructor_arguments(
        params=all_params, global_config=global_config, optimizer_class=optimizer_class
    )
    optimizer = optimizer_class(**optimizer_args)

    logger.debug(
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


def _create_optimizer_class_getter(
    optimizer_name: str,
) -> MutableMapping[str, Type[Optimizer]]:
    """
    We use an interface with the external optimizer library as a default factory,
    meaning that if we cannot find an optimizer in our base optimizer, we check if
    they exist in the external library.
    """
    default_factory = partial(_get_external_optimizers, optimizer_name)
    optimizer_getter: MutableMapping[str, Type[Optimizer]] = defaultdict(
        default_factory
    )

    base_optimizers = get_base_optimizers_dict()
    optimizer_getter.update(base_optimizers)

    return optimizer_getter


def get_base_optimizers_dict() -> Dict[str, Type[Optimizer]]:
    base_optimizers = {
        "sgdm": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "adahessian": AdaHessian,
        "adabelief": partial(AdaBelief, print_change_log=False),
        "adabeliefw": partial(AdaBelief, weight_decouple=True, print_change_log=False),
    }
    assert set(base_optimizers) == get_base_optimizer_names()
    return base_optimizers


def _get_constructor_arguments(
    params: List, global_config: "GlobalConfig", optimizer_class: Type[Optimizer]
) -> dict[str, Any]:
    base = {"params": params, "lr": global_config.lr, "weight_decay": global_config.wd}
    all_extras = {
        "betas": (global_config.b1, global_config.b2),
        "momentum": 0.9,
        "amsgrad": False,
    }
    accepted_args = signature(optimizer_class).parameters.keys()
    common_extras = {k: v for k, v in all_extras.items() if k in accepted_args}
    return {**base, **common_extras}


def get_optimizer_backward_kwargs(optimizer_name: str) -> dict[str, Any]:
    if optimizer_name == "adahessian":
        return {"create_graph": True}
    return {}


def get_optimizer_defaults(optimizer: Optimizer) -> dict:
    return optimizer.defaults


class AttrDelegatedSWAWrapper(torch.optim.swa_utils.AveragedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def maybe_wrap_model_with_swa(
    n_iter_before_swa: Optional[int],
    model: "al_meta_model",
    device: torch.device,
) -> "al_meta_model":
    if n_iter_before_swa is None:
        return model

    swa_model = AttrDelegatedSWAWrapper(model=model, device=device)
    return swa_model
