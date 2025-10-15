from collections import defaultdict
from collections.abc import Callable

import torch
from torch.optim import AdamW, Muon
from torch.optim.optimizer import Optimizer

from eir.utils.logging import get_logger

logger = get_logger(name=__name__, tqdm_compatible=True)


class MuonAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        muon_filter: Callable[[torch.nn.Parameter], bool] | None = None,
        **kwargs,
    ) -> None:
        if muon_filter is None:

            def muon_filter(p):
                return p.ndim == 2

        muon_groups = []
        adamw_groups = []

        for group in params:
            group_params = list(group["params"])
            force_adamw = group.get("force_adamw", False)

            if force_adamw:
                muon_params = []
                adamw_params = group_params
            else:
                muon_params = [p for p in group_params if muon_filter(p)]
                muon_param_ids = {id(p) for p in muon_params}
                adamw_params = [p for p in group_params if id(p) not in muon_param_ids]

            if muon_params:
                muon_group = dict(group)
                muon_group["params"] = muon_params
                muon_groups.append(muon_group)

            if adamw_params:
                adamw_group = dict(group)
                adamw_group["params"] = adamw_params
                adamw_groups.append(adamw_group)

        self.muon_optimizer: Muon | None = None
        self.adamw_optimizer: AdamW | None = None

        muon_kwargs = {k: v for k, v in kwargs.items() if k not in ("betas",)}
        if muon_groups:
            self.muon_optimizer = Muon(
                muon_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                **muon_kwargs,
            )

        adamw_kwargs = {k: v for k, v in kwargs.items() if k not in ("momentum",)}
        if adamw_groups:
            self.adamw_optimizer = AdamW(
                adamw_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                **adamw_kwargs,
            )

        self.defaults = {"lr": lr, "weight_decay": weight_decay}
        self.state = defaultdict(dict)

        muon_param_count = (
            sum(
                p.numel()
                for group in self.muon_optimizer.param_groups
                for p in group["params"]
            )
            if self.muon_optimizer
            else 0
        )
        adamw_param_count = (
            sum(
                p.numel()
                for group in self.adamw_optimizer.param_groups
                for p in group["params"]
            )
            if self.adamw_optimizer
            else 0
        )

        logger.debug(
            f"MuonAdamW optimizer created: "
            f"Muon handles {muon_param_count:,} parameters, "
            f"AdamW handles {adamw_param_count:,} parameters"
        )

    @property
    def param_groups(self) -> list[dict]:  # type: ignore[override]
        groups = []
        if self.muon_optimizer is not None:
            groups.extend(self.muon_optimizer.param_groups)
        if self.adamw_optimizer is not None:
            groups.extend(self.adamw_optimizer.param_groups)
        return groups

    def add_param_group(self, param_group: dict) -> None:
        params = list(param_group["params"])
        muon_params = [p for p in params if p.ndim == 2]
        muon_param_ids = {id(p) for p in muon_params}
        adamw_params = [p for p in params if id(p) not in muon_param_ids]

        if muon_params:
            muon_group = dict(param_group)
            muon_group["params"] = muon_params
            if self.muon_optimizer is None:
                self.muon_optimizer = Muon(
                    [muon_group],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                    momentum=0.95,
                )
            else:
                self.muon_optimizer.add_param_group(muon_group)

        if adamw_params:
            adamw_group = dict(param_group)
            adamw_group["params"] = adamw_params
            if self.adamw_optimizer is None:
                self.adamw_optimizer = AdamW(
                    [adamw_group],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                )
            else:
                self.adamw_optimizer.add_param_group(adamw_group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_optimizer is not None:
            self.muon_optimizer.step()
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon": (
                self.muon_optimizer.state_dict()
                if self.muon_optimizer is not None
                else None
            ),
            "adamw": (
                self.adamw_optimizer.state_dict()
                if self.adamw_optimizer is not None
                else None
            ),
            "defaults": self.defaults,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "defaults" in state_dict:
            self.defaults.update(state_dict["defaults"])

        muon_state = state_dict.get("muon")
        if self.muon_optimizer is not None and muon_state is not None:
            self.muon_optimizer.load_state_dict(muon_state)

        adamw_state = state_dict.get("adamw")
        if self.adamw_optimizer is not None and adamw_state is not None:
            self.adamw_optimizer.load_state_dict(adamw_state)
