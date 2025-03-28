import itertools
import math
import numbers
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import copy
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

# https://github.com/pytorch/ignite/issues/2773
try:
    from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as PyTorchLRScheduler

from eir.train_utils.ignite_port.engine import Engine


class BaseParamScheduler(metaclass=ABCMeta):
    r"""An abstract class for updating an engine state or optimizer's parameter value during
    training.

    Args:
        param_name: name of engine state or optimizer's parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    .. versionadded:: 0.4.7

    """

    def __init__(self, param_name: str, save_history: bool = False):
        self.param_name = param_name
        self.event_index = 0
        self._save_history = save_history
        self._state_attrs = ["event_index", "param_name", "save_history"]

    @property
    def save_history(self) -> bool:
        return self._save_history

    @save_history.setter
    def save_history(self, value: bool) -> None:
        self._save_history = value

    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary containing a whole state of BaseParamScheduler.

        Returns:
            dict:
                a dictionary containing a whole state of BaseParamScheduler
        """
        destination = OrderedDict()
        for name in self._state_attrs:
            if hasattr(self, name):
                val = getattr(self, name)
                if hasattr(val, "state_dict"):
                    val = val.state_dict()
                destination[name] = copy(val)
        return destination

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Copies parameters from :attr:`state_dict` into this BaseParamScheduler.

        Args:
            state_dict: a dict containing parameters.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        for name in self._state_attrs:
            if name not in state_dict:
                raise ValueError(
                    f"Required state attribute '{name}' is absent in provided state_dict '{state_dict.keys()}'"
                )
            val = state_dict[name]
            obj = getattr(self, name)
            if isinstance(val, Mapping) and hasattr(obj, "load_state_dict"):
                obj.load_state_dict(val)
            else:
                setattr(self, name, val)

    @abstractmethod
    def get_param(self) -> list[float] | float:
        """Method to get current parameter values

        Returns:
            list of params, or scalar param
        """
        pass

    @classmethod
    @abstractmethod
    def simulate_values(cls, num_events: int, **scheduler_kwargs: Any) -> list[list[int]]:
        """Method to simulate scheduled values during `num_events` events.

        Args:
            num_events: number of events during the simulation.
            scheduler_kwargs: parameter scheduler configuration kwargs.

        Returns:
            event_index, value
        """
        pass

    @classmethod
    def plot_values(cls, num_events: int, **scheduler_kwargs: Mapping) -> Any:
        """Method to plot simulated scheduled values during `num_events` events.

        This class requires `matplotlib package <https://matplotlib.org/>`_ to be installed:

        .. code-block:: bash

            pip install matplotlib

        Args:
            num_events: number of events during the simulation.
            scheduler_kwargs: parameter scheduler configuration kwargs.

        Returns:
            matplotlib.lines.Line2D

        Examples:
            .. code-block:: python

                import matplotlib.pylab as plt

                plt.figure(figsize=(10, 7))
                LinearCyclicalScheduler.plot_values(num_events=50, param_name='lr',
                                                    start_value=1e-1, end_value=1e-3, cycle_size=10))
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ModuleNotFoundError(
                "This method requires matplotlib to be installed. "
                "Please install it with command: \n pip install matplotlib"
            )

        values = cls.simulate_values(num_events=num_events, **scheduler_kwargs)
        label = scheduler_kwargs.get("param_name", "learning rate")
        ax = plt.plot([e for e, _ in values], [v for _, v in values], label=label)
        plt.legend()
        plt.grid(which="both")
        return ax


class ParamScheduler(BaseParamScheduler):
    """An abstract class for updating an optimizer's parameter value during
    training.

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use

    Note:
        Parameter scheduler works independently of the internal state of the attached optimizer.
        More precisely, whatever the state of the optimizer (newly created or used by another scheduler) the scheduler
        sets defined absolute values.


    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str,
        save_history: bool = False,
        param_group_index: int | None = None,
    ):
        super(ParamScheduler, self).__init__(param_name, save_history)
        if not (
            isinstance(optimizer, Optimizer)
            or (hasattr(optimizer, "param_groups") and isinstance(optimizer.param_groups, Sequence))
        ):
            raise TypeError(
                "Argument optimizer should be torch.optim.Optimizer or has attribute 'param_groups' as list/tuple, "
                f"but given {type(optimizer)}"
            )

        self.optimizer = optimizer
        self.param_group_index = param_group_index
        self._state_attrs += ["param_group_index"]

    def __call__(self, engine: Engine | None, name: str | None = None) -> None:
        value = self._get_param()

        if isinstance(value, list):
            if len(value) != len(self.optimizer_param_groups):
                raise ValueError(
                    "size of value is different than optimizer_param_groups "
                    f"{len(value)} != {len(self.optimizer_param_groups)}"
                )

            for i, param_group in enumerate(self.optimizer_param_groups):
                param_group[self.param_name] = value[i]
        else:
            for i, param_group in enumerate(self.optimizer_param_groups):
                param_group[self.param_name] = value

        if name is None:
            name = self.param_name

        if self.save_history and engine:
            if not hasattr(engine.state, "param_history") or engine.state.param_history is None:
                engine.state.param_history = {}
            engine.state.param_history.setdefault(name, [])  # type: ignore[attr-defined]
            values = [pg[self.param_name] for pg in self.optimizer_param_groups]
            engine.state.param_history[name].append(values)  # type: ignore[attr-defined]
        self.event_index += 1

    @property
    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        if self.param_group_index is None:
            return self.optimizer.param_groups
        return [self.optimizer.param_groups[self.param_group_index]]

    @classmethod
    def simulate_values(cls, num_events: int, **scheduler_kwargs: Any) -> list[list[int]]:
        """Method to simulate scheduled values during `num_events` events.

        Args:
            num_events: number of events during the simulation.
            scheduler_kwargs: parameter scheduler configuration kwargs.

        Returns:
            event_index, value

        Examples:

        .. code-block:: python

            lr_values = np.array(LinearCyclicalScheduler.simulate_values(num_events=50, param_name='lr',
                                                                         start_value=1e-1, end_value=1e-3,
                                                                         cycle_size=10))

            plt.plot(lr_values[:, 0], lr_values[:, 1], label="learning rate")
            plt.xlabel("events")
            plt.ylabel("values")
            plt.legend()

        """
        keys_to_remove = ["optimizer", "save_history"]
        for key in keys_to_remove:
            if key in scheduler_kwargs:
                del scheduler_kwargs[key]
        values = []
        scheduler = cls(optimizer=_get_fake_optimizer(), save_history=False, **scheduler_kwargs)
        for i in range(num_events):
            scheduler(engine=None)
            values.append([i, scheduler.optimizer_param_groups[0][scheduler.param_name]])
        return values

    def _get_param(self) -> list[float] | float:
        # `ParamScheduler` does nothing special, only returning what child class returns.
        # Intermediate child classes edit this method
        return self.get_param()


class CyclicalScheduler(ParamScheduler):
    """An abstract class for updating an optimizer's parameter value over a
    cycle of some size.

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        start_value: value at start of cycle.
        end_value: value at the middle of the cycle.
        cycle_size: length of cycle, value should be larger than 1.
        cycle_mult: ratio by which to change the cycle_size.
            at the end of each cycle (default=1.0).
        start_value_mult: ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult: ratio by which to change the end value at the
            end of each cycle (default=1.0).
        warmup_duration: duration of warm-up to be applied before each cycle.
            Through this warm-up, the parameter starts from the last cycle's end value
            and linearly goes to next cycle's start value. Default is no cyclic warm-up.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    .. versionadded:: 0.4.5

    .. versionchanged:: 0.4.13
        Added cyclic warm-up to the scheduler using ``warmup_duration``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str,
        start_value: float,
        end_value: float,
        cycle_size: int,
        cycle_mult: float = 1.0,
        start_value_mult: float = 1.0,
        end_value_mult: float = 1.0,
        warmup_duration: int = 0,
        save_history: bool = False,
        param_group_index: int | None = None,
    ):
        super(CyclicalScheduler, self).__init__(
            optimizer, param_name, save_history=save_history, param_group_index=param_group_index
        )
        self.start_value = start_value
        self.end_value = end_value
        self.cycle_size = cycle_size
        self.cycle_mult = cycle_mult
        self.cycle = 0
        self.start_value_mult = start_value_mult
        self.end_value_mult = end_value_mult
        self.warmup_duration = warmup_duration
        self.total_cycle_size = self.warmup_duration + self.cycle_size

        if self.cycle_size < 2:
            raise ValueError(f"Argument cycle_size should be positive and larger than 1, but given {cycle_size}")

        self._state_attrs += [
            "start_value",
            "end_value",
            "cycle_size",
            "cycle_mult",
            "cycle",
            "start_value_mult",
            "end_value_mult",
            "warmup_duration",
            "total_cycle_size",
        ]

    def __call__(self, engine: Engine | None, name: str | None = None) -> None:
        if self.event_index != 0 and self.event_index == self.cycle_size:
            self.start_value *= self.start_value_mult
        if self.event_index != 0 and self.event_index == self.total_cycle_size:
            self.event_index = 0
            self.cycle_size = int(self.cycle_size * self.cycle_mult)
            self.warmup_duration = int(self.warmup_duration * self.cycle_mult)
            self.total_cycle_size = self.warmup_duration + self.cycle_size
            self.cycle += 1
            self.end_value *= self.end_value_mult

        return super(CyclicalScheduler, self).__call__(engine, name)

    def _get_param(self) -> list[float] | float:
        """Applies warm-up if the scheduler is in the warm-up phase,
        otherwise returns what is returned by `self.get_param()`
        """
        if self.event_index > self.cycle_size:
            warmup_progress = (self.event_index - self.cycle_size) / self.warmup_duration
            return self.end_value + (self.start_value - self.end_value) * warmup_progress

        return self.get_param()


class LinearCyclicalScheduler(CyclicalScheduler):
    """Linearly adjusts param value to 'end_value' for a half-cycle, then linearly
    adjusts it back to 'start_value' for a half-cycle.

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        start_value: value at start of cycle.
        end_value: value at the middle of the cycle.
        cycle_size: length of cycle.
        cycle_mult: ratio by which to change the cycle_size
            at the end of each cycle (default=1).
        start_value_mult: ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult: ratio by which to change the end value at the
            end of each cycle (default=1.0).
        warmup_duration: duration of warm-up to be applied before each cycle.
            Through this warm-up, the parameter starts from the last cycle's end value
            and linearly goes to next cycle's start value. Default is no cyclic warm-up.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.
        monotonic: whether to schedule only one half of the cycle: descending or ascending.
            If True, this argument can not be used together with ``warmup_duration``.
            (default=False).

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            # Linearly increases the learning rate from 0.0 to 1.0 and back to 0.0
            # over a cycle of 4 iterations
            scheduler = LinearCyclicalScheduler(default_optimizer, "lr", 0.0, 1.0, 4)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 1

            0.0
            0.5
            1.0
            0.5
            ...

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.001},
                    {"params": default_model.fc.parameters(), "lr": 0.01},
                ]
            )

            # Linearly increases the learning rate from 0.0 to 1.0 and back to 0.0
            # over a cycle of 4 iterations
            scheduler1 = LinearCyclicalScheduler(optimizer, "lr (base)", 0.0, 1.0, 4, param_group_index=0)

            # Linearly increases the learning rate from 0.0 to 0.1 and back to 0.0
            # over a cycle of 4 iterations
            scheduler2 = LinearCyclicalScheduler(optimizer, "lr (fc)", 0.0, 0.1, 4, param_group_index=1)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr (base)"],
                      optimizer.param_groups[1]["lr (fc)"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 2

            0.0 0.0
            0.5 0.05
            1.0 0.1
            0.5 0.05
            ...

    .. versionadded:: 0.4.5

    .. versionchanged:: 0.4.13
        Added cyclic warm-up to the scheduler using ``warmup_duration``.

    .. versionchanged:: 0.5.0
        Added monotonic argument.
    """

    def __init__(self, *args: Any, monotonic: bool = False, **kwagrs: Any):
        super(LinearCyclicalScheduler, self).__init__(*args, **kwagrs)
        self.monotonic = monotonic
        if self.warmup_duration > 0 and not self.monotonic:
            raise ValueError(
                "Invalid combination when warmup_duration > 0 and monotonic=False, "
                "please use either set warmup_duration=0 or monotonic=True"
            )

    def get_param(self) -> float:
        """Method to get current optimizer's parameter value"""
        cycle_progress = self.event_index / self.cycle_size

        if self.monotonic:
            return self.start_value + (self.end_value - self.start_value) * cycle_progress
        else:
            return self.end_value + (self.start_value - self.end_value) * abs(cycle_progress - 0.5) * 2


class CosineAnnealingScheduler(CyclicalScheduler):
    """Anneals 'start_value' to 'end_value' over each cycle.

    The annealing takes the form of the first half of a cosine
    wave (as suggested in [Smith17]_).

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        start_value: value at start of cycle.
        end_value: value at the end of the cycle.
        cycle_size: length of cycle.
        cycle_mult: ratio by which to change the cycle_size
            at the end of each cycle (default=1).
        start_value_mult: ratio by which to change the start value at the
            end of each cycle (default=1.0).
        end_value_mult: ratio by which to change the end value at the
            end of each cycle (default=1.0).
        warmup_duration: duration of warm-up to be applied before each cycle.
            Through this warm-up, the parameter starts from the last cycle's end value
            and linearly goes to next cycle's start value. Default is no cyclic warm-up.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.

    Note:
        If the scheduler is bound to an 'ITERATION_*' event, 'cycle_size' should
        usually be the number of batches in an epoch.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            # CosineAnnealing increases the learning rate from 0.0 to 1.0
            # over a cycle of 4 iterations
            scheduler = CosineAnnealingScheduler(default_optimizer, "lr", 0.0, 1.0, 4)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 1

            0.0
            0.1464...
            0.4999...
            0.8535...
            ...

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.001},
                    {"params": default_model.fc.parameters(), "lr": 0.01},
                ]
            )

            # CosineAnnealing increases the learning rate from 0.0 to 1.0
            # over a cycle of 4 iterations
            scheduler_1 = CosineAnnealingScheduler(optimizer, "lr (base)", 0.0, 1.0, 4, param_group_index=0)

            # CosineAnnealing increases the learning rate from 0.0 to 0.1
            # over a cycle of 4 iterations
            scheduler_2 = CosineAnnealingScheduler(optimizer, "lr (fc)", 0.0, 0.1, 4, param_group_index=1)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr (base)"],
                      optimizer.param_groups[1]["lr (fc)"])

            default_trainer.run([0] * 9, max_epochs=1)

        .. testoutput:: 2

            0.0 0.0
            0.1464... 0.01464...
            0.4999... 0.04999...
            0.8535... 0.08535...
            ...

    .. [Smith17] Smith, Leslie N. "Cyclical learning rates for training neural networks."
                 Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017

    .. versionadded:: 0.4.5

    .. versionchanged:: 0.4.13
        Added cyclic warm-up to the scheduler using ``warmup_duration``.
    """

    def get_param(self) -> float:
        """Method to get current optimizer's parameter value"""
        cycle_progress = self.event_index / self.cycle_size
        return self.start_value + ((self.end_value - self.start_value) / 2) * (1 - math.cos(math.pi * cycle_progress))


class ConcatScheduler(ParamScheduler):
    """Concat a list of parameter schedulers.

    The `ConcatScheduler` goes through a list of schedulers given by `schedulers`. Duration of each
    scheduler is defined by `durations` list of integers.

    Args:
        schedulers: list of parameter schedulers.
        durations: list of number of events that lasts a parameter scheduler from schedulers.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            scheduler_1 = LinearCyclicalScheduler(default_optimizer, "lr", 0.0, 1.0, 8)
            scheduler_2 = CosineAnnealingScheduler(default_optimizer, "lr", 1.0, 0.2, 4)

            # Sets the Learning rate linearly from 0.0 to 1.0 over 4 iterations. Then
            # starts an annealing schedule from 1.0 to 0.2 over the next 4 iterations.
            # The annealing cycles are repeated indefinitely.
            combined_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2], durations=[4, ])

            default_trainer.add_event_handler(Events.ITERATION_STARTED, combined_scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.0
            0.25
            0.5
            0.75
            1.0
            0.8828...
            0.6000...
            0.3171...

    .. versionadded:: 0.4.5
    """

    def __init__(self, schedulers: list[ParamScheduler], durations: list[int], save_history: bool = False):
        if not isinstance(schedulers, Sequence):
            raise TypeError(f"Argument schedulers should be a sequence, but given {schedulers}")

        if len(schedulers) < 2:
            raise ValueError(
                f"Argument schedulers should be of more than one parameter schedulers, but given {schedulers}"
            )

        if not isinstance(durations, (list, tuple)):
            raise TypeError(f"Argument durations should be list/tuple, but given {durations}")

        if not all([isinstance(t, numbers.Integral) for t in durations]):
            raise ValueError(f"Argument durations should be list/tuple of integers, but given {durations}")

        if len(schedulers) != len(durations) + 1:
            raise ValueError(
                "Incorrect number schedulers or duration values, " f"given {len(schedulers)} and {len(durations)}"
            )

        for i, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, ParamScheduler) and not isinstance(scheduler, ParamGroupScheduler):
                raise TypeError(
                    f"Value at index {i} of schedulers should be a parameter scheduler, but given {type(scheduler)}"
                )

        self.schedulers = schedulers
        self.durations = durations

        tmp_optimizers = [s.optimizer for s in self.schedulers]
        tmp_list_optimizers = [s if isinstance(s, list) else [s] for s in tmp_optimizers]
        param_optimizers = list(itertools.chain(*tmp_list_optimizers))

        optimizer = list(set(param_optimizers))
        if len(optimizer) != 1:
            raise ValueError("schedulers should be related to same optimizer")

        tmp_param_names = [s.param_name for s in self.schedulers]
        tmp_list_param_names = [s if isinstance(s, list) else [s] for s in tmp_param_names]
        param_names = list(itertools.chain(*tmp_list_param_names))

        param_name = list(set(param_names))
        if len(param_name) != 1:
            raise ValueError("schedulers should be related to same param_name")

        # schedulers should have save_history sync with ParamGroupScheduler
        for s in schedulers:
            s.save_history = save_history

        super(ConcatScheduler, self).__init__(
            optimizer=optimizer[0], param_name=param_name[0], save_history=save_history
        )

        self._scheduler_index = 0
        self._setup_scheduler()
        self._state_attrs += ["_current_duration", "durations", "_scheduler_index"]

    def state_dict(self) -> dict[str, Any]:
        """Returns a dictionary containing a whole state of ConcatScheduler.

        Returns:
            dict:
                a dictionary containing a whole state of ConcatScheduler
        """

        state_dict = super(ConcatScheduler, self).state_dict()
        state_dict["schedulers"] = []
        for s in self.schedulers:
            state_dict["schedulers"].append(s.state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Copies parameters from :attr:`state_dict` into this ConcatScheduler.

        Args:
            state_dict: a dict containing parameters.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        if "schedulers" not in state_dict:
            raise ValueError(
                f"Required state attribute 'schedulers' is absent in provided state_dict '{state_dict.keys()}'"
            )
        sds = state_dict["schedulers"]
        if len(sds) != len(self.schedulers):
            raise ValueError(
                f"Input state_dict contains {len(sds)} state_dicts of concatenated schedulers, "
                f"but {len(self.schedulers)} needed"
            )

        for s, sd in zip(self.schedulers, sds, strict=False):
            s.load_state_dict(sd)
        super(ConcatScheduler, self).load_state_dict(state_dict)
        self._current_scheduler = self.schedulers[self._scheduler_index]

    def _setup_scheduler(self) -> None:
        self._current_scheduler = self.schedulers[self._scheduler_index]
        self._current_duration = (
            self.durations[self._scheduler_index] if self._scheduler_index < len(self.durations) else -1
        )

    def __call__(self, engine: Engine | None, name: str | None = None) -> None:
        if self._current_duration == 0:
            self._scheduler_index += 1
            self._setup_scheduler()
        self._current_scheduler(engine, name)
        self._current_duration -= 1

    @property
    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        # We need to setup optimizer_param_groups as property
        # to synchonize with the latest _current_scheduler and its internal optimizer_param_groups
        return self._current_scheduler.optimizer_param_groups

    @property
    def save_history(self) -> bool:
        return self._current_scheduler.save_history

    @save_history.setter
    def save_history(self, value: bool) -> None:
        for s in self.schedulers:
            s.save_history = value

    def get_param(self) -> list[float] | float:
        return self._current_scheduler.get_param()

    @classmethod
    def simulate_values(  # type: ignore[override]
        cls,
        num_events: int,
        schedulers: list[ParamScheduler],
        durations: list[int],
        param_names: list[str] | tuple[str] | None = None,
    ) -> list[list[int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            schedulers: list of parameter schedulers.
            durations: list of number of events that lasts a parameter scheduler from schedulers.
            param_names: parameter name or list of parameter names to simulate values.
                By default, the first scheduler's parameter name is taken.

        Returns:
            list:
                list of [event_index, value_0, value_1, ...], where values correspond to `param_names`.
        """
        if param_names is not None:
            if not isinstance(param_names, (list, tuple)):
                raise TypeError(f"Argument param_names should be list or tuple, but given {type(param_names)}")
            if not all(isinstance(item, str) for item in param_names):
                raise ValueError(f"Argument param_names should be list or tuple of strings, but given {param_names}")

        tmp_param_optimizers = [s.optimizer for s in schedulers]
        tmp_list_param_optimizers = [s if isinstance(s, list) else [s] for s in tmp_param_optimizers]
        param_optimizers = list(itertools.chain(*tmp_list_param_optimizers))

        tmp_optimizer = list(set(param_optimizers))
        if len(tmp_optimizer) != 1:
            raise ValueError("schedulers should be related to same optimizer")

        optimizer = tmp_optimizer[0]

        # This scheduler uses `ParamScheduler` which
        # should be replicated in order to simulate LR values and
        # not perturb original scheduler.
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_filepath = Path(tmpdirname) / "ignite_lr_scheduler_cache.pt"
            objs = {f"lr_scheduler_{i}": s.state_dict() for i, s in enumerate(schedulers)}
            # all schedulers should be related to the same optimizer
            objs["optimizer"] = optimizer.state_dict()

            torch.save(objs, cache_filepath.as_posix())

            # do not save_history
            for s in schedulers:
                s.save_history = False

            output = []
            scheduler = cls(schedulers=schedulers, save_history=False, durations=durations)
            if param_names is None:
                param_names = [scheduler.param_name]
            for i in range(num_events):
                scheduler(engine=None)
                values = [i]
                for param_name in param_names:
                    params = [p[param_name] for p in scheduler.optimizer_param_groups]
                    values = values + params
                output.append(values)

            objs = torch.load(cache_filepath.as_posix())
            for i, s in enumerate(schedulers):
                s.load_state_dict(objs[f"lr_scheduler_{i}"])
            optimizer.load_state_dict(objs["optimizer"])

            return output


class _CosineAnnealingWarmRestarts:
    def __init__(self, lr_scheduler: CosineAnnealingWarmRestarts):
        self._lr_scheduler = lr_scheduler

    @property
    def last_epoch(self) -> int:
        return self._lr_scheduler.last_epoch

    @last_epoch.setter
    def last_epoch(self, value: int) -> None:
        self._lr_scheduler.last_epoch = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._lr_scheduler.optimizer

    def get_lr(self, epoch: int | None = None) -> list[float]:
        T_mult = self._lr_scheduler.T_mult
        eta_min = self._lr_scheduler.eta_min

        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
            self._lr_scheduler.T_cur = self._lr_scheduler.T_cur + 1
            if self._lr_scheduler.T_cur >= self._lr_scheduler.T_i:
                self._lr_scheduler.T_cur = self._lr_scheduler.T_cur - self._lr_scheduler.T_i
                self._lr_scheduler.T_i = self._lr_scheduler.T_i * T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self._lr_scheduler.T_0:
                if T_mult == 1:
                    self._lr_scheduler.T_cur = epoch % self._lr_scheduler.T_0
                else:
                    n = int(math.log((epoch / self._lr_scheduler.T_0 * (T_mult - 1) + 1), T_mult))
                    self._lr_scheduler.T_cur = epoch - self._lr_scheduler.T_0 * (T_mult**n - 1) / (T_mult - 1)
                    self._lr_scheduler.T_i = self._lr_scheduler.T_0 * T_mult**n
            else:
                self._lr_scheduler.T_i = self._lr_scheduler.T_0
                self._lr_scheduler.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        return [
            eta_min
            + (base_lr - eta_min) * (1 + math.cos(math.pi * self._lr_scheduler.T_cur / self._lr_scheduler.T_i)) / 2
            for base_lr in self._lr_scheduler.base_lrs
        ]


class LRScheduler(ParamScheduler):
    """A wrapper class to call `torch.optim.lr_scheduler` objects as `ignite` handlers.

    Args:
        lr_scheduler: lr_scheduler object to wrap.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        use_legacy: if True, scheduler should be attached to ``Events.ITERATION_COMPLETED``, (default=False).

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            from torch.optim.lr_scheduler import StepLR

            torch_lr_scheduler = StepLR(default_optimizer, step_size=3, gamma=0.1)
            scheduler = LRScheduler(torch_lr_scheduler)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.1
            0.1
            0.1
            0.010...
            0.010...
            0.010...
            0.001...
            0.001...

    .. versionadded:: 0.4.5

    ..  versionchanged:: 0.4.9
        added `use_legacy` argument
    """

    def __init__(
        self,
        lr_scheduler: PyTorchLRScheduler,
        save_history: bool = False,
        use_legacy: bool = False,
    ):
        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        self.lr_scheduler: PyTorchLRScheduler | _CosineAnnealingWarmRestarts = lr_scheduler
        if isinstance(lr_scheduler, CosineAnnealingWarmRestarts):
            self.lr_scheduler = _CosineAnnealingWarmRestarts(lr_scheduler)

        super(LRScheduler, self).__init__(
            optimizer=self.lr_scheduler.optimizer,
            param_name="lr",
            save_history=save_history,
        )
        if use_legacy:
            warnings.warn(
                "Please make sure to attach scheduler to Events.ITERATION_COMPLETED "
                "instead of Events.ITERATION_STARTED to make sure to use "
                "the first lr value from the optimizer, otherwise it will be skipped"
            )
            self.lr_scheduler.last_epoch += 1

        self._state_attrs += ["lr_scheduler"]

    def __call__(self, engine: Engine | None, name: str | None = None) -> None:
        super(LRScheduler, self).__call__(engine, name)
        self.lr_scheduler.last_epoch += 1

    def get_param(self) -> float | list[float]:
        """Method to get current optimizer's parameter value"""
        # Emulate context manager for pytorch>=1.4
        self.lr_scheduler._get_lr_called_within_step = True  # type: ignore[union-attr]
        lr_list = self.lr_scheduler.get_lr()
        self.lr_scheduler._get_lr_called_within_step = False  # type: ignore[union-attr]
        if len(lr_list) == 1:
            return lr_list[0]
        else:
            return lr_list

    @classmethod
    def simulate_values(  # type: ignore[override]
        cls, num_events: int, lr_scheduler: PyTorchLRScheduler, **kwargs: Any
    ) -> list[list[int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            lr_scheduler: lr_scheduler object to wrap.

        Returns:
            event_index, value
        """

        if not isinstance(lr_scheduler, PyTorchLRScheduler):
            raise TypeError(
                "Argument lr_scheduler should be a subclass of "
                f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__}, "
                f"but given {type(lr_scheduler)}"
            )

        # This scheduler uses `torch.optim.lr_scheduler.LRScheduler` which
        # should be replicated in order to simulate LR values and
        # not perturb original scheduler.
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_filepath = Path(tmpdirname) / "ignite_lr_scheduler_cache.pt"
            obj = {
                "lr_scheduler": lr_scheduler.state_dict(),
                "optimizer": lr_scheduler.optimizer.state_dict(),
            }
            torch.save(obj, cache_filepath.as_posix())

            values = []
            scheduler = cls(save_history=False, lr_scheduler=lr_scheduler, **kwargs)
            for i in range(num_events):
                scheduler(engine=None)
                params = [p[scheduler.param_name] for p in scheduler.optimizer_param_groups]
                values.append([i] + params)

            obj = torch.load(cache_filepath.as_posix())
            lr_scheduler.load_state_dict(obj["lr_scheduler"])
            lr_scheduler.optimizer.load_state_dict(obj["optimizer"])

            return values


def create_lr_scheduler_with_warmup(
    lr_scheduler: ParamScheduler | PyTorchLRScheduler,
    warmup_start_value: float,
    warmup_duration: int,
    warmup_end_value: float | None = None,
    save_history: bool = False,
    output_simulated_values: list | None = None,
) -> "ConcatScheduler":
    """
    Helper method to create a learning rate scheduler with a linear warm-up.

    Args:
        lr_scheduler: learning rate scheduler after the warm-up.
        warmup_start_value: learning rate start value of the warm-up phase.
        warmup_duration: warm-up phase duration, number of events.
        warmup_end_value: learning rate end value of the warm-up phase, (default=None). If None,
             warmup_end_value is set to optimizer initial lr.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        output_simulated_values: optional output of simulated learning rate values.
            If output_simulated_values is a list of None, e.g. `[None] * 100`, after the execution it will be filled
            by 100 simulated learning rate values.

    Returns:
        ConcatScheduler

    Note:
        If the first learning rate value provided by `lr_scheduler` is different from `warmup_end_value`, an additional
        event is added after the warm-up phase such that the warm-up ends with `warmup_end_value` value and then
        `lr_scheduler` provides its learning rate values as normally.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            from torch.optim.lr_scheduler import ExponentialLR

            torch_lr_scheduler = ExponentialLR(optimizer=default_optimizer, gamma=0.98)

            default_trainer = get_default_trainer()

            scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                        warmup_start_value=0.0,
                                                        warmup_end_value=0.1,
                                                        warmup_duration=3)

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.0
            0.05
            0.1
            0.098
            0.09604
            0.09411...
            0.09223...
            0.09039...

    .. versionadded:: 0.4.5
    """
    if not isinstance(lr_scheduler, (ParamScheduler, PyTorchLRScheduler)):
        raise TypeError(
            "Argument lr_scheduler should be a subclass of "
            f"torch.optim.lr_scheduler.{PyTorchLRScheduler.__name__} or ParamScheduler, "
            f"but given {type(lr_scheduler)}"
        )

    if not isinstance(warmup_duration, numbers.Integral):
        raise TypeError(f"Argument warmup_duration should be integer, but given {warmup_duration}")

    if not (warmup_duration > 1):
        raise ValueError(f"Argument warmup_duration should be at least 2 events, but given {warmup_duration}")

    warmup_schedulers: list[ParamScheduler] = []

    for param_group_index, param_group in enumerate(lr_scheduler.optimizer.param_groups):
        if warmup_end_value is None:
            param_group_warmup_end_value = param_group["lr"]
        else:
            param_group_warmup_end_value = warmup_end_value

        milestones_values = [(0, warmup_start_value), (warmup_duration - 1, param_group_warmup_end_value)]

        if isinstance(lr_scheduler, PyTorchLRScheduler):
            init_lr = param_group["lr"]
            if init_lr != param_group_warmup_end_value:
                milestones_values.append((warmup_duration, init_lr))

            # We need to advance torch lr_scheduler to avoid duplicated lr value
            # given by PiecewiseLinear and LRScheduler.
            # We suggest to attach output scheduler on ITERATION_STARTED but
            # torch lr_scheduler works with ITERATION_COMPLETED
            # See also https://github.com/pytorch/ignite/pull/2496#issuecomment-1065984440
            lr_scheduler.last_epoch += 1
            lr_scheduler = LRScheduler(lr_scheduler, save_history=save_history)
        else:
            init_lr = lr_scheduler.get_param()
            if init_lr == param_group_warmup_end_value:
                if warmup_duration > 2:
                    d = (param_group_warmup_end_value - warmup_start_value) / (warmup_duration - 1)
                    milestones_values[-1] = (warmup_duration - 2, param_group_warmup_end_value - d)
                else:
                    milestones_values.pop(-1)

        warmup_schedulers.append(
            PiecewiseLinear(
                lr_scheduler.optimizer,
                param_name="lr",
                milestones_values=milestones_values,
                param_group_index=param_group_index,
                save_history=save_history,
            )
        )

    warmup_scheduler = ParamGroupScheduler(warmup_schedulers, save_history=save_history)

    schedulers: list[ParamScheduler | ParamGroupScheduler | PyTorchLRScheduler] = [
        warmup_scheduler,
        lr_scheduler,
    ]
    durations = [milestones_values[-1][0] + 1]
    combined_scheduler = ConcatScheduler(schedulers, durations=durations, save_history=save_history)

    if output_simulated_values is not None:
        if not isinstance(output_simulated_values, list):
            raise TypeError(
                "Argument output_simulated_values should be a list of None, e.g. `[None] * 100`, "
                f"but given {type(output_simulated_values)}."
            )
        num_events = len(output_simulated_values)
        result = ConcatScheduler.simulate_values(num_events=num_events, schedulers=schedulers, durations=durations)
        for i in range(num_events):
            output_simulated_values[i] = result[i]
    return combined_scheduler


class PiecewiseLinear(ParamScheduler):
    """
    Piecewise linear parameter scheduler

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: name of optimizer's parameter to update.
        milestones_values: list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.
        save_history: whether to log the parameter values to
            `engine.state.param_history`, (default=False).
        param_group_index: optimizer's parameters group to use.

    .. code-block:: python

        scheduler = PiecewiseLinear(optimizer, "lr",
                                    milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
        # Attach to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        #
        # Sets the learning rate to 0.5 over the first 10 iterations, then decreases linearly from 0.5 to 0.45 between
        # 10th and 20th iterations. Next there is a jump to 0.3 at the 21st iteration and LR decreases linearly
        # from 0.3 to 0.1 between 21st and 30th iterations and remains 0.1 until the end of the iterations.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode:: 1

            default_trainer = get_default_trainer()

            milestones_values = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler = PiecewiseLinear(
                default_optimizer, "lr", milestones_values=milestones_values)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(default_optimizer.param_groups[0]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 1

            1.0
            1.0
            0.9
            0.8
            0.5
            0.2

        .. testcode:: 2

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.1},
                    {"params": default_model.fc.parameters(), "lr": 1.0},
                ]
            )

            milestones_values1 = [(1, 0.1), (3, 0.08), (5, 0.02)]
            scheduler2 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values1, param_group_index=0)
            # Sets lr equal to 0.1 for till the first iteration
            # Then linearly reduces lr from 0.1 to 0.08 till the third iteration
            # Then linearly reduces lr from 0.08 to 0.05 till the fifth iteration

            milestones_values2 = [(1, 1.0), (3, 0.8), (5, 0.2)]
            scheduler1 = PiecewiseLinear(
                optimizer, "lr", milestones_values=milestones_values2, param_group_index=1)
            # Sets lr equal to 1 for till the first iteration
            # Then linearly reduces lr from 1 to 0.8 till the third iteration
            # Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler1)
            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler2)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr"],
                      optimizer.param_groups[1]["lr"])

            default_trainer.run([0] * 6, max_epochs=1)

        .. testoutput:: 2

            0.1 1.0
            0.1 1.0
            0.09 0.9
            0.08 0.8
            0.05 0.5
            0.02 0.2

    .. versionadded:: 0.4.5
    """

    def __init__(
        self,
        optimizer: Optimizer,
        param_name: str,
        milestones_values: list[tuple[int, float]],
        save_history: bool = False,
        param_group_index: int | None = None,
    ):
        super(PiecewiseLinear, self).__init__(optimizer, param_name, save_history, param_group_index=param_group_index)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f"Argument milestones_values should be a list or tuple, but given {type(milestones_values)}"
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f"Argument milestones_values should be with at least one value, but given {milestones_values}"
            )

        values: list[float] = []
        milestones: list[int] = []
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(f"Value of a milestone should be integer, but given {type(pair[0])}")
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f"Milestones should be increasing integers, but given {pair[0]} is smaller "
                    f"than the previous milestone {milestones[-1]}"
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0
        self._state_attrs += ["values", "milestones", "_index"]

    def _get_start_end(self) -> tuple[int, int, float, float]:
        if self.milestones[0] > self.event_index:
            return self.event_index - 1, self.event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.event_index:
            return (self.event_index, self.event_index + 1, self.values[-1], self.values[-1])
        elif self.milestones[self._index] <= self.event_index < self.milestones[self._index + 1]:
            return (
                self.milestones[self._index],
                self.milestones[self._index + 1],
                self.values[self._index],
                self.values[self._index + 1],
            )
        else:
            self._index += 1
            return self._get_start_end()

    def get_param(self) -> float:
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)


class ParamGroupScheduler:
    """
    Scheduler helper to group multiple schedulers into one.

    Args:
        schedulers: list/tuple of parameter schedulers.
        names: list of names of schedulers.
        save_history: whether to save history or not.

    Examples:

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            optimizer = torch.optim.SGD(
                [
                    {"params": default_model.base.parameters(), "lr": 0.001},
                    {"params": default_model.fc.parameters(), "lr": 0.01},
                ]
            )

            # CosineAnnealing increases the learning rate from 0.0 to 1.0
            # over a cycle of 4 iterations
            scheduler_1 = CosineAnnealingScheduler(optimizer, "lr", 0.0, 1.0, 4, param_group_index=0)

            # CosineAnnealing increases the learning rate from 0.0 to 0.1
            # over a cycle of 4 iterations
            scheduler_2 = CosineAnnealingScheduler(optimizer, "lr", 0.0, 0.1, 4, param_group_index=1)

            scheduler = ParamGroupScheduler(schedulers=[scheduler_1, scheduler_2],
                                            names=["lr (base)", "lr (fc)"])

            default_trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def print_lr():
                print(optimizer.param_groups[0]["lr"],
                      optimizer.param_groups[1]["lr"])

            default_trainer.run([0] * 8, max_epochs=1)

        .. testoutput::

            0.0 0.0
            0.1464... 0.01464...
            0.4999... 0.04999...
            0.8535... 0.08535...
            ...

    .. versionadded:: 0.4.5
    """

    def __init__(self, schedulers: list[ParamScheduler], names: list[str] | None = None, save_history: bool = False):
        if not isinstance(schedulers, Sequence):
            raise TypeError(f"Argument schedulers should be a list/tuple, but given {schedulers}")

        if not all(isinstance(scheduler, ParamScheduler) for scheduler in schedulers):
            raise ValueError(
                f"Argument schedulers should be a list/tuple of parameter schedulers, but given {schedulers}"
            )

        if names is None:
            names = [s.param_name for s in schedulers]

        if not isinstance(names, (list, tuple)):
            raise TypeError(f"Argument names should be a list/tuple, but given {names}")

        if not all(isinstance(n, str) for n in names):
            raise ValueError(f"Argument names should be a list/tuple of parameter scheduler's names, but given {names}")

        if len(names) != len(schedulers):
            raise ValueError(f"{len(schedulers)} should be equal {len(names)}")

        self.schedulers = schedulers
        self.names = names

        # schedulers should have save_history sync with ParamGroupScheduler
        for s in schedulers:
            s.save_history = save_history

        self.optimizer = [s.optimizer for s in self.schedulers]
        self.param_name = [s.param_name for s in self.schedulers]

    def __call__(self, engine: Engine | None, name: str | None = None) -> None:
        for scheduler, name in zip(self.schedulers, self.names, strict=False):
            scheduler(engine, name)

    @property
    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        return [pg for scheduler in self.schedulers for pg in scheduler.optimizer_param_groups]

    @property
    def save_history(self) -> bool:
        return self.schedulers[0].save_history

    @save_history.setter
    def save_history(self, value: bool) -> None:
        for s in self.schedulers:
            s.save_history = value

    def state_dict(self) -> dict[str, list[Any]]:
        """Returns a dictionary containing a whole state of ParamGroupScheduler.

        Returns:
            dict:
                a dictionary containing a whole state of ParamGroupScheduler
        """
        state_dict: dict[str, list[Any]] = OrderedDict()
        state_dict["schedulers"] = []
        for n, s in zip(self.names, self.schedulers, strict=False):
            state_dict["schedulers"].append((n, s.state_dict()))
        return state_dict

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Copies parameters from :attr:`state_dict` into this ParamScheduler.

        Args:
            state_dict: a dict containing parameters.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        if "schedulers" not in state_dict:
            raise ValueError(
                f"Required state attribute '{'schedulers'}' is absent in provided state_dict '{state_dict.keys()}'"
            )
        sds = state_dict["schedulers"]
        if len(sds) != len(self.schedulers):
            raise ValueError(
                f"Input state_dict contains {len(sds)} state_dicts of param group schedulers, "
                f"but {len(self.schedulers)} needed"
            )

        for req_n, s, (n, sd) in zip(self.names, self.schedulers, sds, strict=False):
            if req_n != n:
                raise ValueError(
                    f"Name of scheduler from input state dict does not correspond to required one, {n} vs {req_n}"
                )
            s.load_state_dict(sd)

    @classmethod
    def simulate_values(
        cls, num_events: int, schedulers: list[ParamScheduler], **kwargs: Any
    ) -> list[list[list[float] | float | int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            schedulers: lr_scheduler object to wrap.
            kwargs: kwargs passed to construct an instance of
                :class:`ignite.handlers.param_scheduler.ParamGroupScheduler`.

        Returns:
            list:
                list of [event_index, scheduler_0_value, scheduler_1_value, ...], where scheduler_i_value
                corresponds to the simulated param of scheduler i at 'event_index'th event.
        """

        # This scheduler uses `torch.optim.lr_scheduler.LRScheduler` which
        # should be replicated in order to simulate LR values and
        # not perturb original scheduler.
        with tempfile.TemporaryDirectory() as tmpdirname:
            cache_filepath = Path(tmpdirname) / "ignite_lr_scheduler_cache.pt"
            objs = {f"lr_scheduler_{i}": s.state_dict() for i, s in enumerate(schedulers)}
            # all schedulers should be related to the same optimizer
            objs["optimizer"] = schedulers[0].optimizer.state_dict()

            torch.save(objs, cache_filepath.as_posix())

            values = []
            scheduler = cls(schedulers=schedulers, **kwargs)
            for i in range(num_events):
                params = [scheduler.get_param() for scheduler in schedulers]
                values.append([i] + params)
                scheduler(engine=None)

            objs = torch.load(cache_filepath.as_posix())
            for i, s in enumerate(schedulers):
                s.load_state_dict(objs[f"lr_scheduler_{i}"])
                s.optimizer.load_state_dict(objs["optimizer"])

            return values

    def get_param(self) -> list[float | list[float]]:
        """
        Method to get current `schedulers`' parameter values

        .. versionadded:: 0.4.11
        """
        return [scheduler.get_param() for scheduler in self.schedulers]


class ReduceLROnPlateauScheduler(ParamScheduler):
    """Reduce LR when a metric stops improving.
    Wrapper of `torch.optim.lr_scheduler.ReduceLROnPlateau
    <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>`_.

    Args:
        optimizer: Wrapped optimizer.
        metric_name: metric whose improvement is monitored.
            Must be attached to the same engine.
        trainer: Trainer engine to log LR history in its
            `state.output.param_history`. Is used if `save_history`
            is true. Default: None.
        save_history: Whether to save history or not. If true,
            history will be logged in `trainer`'s `state.output.param_history`.
            Default: False.
        param_group_index: `optimizer`'s parameters group
            to use.  Default: None. Use all `optimizer`'s paramater groups.
        scheduler_kwargs: Keyword arguments to be passed to the wrapped ``ReduceLROnPlateau``.

    Examples:

        .. code-block:: python

            # Metric "accuracy" should increase the best value by
            # more than 1 unit after at most 2 epochs, otherwise LR
            # would get multiplied by 0.5 .

            scheduler = ReduceLROnPlateauScheduler(
                default_optimizer,
                metric_name="accuracy", mode="max",
                factor=0.5, patience=1, threshold_mode='abs',
                threshold=1, trainer=trainer
            )

            metric = Accuracy()
            default_evaluator.attach(metric, "accuracy")

            default_evaluator.add_event_handler(Events.COMPLETED, scheduler)

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            default_trainer = get_default_trainer()

            # Metric "loss" should decrease more than
            # 0.1 of best loss after at most
            # three iterations. Then best loss would get
            # updated, otherwise lr is multiplied by 0.5

            scheduler = ReduceLROnPlateauScheduler(
                default_optimizer, "loss",
                save_history=True, mode="min",
                factor=0.5, patience=3, threshold_mode='rel',
                threshold=0.1, trainer=default_trainer
            )

            metric_values = iter([10, 5, 3, 4, 4, 4, 5, 1])
            default_evaluator.state.metrics = {"loss": None}

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def set_metric_val():
                default_evaluator.state.metrics["loss"] = next(metric_values)

            default_evaluator.add_event_handler(Events.COMPLETED, scheduler)

            @default_trainer.on(Events.ITERATION_COMPLETED)
            def trigger_eval():
                default_evaluator.run([0.])

            default_trainer.run([0.] * 8)

            print(default_trainer.state.param_history["lr"])

        .. testoutput::

            [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.05], [0.05]]

    .. versionadded:: 0.4.9
    """

    def __init__(
        self,
        optimizer: Optimizer,
        metric_name: str,
        trainer: Engine | None = None,
        save_history: bool = False,
        param_group_index: int | None = None,
        **scheduler_kwargs: Any,
    ):
        super(ReduceLROnPlateauScheduler, self).__init__(
            optimizer, "lr", save_history=save_history, param_group_index=param_group_index
        )
        self.metric_name = metric_name
        self.trainer = trainer
        self.optimizer = optimizer

        if "min_lr" in scheduler_kwargs and param_group_index is not None:
            min_lr = scheduler_kwargs["min_lr"]
            if not isinstance(min_lr, float):
                raise TypeError(f"When param_group_index is given, min_lr should be a float, but given {type(min_lr)}")
            _min_lr = min_lr
            min_lr = [0] * len(optimizer.param_groups)
            min_lr[param_group_index] = _min_lr
        else:
            min_lr = 0
        _scheduler_kwargs = scheduler_kwargs.copy()
        _scheduler_kwargs["min_lr"] = min_lr

        if "verbose" in _scheduler_kwargs:
            warnings.warn(
                "Found verbose=True in provided scheduler_kwargs. "
                "It would be set to False. Please use save_history instead."
            )
            _scheduler_kwargs["verbose"] = False

        self.scheduler = ReduceLROnPlateau(optimizer, **_scheduler_kwargs)
        self.scheduler._reduce_lr = self._reduce_lr  # type: ignore[method-assign]

        self._state_attrs += ["metric_name", "scheduler"]

    def __call__(self, engine: Engine, name: str | None = None) -> None:  # type: ignore[override]
        if not hasattr(engine.state, "metrics") or self.metric_name not in engine.state.metrics:
            raise ValueError(
                "Argument engine should have in its 'state', attribute 'metrics' "
                f"which itself has the metric {self.metric_name}."
            )
        self.scheduler.step(engine.state.metrics[self.metric_name])
        super().__call__(self.trainer, name)

    def get_param(self) -> float | list[float]:
        lrs = [pg["lr"] for pg in self.optimizer_param_groups]
        return lrs[0] if len(lrs) == 1 else lrs

    def _reduce_lr(self, epoch: int) -> None:
        for i, param_group in enumerate(self.optimizer_param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.scheduler.factor, self.scheduler.min_lrs[i])
            if old_lr - new_lr > self.scheduler.eps:
                param_group["lr"] = new_lr

    @classmethod
    def simulate_values(  # type: ignore[override]
        cls, num_events: int, metric_values: list[float], init_lr: float, **scheduler_kwargs: Any
    ) -> list[list[int]]:
        """Method to simulate scheduled values during num_events events.

        Args:
            num_events: number of events during the simulation.
            metric_values: values to change LR based on.
            init_lr: initial LR to start with.
            scheduler_kwargs: kwargs passed to construct an instance of
                :class:`ignite.handlers.param_scheduler.ReduceLROnPlateauScheduler`.

        Returns:
            event_index, value

        """
        if len(metric_values) != num_events:
            raise ValueError(
                "Length of argument metric_values should be equal to num_events. "
                f"{len(metric_values)} != {num_events}"
            )

        keys_to_remove = ["optimizer", "metric_name", "save_history"]
        for key in keys_to_remove:
            if key in scheduler_kwargs:
                del scheduler_kwargs[key]
        values = []
        scheduler = cls(
            optimizer=_get_fake_optimizer(torch.optim.SGD, lr=init_lr),
            metric_name="metric",
            save_history=False,
            **scheduler_kwargs,
        )
        engine = Engine(lambda _, __: None)
        for i in range(num_events):
            engine.state.metrics["metric"] = metric_values[i]
            scheduler(engine=engine)
            values.append([i, scheduler.optimizer_param_groups[0][scheduler.param_name]])
        return values


def _get_fake_optimizer(
    optimizer_cls: type[Optimizer] | type[torch.optim.SGD] | None = None, **kwargs: Any
) -> Optimizer | torch.optim.SGD:
    t = torch.zeros([1], requires_grad=True)
    if optimizer_cls is None:
        optimizer_cls = torch.optim.SGD
        kwargs["lr"] = 0.01
    return optimizer_cls([t], **kwargs)
