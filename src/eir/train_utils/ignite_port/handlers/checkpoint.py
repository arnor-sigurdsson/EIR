import collections.abc as collections
import numbers
import os
import stat
import tempfile
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
import torch.nn as nn

from eir.train_utils.ignite_port.base import Serializable
from eir.train_utils.ignite_port.engine import Engine
from eir.train_utils.ignite_port.events import Events
from eir.train_utils.ignite_port.utils import _tree_apply2, _tree_map

__all__ = ["Checkpoint", "DiskSaver", "ModelCheckpoint", "BaseSaveHandler"]


class BaseSaveHandler(metaclass=ABCMeta):
    """Base class for save handlers

    Methods to override:

    - :meth:`~ignite.handlers.checkpoint.BaseSaveHandler.__call__`
    - :meth:`~ignite.handlers.checkpoint.BaseSaveHandler.remove`


    Note:
        In derived class, please, make sure that in distributed configuration overridden methods are called by a single
        process. Distributed configuration on XLA devices should be treated slightly differently: for saving checkpoint
        with `xm.save() <https://pytorch.org/xla/release/1.5/index.html#torch_xla.core.xla_model.save>`_  all processes
        should pass into the function. Otherwise, application gets stuck.

    """

    @abstractmethod
    def __call__(self, checkpoint: Mapping, filename: str, metadata: Mapping | None = None) -> None:
        """Method to save `checkpoint` with `filename`. Additionally, metadata dictionary is provided.

        Metadata contains:

        - `basename`: file prefix (if provided) with checkpoint name, e.g. `epoch_checkpoint`.
        - `score_name`: score name if provided, e.g `val_acc`.
        - `priority`: checkpoint priority value (higher is better), e.g. `12` or `0.6554435`

        Args:
            checkpoint: checkpoint dictionary to save.
            filename: filename associated with checkpoint.
            metadata: metadata on checkpoint to save.

        """

    @abstractmethod
    def remove(self, filename: str) -> None:
        """Method to remove saved checkpoint.

        Args:
            filename: filename associated with checkpoint.

        """


class Checkpoint(Serializable):
    """Checkpoint handler can be used to periodically save and load objects which have attribute
    ``state_dict/load_state_dict``. This class can use specific save handlers to store on the disk or a cloud
    storage, etc. The Checkpoint handler (if used with :class:`~ignite.handlers.DiskSaver`) also handles automatically
    moving data on TPU to CPU before writing the checkpoint.

    Args:
        to_save: Dictionary with the objects to save. Objects should have implemented ``state_dict`` and
            ``load_state_dict`` methods. If contains objects of type torch `DistributedDataParallel`_ or
            `DataParallel`_, their internal wrapped model is automatically saved (to avoid additional key ``module.`` in
            the state dictionary).
        save_handler: String, function or callable object.
            used to save engine and other provided objects. Function receives two objects: checkpoint as a dictionary
            and filename. If ``save_handler`` is callable class, it can
            inherit of :class:`~ignite.handlers.checkpoint.BaseSaveHandler` and optionally implement ``remove`` method
            to keep a fixed number of saved checkpoints. In case if user needs to save engine's checkpoint on a disk,
            ``save_handler`` can be defined with :class:`~ignite.handlers.DiskSaver` or a string specifying
            directory name can be passed to ``save_handler``.
        filename_prefix: Prefix for the file name to which objects will be saved. See Note for details.
        score_function: If not None, it should be a function taking a single argument,
            :class:`~ignite.engine.engine.Engine` object, and returning a score (`float`). Objects with highest scores
            will be retained.
        score_name: If ``score_function`` not None, it is possible to store its value using
            ``score_name``. If ``score_function`` is None, ``score_name`` can be used alone to define ``score_function``
            as ``Checkpoint.get_default_score_fn(score_name)`` by default.
        n_saved: Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        global_step_transform: global step transform function to output a desired global step.
            Input of the function is ``(engine, event_name)``. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`~ignite.handlers.global_step_from_engine`.
        filename_pattern: If ``filename_pattern`` is provided, this pattern will be used to render
            checkpoint filenames. If the pattern is not defined, the default pattern would be used. See Note for
            details.
        include_self: Whether to include the `state_dict` of this object in the checkpoint. If `True`, then
            there must not be another object in ``to_save`` with key ``checkpointer``.
        greater_or_equal: if `True`, the latest equally scored model is stored. Otherwise, the first model.
            Default, `False`.
        save_on_rank: Which rank to save the objects on, in the distributed configuration. If ``save_handler`` is
            string or :class:`~pathlib.Path`, this is also used to instantiate a :class:`~ignite.handlers.DiskSaver`.

    .. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/
        torch.nn.parallel.DistributedDataParallel.html
    .. _DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    Note:
        This class stores a single file as a dictionary of provided objects to save.
        The filename is defined by ``filename_pattern`` and by default has the following
        structure: ``{filename_prefix}_{name}_{suffix}.{ext}`` where

        - ``filename_prefix`` is the argument passed to the constructor,
        - `name` is the key in ``to_save`` if a single object is to store, otherwise `name` is "checkpoint".
        - `suffix` is composed as following ``{global_step}_{score_name}={score}``.

    +----------------+------------+-----------------------+----------------------------------------------+
    | score_function | score_name | global_step_transform |  suffix                                      |
    +================+============+=======================+==============================================+
    |      None      |   None     |        None           | ``{engine.state.iteration}``                 |
    +----------------+------------+-----------------------+----------------------------------------------+
    |       X        |   None     |        None           | ``{score}``                                  |
    +----------------+------------+-----------------------+----------------------------------------------+
    |       X        |   None     |         X             | ``{global_step}_{score}``                    |
    +----------------+------------+-----------------------+----------------------------------------------+
    |       X        |    X       |         X             | ``{global_step}_{score_name}={score}``       |
    +----------------+------------+-----------------------+----------------------------------------------+
    |      None      |   None     |         X             | ``{global_step}``                            |
    +----------------+------------+-----------------------+----------------------------------------------+
    |       X        |    X       |        None           | ``{score_name}={score}``                     |
    +----------------+------------+-----------------------+----------------------------------------------+

    Above `global_step` defined by the output of `global_step_transform` and `score` defined by the output
    of `score_function`.

    By default, none of ``score_function``, ``score_name``, ``global_step_transform`` is defined, then suffix is
    setup by attached engine's current iteration. The filename will be
    `{filename_prefix}_{name}_{engine.state.iteration}.{ext}`.

    For example, ``score_name="neg_val_loss"`` and ``score_function`` that returns `-loss` (as objects with highest
    scores will be retained), then saved filename will be ``{filename_prefix}_{name}_neg_val_loss=-0.1234.pt``.

    Note:
        If ``filename_pattern`` is given, it will be used to render the filenames. ``filename_pattern`` is a string
        that can contain ``{filename_prefix}``, ``{name}``, ``{score}``, ``{score_name}`` and ``{global_step}`` as
        templates.

        For example, let ``filename_pattern="{global_step}-{name}-{score}.pt"`` then the saved filename will be
        ``30000-checkpoint-94.pt``

        **Warning:** Please, keep in mind that if filename collide with already used one to saved a checkpoint,
        new checkpoint will replace the older one. This means that filename like ``checkpoint.pt`` will be saved
        every call and will always be overwritten by newer checkpoints.

    Note:
        To get the last stored filename, handler exposes attribute ``last_checkpoint``:

        .. code-block:: python

            handler = Checkpoint(...)
            ...
            print(handler.last_checkpoint)
            > checkpoint_12345.pt

    Note:
        This class is distributed configuration-friendly: it is not required to instantiate the class in rank 0 only
        process. This class supports automatically distributed configuration and if used with
        :class:`~ignite.handlers.DiskSaver`, checkpoint is stored by rank 0 process.

    .. warning::

        When running on XLA devices or using :class:`~torch.distributed.optim.ZeroRedundancyOptimizer`, it
        should be run in all processes, otherwise application can get stuck while saving the checkpoint.

        .. code-block:: python

            # Wrong:
            # if idist.get_rank() == 0:
            #     handler = Checkpoint(...)
            #     trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)

            # Correct:
            handler = Checkpoint(...)
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)

    Examples:
        Attach the handler to make checkpoints during training:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Checkpoint

            trainer = ...
            model = ...
            optimizer = ...
            lr_scheduler = ...

            to_save = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'trainer': trainer}

            if (checkpoint_iters):
                # A: Output is "checkpoint_<iteration>.pt"
                handler = Checkpoint(
                    to_save, '/tmp/models', n_saved=2
                )
                trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)
            else:
                # B:Output is "checkpoint_<epoch>.pt"
                gst = lambda *_: trainer.state.epoch
                handler = Checkpoint(
                    to_save, '/tmp/models', n_saved=2, global_step_transform=gst
                )
                trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

            trainer.run(data_loader, max_epochs=6)
            > A: ["checkpoint_7000.pt", "checkpoint_8000.pt", ]
            > B: ["checkpoint_5.pt", "checkpoint_6.pt", ]

        Attach the handler to an evaluator to save best model during the training
        according to computed validation metric:

        .. code-block:: python

            from ignite.engine import Engine, Events
            from ignite.handlers import Checkpoint, global_step_from_engine

            trainer = ...
            evaluator = ...
            # Setup Accuracy metric computation on evaluator.
            # evaluator.state.metrics contain 'accuracy',
            # which will be used to define ``score_function`` automatically.
            # Run evaluation on epoch completed event
            # ...

            to_save = {'model': model}
            handler = Checkpoint(
                to_save, '/tmp/models',
                n_saved=2, filename_prefix='best',
                score_name="accuracy",
                global_step_transform=global_step_from_engine(trainer)
            )

            evaluator.add_event_handler(Events.COMPLETED, handler)

            trainer.run(data_loader, max_epochs=10)
            > ["best_model_9_accuracy=0.77.pt", "best_model_10_accuracy=0.78.pt", ]

        Customise the ``save_handler``:

        .. code-block:: python

            handler = Checkpoint(
                to_save, save_handler=DiskSaver('/tmp/models', create_dir=True, **kwargs), n_saved=2
            )

    .. versionchanged:: 0.4.3

        - Checkpoint can save model with same filename.
        - Added ``greater_or_equal`` argument.

    .. versionchanged:: 0.4.7

        - `score_name` can be used to define `score_function` automatically without providing `score_function`.
        - `save_handler` automatically saves to disk if path to directory is provided.
        - `save_on_rank` saves objects on this rank in a distributed configuration.
    """

    Item = NamedTuple("Item", [("priority", int), ("filename", str)])
    _state_dict_all_req_keys = ("_saved",)

    def __init__(
        self,
        to_save: Mapping,
        save_handler: str | Path | Callable | BaseSaveHandler,
        filename_prefix: str = "",
        score_function: Callable | None = None,
        score_name: str | None = None,
        n_saved: int | None = 1,
        global_step_transform: Callable | None = None,
        filename_pattern: str | None = None,
        include_self: bool = False,
        greater_or_equal: bool = False,
        save_on_rank: int = 0,
    ):
        if not isinstance(to_save, collections.Mapping):
            raise TypeError(f"Argument `to_save` should be a dictionary, but given {type(to_save)}")

        self._check_objects(to_save, "state_dict")

        if include_self:
            if not isinstance(to_save, collections.MutableMapping):
                raise TypeError(
                    f"If `include_self` is True, then `to_save` must be mutable, but given {type(to_save)}."
                )

            if "checkpointer" in to_save:
                raise ValueError(f"Cannot have key 'checkpointer' if `include_self` is True: {to_save}")

        if not (
            isinstance(save_handler, str)
            or isinstance(save_handler, Path)
            or callable(save_handler)
            or isinstance(save_handler, BaseSaveHandler)
        ):
            raise TypeError(
                "Argument `save_handler` should be a string or Path object or callable or inherit from BaseSaveHandler"
            )

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(f"global_step_transform should be a function, got {type(global_step_transform)} instead.")

        self.to_save = to_save
        self.filename_prefix = filename_prefix
        if isinstance(save_handler, str) or isinstance(save_handler, Path):
            self.save_handler = DiskSaver(save_handler, create_dir=True, save_on_rank=save_on_rank)
        else:
            self.save_handler = save_handler  # type: ignore
        self.score_function = score_function
        self.score_name = score_name
        if self.score_name is not None and self.score_function is None:
            self.score_function = self.get_default_score_fn(self.score_name)
        self.n_saved = n_saved
        self.ext = "pt"
        self.global_step_transform = global_step_transform
        self.filename_pattern = filename_pattern
        self._saved: list[Checkpoint.Item] = []
        self.include_self = include_self
        self.greater_or_equal = greater_or_equal
        self.save_on_rank = save_on_rank

    def _get_filename_pattern(self, global_step: int | None) -> str:
        if self.filename_pattern is None:
            filename_pattern = self.setup_filename_pattern(
                with_prefix=len(self.filename_prefix) > 0,
                with_score=self.score_function is not None,
                with_score_name=self.score_name is not None,
                with_global_step=global_step is not None,
            )
        else:
            filename_pattern = self.filename_pattern
        return filename_pattern

    def reset(self) -> None:
        """Method to reset saved checkpoint names.

        Use this method if the engine will independently run multiple times:

        .. code-block:: python

            from ignite.handlers import Checkpoint

            trainer = ...
            checkpointer = Checkpoint(...)

            trainer.add_event_handler(Events.COMPLETED, checkpointer)
            trainer.add_event_handler(Events.STARTED, checkpointer.reset)

            # fold 0
            trainer.run(data0, max_epochs=max_epochs)
            print("Last checkpoint:", checkpointer.last_checkpoint)

            # fold 1
            trainer.run(data1, max_epochs=max_epochs)
            print("Last checkpoint:", checkpointer.last_checkpoint)

        .. versionadded:: 0.4.3
        """
        self._saved = []

    @property
    def last_checkpoint(self) -> str | Path | None:
        if len(self._saved) < 1:
            return None

        if not isinstance(self.save_handler, DiskSaver):
            return self._saved[-1].filename

        return self.save_handler.dirname / self._saved[-1].filename

    def _check_lt_n_saved(self, or_equal: bool = False) -> bool:
        if self.n_saved is None:
            return True
        return len(self._saved) < self.n_saved + int(or_equal)

    def _compare_fn(self, new: int | float) -> bool:
        if self.greater_or_equal:
            return new >= self._saved[0].priority
        else:
            return new > self._saved[0].priority

    def __call__(self, engine: Engine) -> None:
        global_step = None
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, engine.last_event_name)

        if self.score_function is not None:
            priority = self.score_function(engine)
            if not isinstance(priority, numbers.Number):
                raise ValueError("Output of score_function should be a number")
        else:
            if global_step is None:
                global_step = engine.state.get_event_attrib_value(Events.ITERATION_COMPLETED)
            priority = global_step

        if self._check_lt_n_saved() or self._compare_fn(priority):
            priority_str = f"{priority}" if isinstance(priority, numbers.Integral) else f"{priority:.4f}"

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
                checkpoint = checkpoint[name]

            filename_pattern = self._get_filename_pattern(global_step)

            filename_dict = {
                "filename_prefix": self.filename_prefix,
                "ext": self.ext,
                "name": name,
                "score_name": self.score_name,
                "score": priority_str if (self.score_function is not None) else None,
                "global_step": global_step,
            }
            filename = filename_pattern.format(**filename_dict)

            metadata = {
                "basename": f"{self.filename_prefix}{'_' * int(len(self.filename_prefix) > 0)}{name}",
                "score_name": self.score_name,
                "priority": priority,
            }

            try:
                index = list(map(lambda it: it.filename == filename, self._saved)).index(True)
                to_remove = True
            except ValueError:
                index = 0
                to_remove = not self._check_lt_n_saved()

            if to_remove:
                item = self._saved.pop(index)
                if isinstance(self.save_handler, BaseSaveHandler):
                    self.save_handler.remove(item.filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda it: it[0])

            if self.include_self:
                # Now that we've updated _saved, we can add our own state_dict.
                checkpoint["checkpointer"] = self.state_dict()

            try:
                self.save_handler(checkpoint, filename, metadata)
            except TypeError:
                self.save_handler(checkpoint, filename)

    def _setup_checkpoint(self) -> dict[str, Any]:
        if self.to_save is not None:

            def func(obj: Any, **kwargs: Any) -> dict:
                if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    obj = obj.module
                return obj.state_dict()

            return cast(dict[str, Any], _tree_map(func, self.to_save))
        return {}

    @staticmethod
    def setup_filename_pattern(
        with_prefix: bool = True, with_score: bool = True, with_score_name: bool = True, with_global_step: bool = True
    ) -> str:
        """Helper method to get the default filename pattern for a checkpoint.

        Args:
            with_prefix: If True, the ``filename_prefix`` is added to the filename pattern:
                ``{filename_prefix}_{name}...``. Default, True.
            with_score: If True, ``score`` is added to the filename pattern: ``..._{score}.{ext}``.
                Default, True. At least one of ``with_score`` and ``with_global_step`` should be True.
            with_score_name: If True, ``score_name`` is added to the filename pattern:
                ``..._{score_name}={score}.{ext}``. If activated, argument ``with_score`` should be
                also True, otherwise an error is raised. Default, True.
            with_global_step: If True, ``{global_step}`` is added to the
                filename pattern: ``...{name}_{global_step}...``.
                At least one of ``with_score`` and ``with_global_step`` should be True.

        Examples:
            .. code-block:: python

                from ignite.handlers import Checkpoint

                filename_pattern = Checkpoint.setup_filename_pattern()

                print(filename_pattern)
                > "{filename_prefix}_{name}_{global_step}_{score_name}={score}.{ext}"

        .. versionadded:: 0.4.3
        """
        filename_pattern = "{name}"

        if not (with_global_step or with_score):
            raise ValueError("At least one of with_score and with_global_step should be True.")

        if with_global_step:
            filename_pattern += "_{global_step}"

        if with_score_name and with_score:
            filename_pattern += "_{score_name}={score}"
        elif with_score:
            filename_pattern += "_{score}"
        elif with_score_name:
            raise ValueError("If with_score_name is True, with_score should be also True")

        if with_prefix:
            filename_pattern = "{filename_prefix}_" + filename_pattern

        filename_pattern += ".{ext}"
        return filename_pattern

    @staticmethod
    def _check_objects(objs: Mapping, attr: str) -> None:
        def func(obj: Any, **kwargs: Any) -> None:
            if not hasattr(obj, attr):
                raise TypeError(f"Object {type(obj)} should have `{attr}` method")

        _tree_map(func, objs)

    @staticmethod
    def load_objects(to_load: Mapping, checkpoint: str | Mapping | Path, **kwargs: Any) -> None:
        """Helper method to apply ``load_state_dict`` on the objects from ``to_load`` using states from ``checkpoint``.

        Args:
            to_load: a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint: a path, a string filepath or a dictionary with state_dicts to load, e.g.
                `{"model": model_state_dict, "optimizer": opt_state_dict}`. If `to_load` contains a single key,
                then checkpoint can contain directly corresponding state_dict.
            kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)

        Examples:
            .. code-block:: python

                import tempfile
                from pathlib import Path

                import torch

                from ignite.engine import Engine, Events
                from ignite.handlers import ModelCheckpoint, Checkpoint

                trainer = Engine(lambda engine, batch: None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    handler = ModelCheckpoint(tmpdirname, 'myprefix', n_saved=None, create_dir=True)

                    model = torch.nn.Linear(3, 3)
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

                    to_save = {"weights": model, "optimizer": optimizer}

                    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, to_save)
                    trainer.run(torch.randn(10, 1), 5)

                    to_load = to_save
                    checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                    checkpoint = torch.load(checkpoint_fp)
                    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

                    # or using a string for checkpoint filepath

                    to_load = to_save
                    checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_fp)

        Note:
            If ``to_load`` contains objects of type torch `DistributedDataParallel`_ or
            `DataParallel`_, method ``load_state_dict`` will applied to their internal wrapped model (``obj.module``).

        .. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/
            torch.nn.parallel.DistributedDataParallel.html
        .. _DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
        """
        if not isinstance(checkpoint, (collections.Mapping, str, Path)):
            raise TypeError(f"Argument checkpoint should be a string or a dictionary, but given {type(checkpoint)}")

        Checkpoint._check_objects(to_load, "load_state_dict")

        if isinstance(checkpoint, (str, Path)):
            checkpoint_obj = torch.load(checkpoint)
        else:
            checkpoint_obj = checkpoint

        def _load_object(obj: Any, chkpt_obj: Any) -> None:
            if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                obj = obj.module

            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(chkpt_obj, **kwargs)
            else:
                obj.load_state_dict(chkpt_obj)

        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint_obj:
                _load_object(obj, checkpoint_obj)
                return

        _tree_apply2(_load_object, to_load, checkpoint_obj)

    def reload_objects(self, to_load: Mapping, load_kwargs: dict | None = None, **filename_components: Any) -> None:
        """Helper method to apply ``load_state_dict`` on the objects from ``to_load``. Filename components such as
        name, score and global state can be configured.

        Args:
            to_load: a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            load_kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)
            filename_components: Filename components used to define the checkpoint file path.
                Keyword arguments accepted are `name`, `score` and `global_state`.

        Examples:
            .. code-block:: python

                import tempfile

                import torch

                from ignite.engine import Engine, Events
                from ignite.handlers import ModelCheckpoint

                trainer = Engine(lambda engine, batch: None)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    checkpoint = ModelCheckpoint(tmpdirname, 'myprefix', n_saved=None, create_dir=True)

                    model = torch.nn.Linear(3, 3)
                    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

                    to_save = {"weights": model, "optimizer": optimizer}

                    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), checkpoint, to_save)
                    trainer.run(torch.randn(10, 1), 5)

                    to_load = to_save
                    # load checkpoint myprefix_checkpoint_40.pt
                    checkpoint.reload_objects(to_load=to_load, global_step=40)

        Note:
            If ``to_load`` contains objects of type torch `DistributedDataParallel`_ or
            `DataParallel`_, method ``load_state_dict`` will applied to their internal wrapped model (``obj.module``).

        Note:
            This method works only when the ``save_handler`` is of types string,
            :class:`~pathlib.Path` or :class:`~ignite.handlers.checkpoint.DiskSaver`.

        .. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/
            torch.nn.parallel.DistributedDataParallel.html
        .. _DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
        """
        if not isinstance(self.save_handler, DiskSaver):
            raise AttributeError(
                f"Checkpoint's `save_handler` should be of type `DiskSaver`, given {type(self.save_handler)}"
            )

        global_step = filename_components.get("global_step")

        filename_pattern = self._get_filename_pattern(global_step)

        checkpoint = self._setup_checkpoint()
        name = "checkpoint"
        if len(checkpoint) == 1:
            for k in checkpoint:
                name = k
        name = filename_components.get("name", name)
        score = filename_components.get("score")

        filename_dict = {
            "filename_prefix": self.filename_prefix,
            "ext": self.ext,
            "name": name,
            "score_name": self.score_name,
            "score": score,
            "global_step": global_step,
        }

        checkpoint_fp = filename_pattern.format(**filename_dict)

        path = self.save_handler.dirname / checkpoint_fp

        load_kwargs = {} if load_kwargs is None else load_kwargs

        Checkpoint.load_objects(to_load=to_load, checkpoint=path, **load_kwargs)

    def state_dict(self) -> OrderedDict:
        """Method returns state dict with saved items: list of ``(priority, filename)`` pairs.
        Can be used to save internal state of the class.
        """
        # TODO: this method should use _state_dict_all_req_keys
        return OrderedDict([("_saved", [(p, f) for p, f in self._saved])])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replaces internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "saved" key and list of ``(priority, filename)`` pairs as values.
        """
        super().load_state_dict(state_dict)
        self._saved = [Checkpoint.Item(p, f) for p, f in state_dict["_saved"]]

    @staticmethod
    def get_default_score_fn(metric_name: str, score_sign: float = 1.0) -> Callable:
        """Helper method to get default score function based on the metric name.

        Args:
            metric_name: metric name to get the value from ``engine.state.metrics``.
                Engine is the one to which :class:`~ignite.handlers.checkpoint.Checkpoint` handler is added.
            score_sign: sign of the score: 1.0 or -1.0. For error-like metrics, e.g. smaller is better,
                a negative score sign should be used (objects with larger score are retained). Default, 1.0.

        Examples:
            .. code-block:: python

                from ignite.handlers import Checkpoint

                best_acc_score = Checkpoint.get_default_score_fn("accuracy")

                best_model_handler = Checkpoint(
                    to_save, save_handler, score_name="val_accuracy", score_function=best_acc_score
                )
                evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

            Usage with error-like metric:

            .. code-block:: python

                from ignite.handlers import Checkpoint

                neg_loss_score = Checkpoint.get_default_score_fn("loss", -1.0)

                best_model_handler = Checkpoint(
                    to_save, save_handler, score_name="val_neg_loss", score_function=neg_loss_score
                )
                evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

        .. versionadded:: 0.4.3
        """
        if score_sign not in (1.0, -1.0):
            raise ValueError("Argument score_sign should be 1 or -1")

        def wrapper(engine: Engine) -> float:
            return score_sign * engine.state.metrics[metric_name]

        return wrapper


class DiskSaver(BaseSaveHandler):
    """Handler that saves input checkpoint on a disk.

    Args:
        dirname: Directory path where the checkpoint will be saved
        atomic: if True, checkpoint is serialized to a temporary file, and then
            moved to final destination, so that files are guaranteed to not be damaged
            (for example if exception occurs during saving).
        create_dir: if True, will create directory ``dirname`` if it doesnt exist.
        require_empty: If True, will raise exception if there are any files in the
            directory ``dirname``.
        save_on_rank: The rank on which the checkpoint will be saved. Used in distributed
            configuration.
        kwargs: Accepted keyword arguments for `torch.save` or `xm.save`.

    .. versionchanged:: 0.4.2
        Accept ``kwargs`` for `torch.save` or `xm.save`.

    .. versionchanged:: 0.4.10
        Argument ``save_on_rank`` was added to specify the rank on which checkpoint should be saved.
    """

    def __init__(
        self,
        dirname: str | Path,
        atomic: bool = True,
        create_dir: bool = True,
        require_empty: bool = True,
        save_on_rank: int = 0,
        **kwargs: Any,
    ):
        self.dirname = Path(dirname).expanduser()
        self._atomic = atomic
        self.save_on_rank = save_on_rank

        self._check_and_setup(self.dirname, create_dir, require_empty)
        self.kwargs = kwargs

    @staticmethod
    def _check_and_setup(dirname: Path, create_dir: bool, require_empty: bool) -> None:
        if create_dir:
            if not dirname.exists():
                dirname.mkdir(parents=True)
        # Ensure that dirname exists
        if not dirname.exists():
            raise ValueError(f"Directory path '{dirname}' is not found")

        if require_empty:
            matched = [fname for fname in os.listdir(dirname) if fname.endswith(".pt")]
            if len(matched) > 0:
                raise ValueError(
                    f"Files {matched} with extension '.pt' are already present "
                    f"in the directory {dirname}. If you want to use this "
                    "directory anyway, pass `require_empty=False`."
                    ""
                )

    def __call__(self, checkpoint: Mapping, filename: str, metadata: Mapping | None = None) -> None:
        path = self.dirname / filename

        self._save_func(checkpoint, path, torch.save)

    def _save_func(self, checkpoint: Mapping, path: Path, func: Callable) -> None:
        if not self._atomic:
            func(checkpoint, path, **self.kwargs)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)
            tmp_file = tmp.file
            tmp_name = tmp.name
            try:
                func(checkpoint, tmp_file, **self.kwargs)
            except BaseException:
                tmp.close()
                os.remove(tmp_name)
                raise
            else:
                tmp.close()
                os.replace(tmp.name, path)
                # append group/others read mode
                os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH)

    def remove(self, filename: str) -> None:
        path = self.dirname / filename
        path.unlink()


class ModelCheckpoint(Checkpoint):
    """ModelCheckpoint handler, inherits from :class:`~ignite.handlers.checkpoint.Checkpoint`, can be used
    to periodically save objects to disk only. If needed to store checkpoints to
    another storage type, please consider :class:`~ignite.handlers.checkpoint.Checkpoint`.
    It also provides `last_checkpoint` attribute to show the last saved checkpoint.

    This handler expects two arguments:

        - an :class:`~ignite.engine.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.

    See Examples for further details.

    .. warning::

        Behaviour of this class has been changed since v0.3.0.

        There is no more internal counter that has been used to indicate the number of save actions. User could
        see its value `step_number` in the filename, e.g. `{filename_prefix}_{name}_{step_number}.pt`. Actually,
        `step_number` is replaced by current engine's epoch if `score_function` is specified and current iteration
        otherwise.

        A single `pt` file is created instead of multiple files.

    Args:
        dirname: Directory path where objects will be saved.
        filename_prefix: Prefix for the file names to which objects will be saved. See Notes of
            :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
        score_function: if not None, it should be a function taking a single argument, an
            :class:`~ignite.engine.engine.Engine` object, and return a score (`float`). Objects with highest scores
            will be retained.
        score_name: if ``score_function`` not None, it is possible to store its value using
            `score_name`. See Examples of :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
        n_saved: Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        atomic: If True, objects are serialized to a temporary file, and then moved to final
            destination, so that files are guaranteed to not be damaged (for example if exception
            occurs during saving).
        require_empty: If True, will raise exception if there are any files starting with
            ``filename_prefix`` in the directory ``dirname``.
        create_dir: If True, will create directory ``dirname`` if it does not exist.
        global_step_transform: global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`~ignite.handlers.global_step_from_engine`.
        filename_pattern: If ``filename_pattern`` is provided, this pattern will be used to render
            checkpoint filenames. If the pattern is not defined, the default pattern would be used.
            See :class:`~ignite.handlers.checkpoint.Checkpoint` for details.
        include_self: Whether to include the `state_dict` of this object in the checkpoint. If `True`, then
            there must not be another object in ``to_save`` with key ``checkpointer``.
        greater_or_equal: if `True`, the latest equally scored model is stored. Otherwise, the first model.
            Default, `False`.
        save_on_rank: Which rank to save the objects on, in the distributed configuration. Used to
            instantiate a :class:`~ignite.handlers.DiskSaver` and is also passed to the parent class.
        kwargs: Accepted keyword arguments for `torch.save` or `xm.save` in `DiskSaver`.

    .. versionchanged:: 0.4.2
        Accept ``kwargs`` for `torch.save` or `xm.save`

    .. versionchanged:: 0.4.9
        Accept ``filename_pattern`` and ``greater_or_equal`` for parity
        with :class:`~ignite.handlers.checkpoint.Checkpoint`

    .. versionchanged:: 0.4.10
        Added `save_on_rank` arg to save objects on this rank in a distributed configuration

    Examples:
        .. testcode:: python

            import os
            from ignite.engine import Engine, Events
            from ignite.handlers import ModelCheckpoint
            from torch import nn
            trainer = Engine(lambda engine, batch: None)
            handler = ModelCheckpoint('/tmp/models', 'myprefix', n_saved=2, create_dir=True, require_empty=False)
            model = nn.Linear(3, 3)
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, {'mymodel': model})
            trainer.run([0, 1, 2, 3, 4], max_epochs=6)
            print(sorted(os.listdir('/tmp/models')))
            print(handler.last_checkpoint)

        .. testoutput:: python

            ['myprefix_mymodel_20.pt', 'myprefix_mymodel_30.pt']
            /tmp/models/myprefix_mymodel_30.pt
    """

    def __init__(
        self,
        dirname: str | Path,
        filename_prefix: str = "",
        score_function: Callable | None = None,
        score_name: str | None = None,
        n_saved: int | None = 1,
        atomic: bool = True,
        require_empty: bool = True,
        create_dir: bool = True,
        global_step_transform: Callable | None = None,
        filename_pattern: str | None = None,
        include_self: bool = False,
        greater_or_equal: bool = False,
        save_on_rank: int = 0,
        **kwargs: Any,
    ):
        disk_saver = DiskSaver(
            dirname,
            atomic=atomic,
            create_dir=create_dir,
            require_empty=require_empty,
            save_on_rank=save_on_rank,
            **kwargs,
        )

        super(ModelCheckpoint, self).__init__(
            to_save={},
            save_handler=disk_saver,
            filename_prefix=filename_prefix,
            score_function=score_function,
            score_name=score_name,
            n_saved=n_saved,
            global_step_transform=global_step_transform,
            filename_pattern=filename_pattern,
            include_self=include_self,
            greater_or_equal=greater_or_equal,
            save_on_rank=save_on_rank,
        )

    @property
    def last_checkpoint(self) -> str | Path | None:
        if len(self._saved) < 1:
            return None

        if not isinstance(self.save_handler, DiskSaver):
            raise RuntimeError(f"Internal error, save_handler should be DiskSaver, but has {type(self.save_handler)}.")

        return self.save_handler.dirname / self._saved[-1].filename

    def __call__(self, engine: Engine, to_save: Mapping):  # type: ignore
        if len(to_save) == 0:
            raise RuntimeError("No objects to checkpoint found.")

        self._check_objects(to_save, "state_dict")
        self.to_save = to_save
        super(ModelCheckpoint, self).__call__(engine)
