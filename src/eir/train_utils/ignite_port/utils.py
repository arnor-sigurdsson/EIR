import collections.abc as collections
import inspect
from collections.abc import Callable
from typing import Any, Union, cast

import torch


def _check_signature(
    fn: Callable, fn_description: str, *args: Any, **kwargs: Any
) -> None:
    # if handler with filter, check the handler rather than the decorator
    if hasattr(fn, "_parent"):
        signature = inspect.signature(fn._parent())
    else:
        signature = inspect.signature(fn)
    try:  # try without engine
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        fn_params = list(signature.parameters)
        exception_msg = str(exc)
        passed_params = list(args) + list(kwargs)
        raise ValueError(
            f"Error adding {fn} '{fn_description}': "
            f"takes parameters {fn_params} but will be called with {passed_params}"
            f"({exception_msg})."
        )


def _to_hours_mins_secs(time_taken: float | int) -> tuple[int, int, float]:
    """Convert seconds to hours, mins, seconds and milliseconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return round(hours), round(mins), secs


class _CollectionItem:
    types_as_collection_item: tuple = (int, float, torch.Tensor)

    def __init__(self, collection: dict | list, key: int | str) -> None:
        if not isinstance(collection, (dict, list)):
            raise TypeError(
                f"Input type is expected to be a mapping or list, but got {type(collection)} "
                f"for input key '{key}'."
            )
        if isinstance(collection, list) and isinstance(key, str):
            raise ValueError("Key should be int for collection of type list")

        self.collection = collection
        self.key = key

    def load_value(self, value: Any) -> None:
        self.collection[self.key] = value  # type: ignore[index]

    def value(self) -> Any:
        return self.collection[self.key]  # type: ignore[index]

    @staticmethod
    def wrap(
        object: dict | list, key: int | str, value: Any
    ) -> Union[Any, "_CollectionItem"]:
        return (
            _CollectionItem(object, key)
            if value is None
            or isinstance(value, _CollectionItem.types_as_collection_item)
            else value
        )


def _tree_map(
    func: Callable,
    x: Any | collections.Sequence | collections.Mapping,
    key: int | str | None = None,
) -> Any | collections.Sequence | collections.Mapping:
    if isinstance(x, collections.Mapping):
        return cast(Callable, type(x))(
            {k: _tree_map(func, sample, key=k) for k, sample in x.items()}
        )
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return cast(Callable, type(x))(*(_tree_map(func, sample) for sample in x))
    if isinstance(x, collections.Sequence):
        return cast(Callable, type(x))(
            [_tree_map(func, sample, key=i) for i, sample in enumerate(x)]
        )
    return func(x, key=key)


def _tree_apply2(
    func: Callable,
    x: Any | list | dict,
    y: Any | collections.Sequence | collections.Mapping,
) -> None:
    if isinstance(x, dict) and isinstance(y, collections.Mapping):
        for k, v in x.items():
            if k not in y:
                raise ValueError(f"Key '{k}' from x is not found in y: {y.keys()}")
            _tree_apply2(func, _CollectionItem.wrap(x, k, v), y[k])
    elif isinstance(x, list) and isinstance(y, collections.Sequence):
        if len(x) != len(y):
            raise ValueError(
                f"Size of y: {len(y)} does not match the size of x: '{len(x)}'"
            )
        for i, (v1, v2) in enumerate(zip(x, y, strict=False)):
            _tree_apply2(func, _CollectionItem.wrap(x, i, v1), v2)
    else:
        return func(x, y)
