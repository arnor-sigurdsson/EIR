__version__ = "0.13.1"

import re
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol, TextIO


class ShowWarningProtocol(Protocol):
    def __call__(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Optional[TextIO] = None,
        line: Optional[str] = None,
    ) -> None: ...


@dataclass
class WarningRule:
    module_pattern: str
    categories: Optional[list[type[Warning]]] = None
    message_pattern: Optional[str] = None


class WarningFilter:
    def __init__(self, rules: list[WarningRule]) -> None:
        self.rules = rules
        self._original_showwarning: ShowWarningProtocol = warnings.showwarning
        self._compiled_rules = [
            (
                re.compile(pattern=rule.module_pattern),
                (
                    re.compile(pattern=rule.message_pattern)
                    if rule.message_pattern
                    else None
                ),
                rule.categories,
            )
            for rule in rules
        ]

    def __call__(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Optional[TextIO] = None,
        line: Optional[str] = None,
    ) -> None:
        for module_pattern, message_pattern, categories in self._compiled_rules:
            if module_pattern.search(filename):
                if categories and category not in categories:
                    continue
                if message_pattern and not message_pattern.search(str(message)):
                    continue
                return

        self._original_showwarning(
            message=message,
            category=category,
            filename=filename,
            lineno=lineno,
            file=file,
            line=line,
        )

    def install(self) -> None:
        warnings.showwarning = self


warning_filter = WarningFilter(
    [
        WarningRule(
            module_pattern=r"ignite/handlers/checkpoint",
            categories=[DeprecationWarning],
        ),
        WarningRule(
            module_pattern=r"torch/distributed/optim",
            message_pattern=r"TorchScript.*support for functional optimizers",
        ),
    ]
)
warning_filter.install()
