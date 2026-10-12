"""Deprecated camelCase aliases kept so code written against guava-rt <= 0.5 keeps working."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def _warn(old: str, new: str) -> None:
    # stacklevel 3: skip this helper and the wrapper, pointing at the caller
    warnings.warn(f"{old} is deprecated; use {new} instead", DeprecationWarning, stacklevel=3)


def alias(func: F, old: str) -> F:
    """Expose ``func`` under its deprecated name ``old``, warning on every call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _warn(old, func.__name__)
        return func(*args, **kwargs)

    wrapper.__name__ = old
    wrapper.__qualname__ = func.__qualname__.rsplit(".", 1)[0] + "." + old
    return cast(F, wrapper)


def alias_property(new: str, old: str) -> property:
    """Read-only attribute ``old`` that forwards to ``new``, warning on every access."""

    def getter(self: Any) -> Any:
        _warn(old, new)
        return getattr(self, new)

    return property(getter)


def renamed_kwargs(**renames: str) -> Callable[[F], F]:
    """Accept deprecated keyword names (``old=new``) and forward them to their new names."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"got both {old!r} and its replacement {new!r}")
                    _warn(f"keyword argument {old!r}", repr(new))
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
