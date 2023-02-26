from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar, overload

_F = TypeVar('_F', bound=Callable)


def memprof(name_or_callback: str | Callable[[float], object] | None = ...,
            /) -> AbstractContextManager[None]:
    ...


def timer(name_or_callback: str | Callable[[int], object] | None = ...,
          time: Callable[[], int] = ...,
          /) -> AbstractContextManager[None]:
    ...


@overload
def time_this(fn: _F, /, *, name: str | None = ...) -> _F:
    ...


@overload
def time_this(*, name: str | None = ...) -> Callable[[_F], _F]:
    ...
