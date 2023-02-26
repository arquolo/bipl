from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import TypeVar, overload

from typing_extensions import ParamSpec

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_P = ParamSpec('_P')
_BatchedFn = TypeVar('_BatchedFn', bound=Callable[[list], Iterable])


def threadlocal(fn: Callable[_P, _T], *args: _P.args,
                **kwargs: _P.kwargs) -> Callable[[], _T]:
    ...


def interpreter_lock(timeout: float = ...) -> AbstractContextManager[None]:
    ...


def call_once(fn: _F) -> _F:
    ...


def shared_call(fn: _F) -> _F:
    ...


@overload
def streaming(*,
              batch_size: int,
              timeout: float = ...,
              workers: int = ...,
              pool_timeout: float = ...) -> Callable[[_BatchedFn], _BatchedFn]:
    ...


@overload
def streaming(func: _BatchedFn,
              *,
              batch_size: int,
              timeout: float = ...,
              workers: int = ...,
              pool_timeout: float = ...) -> _BatchedFn:
    ...
