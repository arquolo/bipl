from __future__ import annotations

__all__ = ['as_actor', 'coroutine']

from collections import Counter, deque
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator
from functools import update_wrapper
from threading import Lock
from typing import TypeVar, cast

import wrapt

from ._more import _deiter

_T = TypeVar('_T')
_R = TypeVar('_R')
_F = TypeVar('_F', bound=Callable[..., Generator])


def coroutine(fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        coro = fn(*args, **kwargs)
        coro.send(None)
        return coro

    return cast(_F, update_wrapper(wrapper, fn))


class _Sync(wrapt.ObjectProxy):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self._self_lock = Lock()

    def _call(self, op, *args, **kwargs):
        with self._self_lock:
            return op(*args, **kwargs)

    def __next__(self):
        return self._call(self.__wrapped__.__next__)

    def send(self, item):
        return self._call(self.__wrapped__.send, item)

    def throw(self, typ, val=None, tb=None):
        return self._call(self.__wrapped__.throw, typ, val, tb)

    def close(self):
        return self._call(self.__wrapped__.close)


def threadsafe_iter(fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        gen = fn(*args, **kwargs)
        return _Sync(gen)

    return cast(_F, update_wrapper(wrapper, fn))


@threadsafe_iter
@coroutine
def summary() -> Generator[None, Hashable | None, None]:
    # ? delete this or find use case
    state = Counter[Hashable]()
    while True:
        key = yield
        if key is None:
            state.clear()
        else:
            state[key] += 1
            print(dict(state), flush=True, end='\r')


@threadsafe_iter
@coroutine
def as_actor(
    transform: Callable[[Iterable[_T]], Iterator[_R]]
) -> Generator[_R, _T, None]:
    buf = deque[_T]()
    gen = transform(_deiter(buf))  # infinite

    # shortcuts
    buf_append, gen_next = buf.append, gen.__next__

    x = yield  # type: ignore[misc]  # preseed coroutine
    while True:
        buf_append(x)
        x = yield gen_next()
