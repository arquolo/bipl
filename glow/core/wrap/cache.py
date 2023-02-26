from __future__ import annotations

__all__ = ['memoize']

# TODO: add case capacity=None for unbound cache

import argparse
import asyncio
import concurrent.futures as cf
import enum
import functools
from collections import Counter
from collections.abc import (Callable, Hashable, Iterable, KeysView,
                             MutableMapping)
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import RLock
from typing import (Any, ClassVar, Generic, Literal, NamedTuple, SupportsInt,
                    TypeVar, cast)
from weakref import WeakValueDictionary

from .._repr import si_bin
from .._sizeof import sizeof
from .reusable import make_loop
from .util import make_key


class _Empty(enum.Enum):
    token = 0


_T = TypeVar('_T')
_F = TypeVar('_F', bound=Callable)
_BatchedFn = TypeVar('_BatchedFn', bound=Callable[[list], Iterable])
_Policy = Literal['raw', 'lru', 'mru']
_KeyFn = Callable[..., Hashable]
_empty = _Empty.token


@dataclass(repr=False)
class _Node(Generic[_T]):
    __slots__ = ('value', 'size')
    value: _T
    size: int

    def __repr__(self) -> str:
        return repr(self.value)


class Stats(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__ = Counter()
        self.__dict__.update(**kwargs)

    def __getattr__(self, name: str) -> int:
        return self.__dict__[name]


class _IStore(Generic[_T]):
    def __len__(self) -> int:
        raise NotImplementedError

    def store_clear(self) -> None:
        raise NotImplementedError

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        raise NotImplementedError

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        raise NotImplementedError

    def can_swap(self, size: int) -> bool:
        raise NotImplementedError


@dataclass(repr=False)
class _InitializedStore:
    capacity: int
    size: int = 0
    stats: Stats = field(default_factory=Stats)


@dataclass(repr=False)
class _DictMixin(_InitializedStore, _IStore[_T]):
    lock: RLock = field(default_factory=RLock)

    def clear(self):
        with self.lock:
            self.store_clear()
            self.size = 0

    def keys(self) -> KeysView:
        raise NotImplementedError

    def __getitem__(self, key: Hashable) -> _T | _Empty:
        with self.lock:
            if node := self.store_get(key):
                self.stats.hits += 1
                return node.value
        return _empty

    def __setitem__(self, key: Hashable, value: _T) -> None:
        with self.lock:
            self.stats.misses += 1
            size = int(sizeof(value))
            if (self.size + size <= self.capacity) or self.can_swap(size):
                self.store_set(key, _Node(value, size))
                self.size += size


@dataclass(repr=False)
class _ReprMixin(_InitializedStore, _IStore[_T]):
    refs: ClassVar[MutableMapping[int, _ReprMixin]] = WeakValueDictionary()

    def __post_init__(self) -> None:
        self.refs[id(self)] = self

    def __repr__(self) -> str:
        args = [
            f'items={len(self)}',
            f'used={si_bin(self.size)}',
            f'total={si_bin(self.capacity)}',
        ]
        if any(vars(self.stats).values()):
            args.append(f'stats={self.stats}')
        return f'{type(self).__name__}({", ".join(args)})'

    @classmethod
    def status(cls) -> str:
        return '\n'.join(
            f'{id_:x}: {value!r}' for id_, value in sorted(cls.refs.items()))


@dataclass(repr=False)
class _Store(_ReprMixin[_T], _DictMixin[_T]):
    store: dict[Hashable, _Node[_T]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.store)

    def keys(self) -> KeysView:
        return self.store.keys()

    def store_clear(self) -> None:
        self.store.clear()

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        return self.store.get(key)

    def store_set(self, key: Hashable, node: _Node[_T]) -> None:
        self.store[key] = node


class _HeapCache(_Store[_T]):
    def can_swap(self, size: int) -> bool:
        return False


class _LruCache(_Store[_T]):
    drop_recent = False

    def store_get(self, key: Hashable) -> _Node[_T] | None:
        if node := self.store.pop(key, None):
            self.store[key] = node
            return node
        return None

    def can_swap(self, size: int) -> bool:
        if size > self.capacity:
            return False

        while self.size + size > self.capacity:
            if self.drop_recent:
                self.size -= self.store.popitem()[1].size
            else:
                self.size -= self.store.pop(next(iter(self.store))).size
            self.stats.dropped += 1
        return True


class _MruCache(_LruCache[_T]):
    drop_recent = True


# -------------------------------- wrapping --------------------------------


def _memoize(cache: _DictMixin, key_fn: _KeyFn, fn: _F) -> _F:
    def wrapper(*args, **kwargs):
        key = key_fn(*args, **kwargs)

        if (value := cache[key]) is not _empty:
            return value

        cache[key] = value = fn(*args, **kwargs)
        return value

    wrapper.cache = cache  # type: ignore[attr-defined]
    return cast(_F, functools.update_wrapper(wrapper, fn))


# ----------------------- wrapper with batching support ----------------------


class _Job(NamedTuple):
    token: Any
    future: asyncio.Future | cf.Future


def _dispatch(
    fn: Callable[[list], Iterable],
    evict: Callable[[Hashable], object],
    queue: dict[Hashable, _Job],
):
    jobs = {**queue}
    queue.clear()

    try:
        *values, = fn([job.token for job in jobs.values()])
        assert len(values) == len(jobs)

        for job, value in zip(jobs.values(), values):
            job.future.set_result(value)

    except BaseException as exc:  # noqa: PIE786
        for key, job in jobs.items():
            evict(key)
            job.future.set_exception(exc)


def _memoize_batched_aio(key_fn: _KeyFn, fn: _BatchedFn) -> _BatchedFn:
    assert callable(fn)
    cache: dict[Hashable, asyncio.Future] = {}
    queue: dict[Hashable, _Job] = {}
    loop = make_loop()

    def _load(token) -> asyncio.Future:
        key = key_fn(token)
        if result := cache.get(key):
            return result

        loop = asyncio.get_running_loop()
        cache[key] = future = loop.create_future()
        queue[key] = _Job(token, future)
        if len(queue) == 1:
            loop.call_soon(_dispatch, fn, cache, queue)

        return future

    async def _load_many(tokens: Iterable) -> tuple:
        return await asyncio.gather(*map(_load, tokens))

    def wrapper(tokens: Iterable) -> tuple:
        coro = _load_many(tokens)
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    wrapper.cache = cache  # type: ignore[attr-defined]
    return cast(_BatchedFn, functools.update_wrapper(wrapper, fn))


def _memoize_batched(key_fn: _KeyFn, fn: _BatchedFn) -> _BatchedFn:
    assert callable(fn)
    lock = RLock()
    cache: dict[Hashable, cf.Future] = {}
    queue: dict[Hashable, _Job] = {}

    def _load(stack: ExitStack, token: object) -> cf.Future:
        key = key_fn(token)
        with lock:
            if result := cache.get(key):
                return result

            cache[key] = future = cf.Future()  # type: ignore[var-annotated]
            queue[key] = _Job(token, future)
            if len(queue) == 1:
                stack.callback(_dispatch, fn, cache.pop, queue)

        return future

    def wrapper(tokens: Iterable) -> list:
        futs = []
        with ExitStack() as stack:
            futs += [_load(stack, token) for token in tokens]
        return [fut.result() for fut in futs]

    wrapper.cache = cache  # type: ignore[attr-defined]
    return cast(_BatchedFn, functools.update_wrapper(wrapper, fn))


# ----------------------------- factory wrappers -----------------------------


def memoize(
    capacity: SupportsInt,
    *,
    batched: bool = False,
    policy: _Policy = 'raw',
    key_fn: _KeyFn = make_key,
) -> Callable[[_F], _F] | Callable[[_BatchedFn], _BatchedFn]:
    """Returns dict-cache decorator.

    Parameters:
    - capacity - size in bytes.
    - policy - eviction policy, either "raw" (no eviction), or "lru"
      (evict oldest), or "mru" (evict most recent).
    """
    capacity = int(capacity)
    if not capacity:
        return lambda fn: fn

    caches: dict[str, type[_Store]] = {
        'raw': _HeapCache,
        'lru': _LruCache,
        'mru': _MruCache,
    }
    if (cache_cls := caches.get(policy)) is not None:
        if batched:
            return functools.partial(_memoize_batched, key_fn)
        return functools.partial(_memoize, cache_cls(capacity), key_fn)
    raise ValueError(f'Unknown policy: "{policy}". '
                     f'Only "{set(caches)}" are available')
