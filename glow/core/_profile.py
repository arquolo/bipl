from __future__ import annotations

__all__ = ['memprof', 'time_this', 'timer']

import atexit
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate, count
from threading import get_ident
from time import perf_counter_ns, process_time_ns, thread_time_ns
from typing import TYPE_CHECKING

from wrapt import ObjectProxy

from ._repr import si, si_bin
from .debug import whereami

if TYPE_CHECKING:
    import psutil
    _THIS: psutil.Process | None

_THIS = None


@contextmanager
def memprof(name_or_callback: str | Callable[[float], object] | None = None,
            /) -> Iterator[None]:
    global _THIS
    if _THIS is None:
        import psutil
        _THIS = psutil.Process()

    init = _THIS.memory_info().rss
    try:
        yield
    finally:
        size = _THIS.memory_info().rss - init
        if callable(name_or_callback):
            name_or_callback(size)
        else:
            name = name_or_callback
            if name is None:
                name = f'{whereami(2, 1)} line'
            sign = '+' if size >= 0 else ''
            print(f'{name} done: {sign}{si_bin(size)}')


@contextmanager
def timer(name_or_callback: str | Callable[[int], object] | None = None,
          time: Callable[[], int] = perf_counter_ns,
          /) -> Iterator[None]:
    begin = time()
    try:
        yield
    finally:
        end = time()
        if callable(name_or_callback):
            name_or_callback(end - begin)
        else:
            name = name_or_callback or f'{whereami(2, 1)} line'
            print(f'{name} done in {si((end - begin) / 1e9)}s')


def _to_fname(obj) -> str:
    if not hasattr(obj, '__module__') or not hasattr(obj, '__qualname__'):
        obj = type(obj)
    if obj.__module__ == 'builtins':
        return obj.__qualname__
    return f'{obj.__module__}.{obj.__qualname__}'


class _Times(dict[int, int]):
    def add(self, value: int):
        idx = get_ident()
        self[idx] = self.get(idx, 0) + value

    def total(self) -> int:
        return sum(self.values())


class _Nlwp:
    __slots__ = ('_add_event', '_get_max')

    def __init__(self) -> None:
        events = deque[int]()
        self._add_event = events.append

        deltas = iter(events.popleft, None)
        totals = accumulate(deltas)
        maximums = accumulate(totals, max, initial=0)
        self._get_max = maximums.__next__

    def __enter__(self):
        self._add_event(+1)
        self._get_max()

    def __exit__(self, *args):
        self._add_event(-1)
        self._get_max()

    def max(self) -> int:
        self._add_event(0)
        return self._get_max()


@dataclass(frozen=True)
class _Stat:
    calls: count = field(default_factory=count)
    nlwp: _Nlwp = field(default_factory=_Nlwp)
    cpu_ns: _Times = field(default_factory=_Times)
    all_ns: _Times = field(default_factory=_Times)

    def __call__(self, op, *args, **kwargs):
        with self.nlwp, \
                timer(self.all_ns.add), \
                timer(self.cpu_ns.add, thread_time_ns):
            return op(*args, **kwargs)

    def stat(self) -> tuple[float, float, str] | None:
        if not (n := next(self.calls)):
            return None
        w = self.nlwp.max()
        t = self.cpu_ns.total() / 1e9
        a = self.all_ns.total() / 1e9

        tail = (f'{n} x {si(t / n)}s' +
                (f' @ {w}T' if w > 1 else '')) if n > 1 else ''
        return t, (a - t), tail


class _Proxy(ObjectProxy):
    def __init__(self, wrapped, wrapper):
        super().__init__(wrapped)
        self._self_wrapper = wrapper


class _TimedCall(_Proxy):
    def __get__(self, instance, owner):
        fn = self.__wrapped__.__get__(instance, owner)
        return _BoundTimedCall(fn, self._self_wrapper)

    def __call__(self, *args, **kwargs):
        next(self._self_wrapper.calls)
        r = self._self_wrapper(self.__wrapped__, *args, **kwargs)
        if isinstance(r, Iterator):
            return _TimedIter(r, self._self_wrapper)
        return r


class _BoundTimedCall(_TimedCall):
    def __get__(self, instance, owner):
        return self


class _TimedIter(_Proxy):
    def __iter__(self):
        return self

    def __next__(self):
        return self._self_wrapper(self.__wrapped__.__next__)

    def send(self, value):
        return self._self_wrapper(self.__wrapped__.send, value)

    def throw(self, typ, val=None, tb=None):
        return self._self_wrapper(self.__wrapped__.throw, typ, val, tb)

    def close(self):
        return self._self_wrapper(self.__wrapped__.close)


# Wall time, i.e. sum of per-thread times, excluding sleep
_start = process_time_ns()
_stats = defaultdict[str, _Stat](_Stat)


@atexit.register
def _print_stats(*names: str):
    all_busy = (process_time_ns() - _start + 1) / 1e9

    stats = []
    names = names or tuple(_stats)
    for name in names:
        if not (stat := _stats.pop(name, None)):
            continue
        if not (lines := stat.stat()):
            continue
        stats.append((*lines, name))

    for busy, idle, tail, name in sorted(stats):
        print(
            f'{busy/all_busy:6.2%}',
            f'{si(busy):>5s}s + {si(idle):>5s}s',
            name,
            tail,
            sep=' - ')


def time_this(fn=None, /, *, name: str | None = None):
    """Log function and/or generator timings at program exit"""
    if fn is None:
        return partial(time_this, name=name)

    if name is None:
        name = _to_fname(fn)

    fn.log_timing = partial(_print_stats, name)
    return _TimedCall(fn, _stats[name])
