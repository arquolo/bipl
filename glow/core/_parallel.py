from __future__ import annotations

__all__ = ['buffered', 'map_n', 'starmap_n']

import atexit
import enum
import os
import signal
import sys
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator, Sized
from concurrent.futures import Executor, Future
from contextlib import ExitStack, contextmanager
from cProfile import Profile
from functools import partial
from itertools import chain, filterfalse, islice, starmap
from multiprocessing.managers import BaseManager
from operator import methodcaller
from pstats import Stats
from queue import Empty, SimpleQueue
from threading import Lock
from time import perf_counter, sleep
from typing import Protocol, TypeVar, cast

import loky

try:
    import psutil
except ImportError:
    psutil = None

from ._more import chunked
from ._reduction import move_to_shmem, reducers
from ._thread_quota import ThreadQuota

_T = TypeVar('_T')
_F = TypeVar('_F', bound=Future)

_NUM_CPUS = os.cpu_count() or 0
_NUM_CPUS = min(_NUM_CPUS, int(os.getenv('GLOW_CPUS', _NUM_CPUS)))
_IDLE_WORKER_TIMEOUT = 10
_GRANULAR_SCHEDULING = False  # TODO: investigate whether this improves load


class _Empty(enum.Enum):
    token = 0


_empty = _Empty.token

# ------------------- some useful interfaces and functions -------------------


class _Queue(Protocol[_T]):
    def get(self, block: bool = ..., timeout: float | None = ...) -> _T:
        ...

    def put(self, item: _T) -> None:
        ...


class _Event(Protocol):
    def is_set(self) -> bool:
        ...

    def set(self) -> None:  # noqa: A003
        ...


def _get_cpu_count_limits(upper_bound: int = sys.maxsize,
                          mp: bool = False) -> Iterator[int]:
    yield upper_bound
    yield os.cpu_count() or 1

    # Windows platform lacks memory overcommit, so it's sensitive to VMS growth
    if not mp or sys.platform != 'win32' or 'torch' not in sys.modules:
        return

    import torch
    if torch.version.cuda and torch.version.cuda >= '11.7.0':
        # It's expected that torch will fix .nv_fatb readonly flag in its DLLs
        # See https://stackoverflow.com/a/69489193/9868257
        return

    if psutil is None:
        warnings.warn('Max process count may be calculated incorrectly, '
                      'leading to application crash or even BSOD. '
                      'Install psutil to avoid that')
        return

    # Overcommit on Windows is forbidden, thus VMS planning is necessary
    vms: int = psutil.Process().memory_info().vms
    free_vms: int = psutil.virtual_memory().free + psutil.swap_memory().free
    yield free_vms // vms


def max_cpu_count(upper_bound: int = sys.maxsize, mp: bool = False) -> int:
    return min(_get_cpu_count_limits(upper_bound, mp))


_PATIENCE = 0.01


def _ki_call(fn: Callable[..., _T], *exc: type[BaseException]) -> _T:
    # See issues
    # https://bugs.python.org/issue29971
    # https://github.com/dask/dask/pull/2144#issuecomment-290556996
    # https://github.com/dask/dask/pull/2144/files
    while True:
        try:
            return fn(timeout=_PATIENCE)
        except exc:
            sleep(0)  # Force switch to another thread to proceed


if sys.platform == 'win32':
    from concurrent.futures import TimeoutError as _TimeoutError

    def _result(f: Future[_T]) -> _T:
        return _ki_call(f.result, _TimeoutError)
else:
    _result = Future.result


def _get_q_get(q: _Queue[_T]) -> Callable[[], _T]:
    if sys.platform != 'win32':
        return q.get
    return partial(_ki_call, q.get, Empty)


# ---------------------------- pool initialization ----------------------------


def _mp_profile():
    """Multiprocessed profiler"""
    prof = Profile()
    prof.enable()

    def _finalize(lines=50):
        prof.disable()
        with open(f'prof-{os.getpid()}.txt', 'w') as fp:  # noqa: PL123
            Stats(prof, stream=fp).sort_stats('cumulative').print_stats(lines)

    atexit.register(_finalize)


def _initializer():
    # `signal.signal` suppresses KeyboardInterrupt in child processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if os.environ.get('_GLOW_MP_PROFILE'):
        _mp_profile()


@contextmanager
def _get_executor(max_workers: int, mp: bool) -> Iterator[Executor]:
    if mp:
        processes: loky.ProcessPoolExecutor = loky.get_reusable_executor(
            max_workers,
            'loky_init_main',
            _IDLE_WORKER_TIMEOUT,
            job_reducers=reducers,
            result_reducers=reducers,
            initializer=_initializer,
        )
        # In generator 'finally' is not reliable enough, use atexit
        hook = atexit.register(processes.shutdown, kill_workers=True)
        yield processes
        atexit.unregister(hook)
    else:
        threads = ThreadQuota(max_workers)
        try:
            yield threads
        finally:
            is_success = sys.exc_info()[0] is None
            threads.shutdown(wait=is_success, cancel_futures=True)


def _get_manager(executor: Executor):
    if isinstance(executor, loky.ProcessPoolExecutor):  # noqa: R505
        return executor._context.Manager()
    else:
        from multiprocessing.dummy import Manager
        return Manager()


# -------- bufferize iterable by offloading to another thread/process --------


def _consume(iterable: Iterable[_T], q: _Queue[_T | _Empty], ev: _Event):
    try:
        for item in iterable:
            if ev.is_set():
                break
            q.put(item)
    finally:
        q.put(_empty)  # Signal to stop iteration
        q.put(_empty)  # Match last q.get


class buffered(Iterator[_T]):  # noqa: N801
    """
    Iterates over `iterable` in background thread with at most `latency`
    items ahead from caller
    """
    __slots__ = ('_get', '_task', 'close', '__weakref__')

    def __init__(self,
                 iterable: Iterable[_T],
                 /,
                 *,
                 latency: int = 2,
                 mp: bool | Executor = False):
        s = ExitStack()
        if isinstance(mp, Executor):
            executor = mp
        else:
            executor = s.enter_context(_get_executor(1, mp))

        mgr = _get_manager(executor)
        if isinstance(mgr, BaseManager):
            s.enter_context(mgr)

        ev: _Event = mgr.Event()
        q: _Queue[_T | _Empty] = mgr.Queue(latency)
        self._task = executor.submit(_consume, iterable, q, ev)  # type: ignore
        self._get = q_get = _get_q_get(q)

        # If main killed, wakes up consume to check ev
        # Otherwise collects 2nd _empty from q.
        # Called 2nd
        s.callback(q_get)

        # If main killed, signals consume to stop.
        # If consume is already stopped (on error or not), does nothing.
        # Called 1st
        s.callback(ev.set)

        self.close = weakref.finalize(self, s.close)

    def __iter__(self) -> Iterator[_T]:
        return self

    def __next__(self) -> _T:
        if self.close.alive:
            if (item := self._get()) is not _empty:
                return item

            self.close()
            _result(self._task)  # Throws exception if worker killed itself

        raise StopIteration


# ---------------------------- automatic batching ----------------------------


class _AutoSize:
    MIN_DURATION = 0.2
    MAX_DURATION = 2.0
    size: int = 1
    duration: float = 0.0

    def __init__(self) -> None:
        self.lock = Lock()

    def suggest(self) -> int:
        with self.lock:
            if 0 < self.duration < self.MIN_DURATION:
                self.size *= 2
                self.duration = 0.0

            elif self.duration > self.MAX_DURATION:
                size = int(2 * self.size * self.MIN_DURATION / self.duration)
                size = max(size, 1)
                if self.size != size:
                    self.duration = 0.0
                    self.size = size

            return self.size

    def update(self, start_time: float, fut: Future[Sized]):
        # Compute as soon as future became done, discard later if not needed
        duration = perf_counter() - start_time

        try:
            r = fut.result()  # Do not disturb Future._condition for nothing
        except BaseException:  # noqa: PIE786
            return

        with self.lock:
            if len(r) != self.size:
                return
            self.duration = ((0.8 * self.duration + 0.2 * duration)
                             if self.duration > 0 else duration)


# ---------------------- map iterable through function ----------------------


def _schedule(make_future: Callable[..., _F], args_zip: Iterable[Iterable],
              chunksize: int) -> Iterator[_F]:
    return starmap(make_future, chunked(args_zip, chunksize))


def _schedule_auto(
    make_future: Callable[..., _F],
    args_zip: Iterator[Iterable],
    max_workers: int,
) -> Iterator[_F]:
    # For the whole wave make futures with the same job size
    size = _AutoSize()
    while tuples := [*islice(args_zip, size.suggest() * max_workers)]:
        chunksize = len(tuples) // max_workers or 1
        for f in starmap(make_future, chunked(tuples, chunksize)):
            f.add_done_callback(partial(size.update, perf_counter()))
            yield f


def _schedule_auto_v2(make_future: Callable[..., _F],
                      args_zip: Iterator[Iterable]) -> Iterator[_F]:
    # Vary job size from future to future
    size = _AutoSize()
    while args := [*islice(args_zip, size.suggest())]:
        f = make_future(*args)
        f.add_done_callback(partial(size.update, perf_counter()))
        yield f


def _get_unwrap_iter(s: ExitStack, todo: set[Future[_T]],
                     get_f: Callable[[], Future[_T]],
                     fs: Iterator[Future[_T]]) -> Iterator[_T]:
    with s:
        while todo:
            f = get_f()
            todo.remove(f)

            yield _result(f)  # wait with timeout
            todo.update(islice(fs, 1))


def _unwrap(s: ExitStack, fs: Iterable[Future[_T]], qsize: int | None,
            order: bool) -> Iterator[_T]:
    q = SimpleQueue[Future[_T]]()
    q_put = q.put if order else methodcaller('add_done_callback', q.put)

    # As q.put always gives falsy None, filterfalse to call it as a side effect
    fs = filterfalse(q_put, fs)  # type: ignore[arg-type]
    try:
        todo = set(islice(fs, qsize))  # Prefetch
    except BaseException:
        s.close()  # Unwind here on an error
        raise
    else:
        return _get_unwrap_iter(s, todo, _get_q_get(q), fs)


def _batch_invoke(func: Callable[..., _T], *items: tuple) -> list[_T]:
    return [*starmap(func, items)]


def starmap_n(func: Callable[..., _T],
              iterable: Iterable[Iterable],
              /,
              *,
              max_workers: int | None = None,
              prefetch: int | None = 2,
              mp: bool = False,
              chunksize: int | None = None,
              order: bool = True) -> Iterator[_T]:
    """
    Equivalent to itertools.starmap(fn, iterable).

    Return an iterator whose values are returned from the function evaluated
    with an argument tuple taken from the given sequence.

    Options:

    - workers - Count of workers, by default all hardware threads are occupied.
    - prefetch - Extra count of scheduled jobs, if not set equals to infinity.
    - mp - Whether use processes or threads.
    - chunksize - The size of the chunks the iterable will be broken into
      before being passed to a processes. Estimated automatically.
      Ignored when threads are used.
    - order - Whether keep results order, or ignore it to increase performance.

    Unlike multiprocessing.Pool or concurrent.futures.Executor this one:

    - never deadlocks on any exception or Ctrl-C interruption.
    - accepts infinite iterables due to lazy task creation (option prefetch).
    - has single interface for both threads and processes.
    - TODO: serializes array-like data using out-of-band Pickle 5 buffers.
    - before first `__next__` call it submits at most `prefetch` jobs
      to warmup pool of workers.

    Notes:

    - To reduce latency set order to False, order of results will be arbitrary.
    - To increase CPU usage increase prefetch or set it to None.
    - In terms of CPU usage there's no difference between
      prefetch=None and order=False, so choose wisely.
    - Setting order to False makes no use of prefetch more than 0.

    """
    if max_workers is None:
        max_workers = max_cpu_count(_NUM_CPUS, mp)

    if not max_workers or not _NUM_CPUS:
        return starmap(func, iterable)  # Fallback to single thread

    if mp and chunksize is None and prefetch is None:
        raise ValueError('With multiprocessing either chunksize or prefetch '
                         'must be not None')

    if prefetch is not None:
        prefetch += max_workers

    it = iter(iterable)
    s = ExitStack()
    submit = s.enter_context(_get_executor(max_workers, mp)).submit

    if mp:
        func = move_to_shmem(func)
    else:
        chunksize = chunksize or 1

    if chunksize == 1:
        submit_one = cast(
            Callable[..., Future[_T]],
            partial(submit, func),
        )
        return _unwrap(s, starmap(submit_one, it), prefetch, order)

    submit_many = cast(
        Callable[..., Future[list[_T]]],
        partial(submit, _batch_invoke, func),
    )
    if chunksize is not None:
        # Fixed chunksize
        fs = _schedule(submit_many, it, chunksize)
    elif not _GRANULAR_SCHEDULING:
        # Dynamic chunksize scaling, submit tasks in waves
        fs = _schedule_auto(submit_many, it, max_workers)
    else:
        # Dynamic chunksize scaling
        fs = _schedule_auto_v2(submit_many, it)

    chunks = _unwrap(s, fs, prefetch, order)
    return chain.from_iterable(chunks)


def map_n(func: Callable[..., _T],
          /,
          *iterables: Iterable,
          max_workers: int | None = None,
          prefetch: int | None = 2,
          mp: bool = False,
          chunksize: int | None = None,
          order: bool = True) -> Iterator[_T]:
    """
    Returns iterator equivalent to map(func, *iterables).

    Make an iterator that computes the function using arguments from
    each of the iterables. Stops when the shortest iterable is exhausted.

    For extra options, see starmap_n, which is used under hood.
    """
    return starmap_n(
        func,
        zip(*iterables),
        max_workers=max_workers,
        prefetch=prefetch,
        mp=mp,
        chunksize=chunksize,
        order=order)
